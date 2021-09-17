"""Tune-integrated training script for parallel experiments."""

import argparse
import collections
import glob
import os
import pprint
import time

import math
import numpy as np
import pandas as pd
import ray
from ray import tune
from ray.tune import logger as tune_logger
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import wandb

import common
import datasets
import estimators as estimators_lib
import experiments
import factorized_sampler
import fair_sampler
import join_utils
import made
import train_utils
import transformer
import utils
import multiprocessing as mp

os.environ['RAY_DEBUG_DISABLE_MEMORY_MONITOR']= '0.999'
parser = argparse.ArgumentParser()
os.environ["CUDA_VISIBLE_DEVICES"]= "0,4"
parser.add_argument('--run',
                    nargs='+',
                    default=experiments.TEST_CONFIGS.keys(),
                    type=str,
                    required=False,
                    help='List of experiments to run.')
# Resources per trial.
parser.add_argument('--cpus',
                    default=1,
                    type=int,
                    required=False,
                    help='Number of CPU cores per trial.')
parser.add_argument(
    '--gpus',
    default=1,
    type=int,
    required=False,
    help='Number of GPUs per trial. No effect if no GPUs are available.')

args = parser.parse_args()
torch.autograd.set_detect_anomaly(True)

class DataParallelPassthrough(torch.nn.DataParallel):
    """Wraps a model with nn.DataParallel and provides attribute accesses."""

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def TotalGradNorm(parameters, norm_type=2):
    total_norm = 0
    for p in parameters:
        if p.grad is None:
            continue
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item()**norm_type
    total_norm = total_norm**(1. / norm_type)
    return total_norm

def get_qerror(est_card, card):
    # if card == 0 and est_card != 0:
    #     return est_card
    # if card != 0 and est_card == 0:
    #     return card
    # if card == 0 and est_card == 0:
    #     return 1.0
    if est_card > card:
        if card > 0:
            return est_card / card
        else:
            return est_card
    else:
        if est_card > 0:
            return card / est_card
        else:
            return card

def run_epoch(split,
              model,
              opt,
              train_data,
              val_data=None,
              batch_size=100,
              upto=None,
              epoch_num=None,
              epochs=1,
              verbose=False,
              log_every=10,
              return_losses=False,
              table_bits=None,
              warmups=1000,
              loader=None,
              constant_lr=None,
              use_meters=True,
              summary_writer=None,
              lr_scheduler=None,
              custom_lr_lambda=None,
              label_smoothing=0.0,
              semi_train=False,
              estimator=None,
              query_list=None,
              card_list=None,
              q_weight=0):
    torch.set_grad_enabled(split == 'train')
    model.train() if split == 'train' else model.eval()
    dataset = train_data if split == 'train' else val_data
    losses = []

    torch.autograd.set_detect_anomaly(True)
    if loader is None:
        loader = data.DataLoader(dataset,
                                 batch_size=batch_size,
                                 shuffle=(split == 'train'))

    # How many orderings to run for the same batch?
    nsamples = 1
    if hasattr(model, 'orderings'):
        nsamples = len(model.orderings)
        if verbose:
            print('setting nsamples to', nsamples)

    dur_meter = train_utils.AverageMeter(
        'dur', lambda v: '{:.0f}s'.format(v), display_average=False)
    lr_meter = train_utils.AverageMeter('lr', ':.5f', display_average=False)
    tups_meter = train_utils.AverageMeter('tups',
                                          utils.HumanFormat,
                                          display_average=False)
    loss_meter = train_utils.AverageMeter('loss (bits/tup)', ':.2f')
    train_throughput = train_utils.AverageMeter('tups/s',
                                                utils.HumanFormat,
                                                display_average=False)
    batch_time = train_utils.AverageMeter('sgd_ms', ':3.1f')
    data_time = train_utils.AverageMeter('data_ms', ':3.1f')
    progress = train_utils.ProgressMeter(upto, [
        batch_time,
        data_time,
        dur_meter,
        lr_meter,
        tups_meter,
        train_throughput,
        loss_meter,
    ])

    if semi_train and split == 'train':
        q_bs = math.ceil(len(query_list) / upto)
        q_bs = int(q_bs)

        query_card_list = list(zip(query_list, card_list))

        np.random.shuffle(query_card_list)

    begin_time = t1 = time.time()
    for step, xb in enumerate(loader):
        data_time.update((time.time() - t1) * 1e3)

        if split == 'train':
            if isinstance(dataset, data.IterableDataset):
                # Can't call len(loader).
                global_steps = upto * epoch_num + step + 1
            else:
                global_steps = len(loader) * epoch_num + step + 1

            if constant_lr:
                lr = constant_lr
                for param_group in opt.param_groups:
                    param_group['lr'] = lr
            elif custom_lr_lambda:
                lr_scheduler = None
                lr = custom_lr_lambda(global_steps)
                for param_group in opt.param_groups:
                    param_group['lr'] = lr
            elif lr_scheduler is None:
                t = warmups
                if warmups < 1:  # A ratio.
                    t = int(warmups * upto * epochs)

                d_model = model.embed_size
                lr = (d_model**-0.5) * min(
                    (global_steps**-.5), global_steps * (t**-1.5))
                for param_group in opt.param_groups:
                    param_group['lr'] = lr
            else:
                # We'll call lr_scheduler.step() below.
                lr = opt.param_groups[0]['lr']

        if upto and step >= upto:
            break

        if isinstance(xb, list):
            # This happens if using data.TensorDataset.
            assert len(xb) == 1, xb
            xb = xb[0]

        xb = xb.float().to(train_utils.get_device(), non_blocking=True)

        # Forward pass, potentially through several orderings.
        xbhat = None
        model_logits = []
        num_orders_to_forward = 1
        if split == 'test' and nsamples > 1:
            # At test, we want to test the 'true' nll under all orderings.
            num_orders_to_forward = nsamples

        for i in range(num_orders_to_forward):
            if hasattr(model, 'update_masks'):
                # We want to update_masks even for first ever batch.
                model.update_masks()

            model_out = model(xb)
            model_logits.append(model_out)
            if xbhat is None:
                xbhat = torch.zeros_like(model_out)
            xbhat += model_out

        if num_orders_to_forward == 1:
            loss = model.nll(xbhat, xb, label_smoothing=label_smoothing).mean()
        else:
            # Average across orderings & then across minibatch.
            #
            #   p(x) = 1/N sum_i p_i(x)
            #   log(p(x)) = log(1/N) + log(sum_i p_i(x))
            #             = log(1/N) + logsumexp ( log p_i(x) )
            #             = log(1/N) + logsumexp ( - nll_i (x) )
            #
            # Used only at test time.
            logps = []  # [batch size, num orders]
            assert len(model_logits) == num_orders_to_forward, len(model_logits)
            for logits in model_logits:
                # Note the minus.
                logps.append(
                    -model.nll(logits, xb, label_smoothing=label_smoothing))
            logps = torch.stack(logps, dim=1)
            logps = logps.logsumexp(dim=1) + torch.log(
                torch.tensor(1.0 / nsamples, device=logps.device))
            loss = (-logps).mean()
        if semi_train and split == 'train':
            if step * q_bs >= len(query_list):
                q_c_tmp = query_card_list[step * q_bs - len(query_list): (step + 1) * q_bs - len(query_list)]
            elif (step + 1) * q_bs > len(query_list):
                q_c_tmp = query_card_list[step * q_bs:] + query_card_list[0: (step + 1) * q_bs - len(query_list)]
            else:
                q_c_tmp = query_card_list[step * q_bs: (step + 1) * q_bs]
            train_queries = [q for q, c in q_c_tmp]
            train_cards = [c for q, c in q_c_tmp]
            train_cards = np.array(train_cards)
            train_cards = torch.as_tensor(train_cards, dtype=torch.float32)
            train_cards = train_cards.to(train_utils.get_device())

            q_loss_array = []

            for i, query in enumerate(train_queries):
                cols, ops, vals = query
                card = train_cards[i]
                est_card = estimator.Query(cols, ops, vals)
                q_loss_array.append(get_qerror(est_card, card))

            q_loss = q_loss_array[0]
            for i in range(len(q_loss_array)-1):
                q_loss = q_loss + q_loss_array[i+1]

            q_loss = q_loss / len(q_loss_array)


            all_loss = loss + q_weight * q_loss
        else:
            all_loss = loss

        losses.append(loss.detach().item())
        if split == 'train':
            opt.zero_grad()
            if semi_train:
                all_loss.backward(retain_graph=True)
                #all_loss.backward()
            else:
                all_loss.backward()
            l2_grad_norm = TotalGradNorm(model.parameters())

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1, norm_type=2)
            opt.step()
            if lr_scheduler is not None:
                lr_scheduler.step()

            loss_bits = loss.item() / np.log(2)

            # Number of tuples processed in this epoch so far.
            ntuples = (step + 1) * batch_size
            if use_meters:
                dur = time.time() - begin_time
                lr_meter.update(lr)
                tups_meter.update(ntuples)
                loss_meter.update(loss_bits)
                dur_meter.update(dur)
                train_throughput.update(ntuples / dur)

            if summary_writer is not None:
                # wandb.log({
                #     'train/lr': lr,
                #     'train/tups': ntuples,
                #     'train/tups_per_sec': ntuples / dur,
                #     'train/nll': loss_bits,
                #     'train/global_step': global_steps,
                #     'train/l2_grad_norm': l2_grad_norm,
                # })
                summary_writer.add_scalar('train/lr',
                                          lr,
                                          global_step=global_steps)
                summary_writer.add_scalar('train/tups',
                                          ntuples,
                                          global_step=global_steps)
                summary_writer.add_scalar('train/tups_per_sec',
                                          ntuples / dur,
                                          global_step=global_steps)
                summary_writer.add_scalar('train/nll',
                                          loss_bits,
                                          global_step=global_steps)

            if step % log_every == 0:
                if table_bits:
                    print(
                        'Epoch {} Iter {}, {} entropy gap {:.4f} bits (loss {:.3f}, data {:.3f}) {:.5f} lr, {} tuples seen ({} tup/s)'
                        .format(
                            epoch_num, step, split,
                            loss.item() / np.log(2) - table_bits,
                            loss.item() / np.log(2), table_bits, lr,
                            utils.HumanFormat(ntuples),
                            utils.HumanFormat(ntuples /
                                              (time.time() - begin_time))))
                elif not use_meters:
                    print(
                        'Epoch {} Iter {}, {} loss {:.3f} bits/tuple, {:.5f} lr'
                        .format(epoch_num, step, split,
                                loss.item() / np.log(2), lr))

        if verbose:
            print('%s epoch average loss: %f' % (split, np.mean(losses)))

        batch_time.update((time.time() - t1) * 1e3)
        t1 = time.time()
        if split == 'train' and step % log_every == 0 and use_meters:
            progress.display(step)

    if return_losses:
        return losses
    return np.mean(losses)


def MakeMade(
        table,
        scale,
        layers,
        cols_to_train,
        seed,
        factor_table=None,
        fixed_ordering=None,
        special_orders=0,
        order_content_only=True,
        order_indicators_at_front=True,
        inv_order=True,
        residual=True,
        direct_io=True,
        input_encoding='embed',
        output_encoding='embed',
        embed_size=32,
        dropout=True,
        grouped_dropout=False,
        per_row_dropout=False,
        fixed_dropout_ratio=False,
        input_no_emb_if_leq=False,
        embs_tied=True,
        resmade_drop_prob=0.,
        # Join specific:
        num_joined_tables=None,
        table_dropout=None,
        table_num_columns=None,
        table_column_types=None,
        table_indexes=None,
        table_primary_index=None,
        # DMoL
        num_dmol=0,
        scale_input=False,
        dmol_cols=[]):
    dmol_col_indexes = []
    if dmol_cols:
        for i in range(len(cols_to_train)):
            if cols_to_train[i].name in dmol_cols:
                dmol_col_indexes.append(i)

    model = made.MADE(
        nin=len(cols_to_train),
        hidden_sizes=[scale] *
        layers if layers > 0 else [512, 256, 512, 128, 1024],
        nout=sum([c.DistributionSize() for c in cols_to_train]),
        num_masks=max(1, special_orders),
        natural_ordering=True,
        input_bins=[c.DistributionSize() for c in cols_to_train],
        do_direct_io_connections=direct_io,
        input_encoding=input_encoding,
        output_encoding=output_encoding,
        embed_size=embed_size,
        input_no_emb_if_leq=input_no_emb_if_leq,
        embs_tied=embs_tied,
        residual_connections=residual,
        factor_table=factor_table,
        seed=seed,
        fixed_ordering=fixed_ordering,
        resmade_drop_prob=resmade_drop_prob,

        # Wildcard skipping:
        dropout_p=dropout,
        fixed_dropout_p=fixed_dropout_ratio,
        grouped_dropout=grouped_dropout,
        learnable_unk=True,
        per_row_dropout=per_row_dropout,

        # DMoL
        num_dmol=num_dmol,
        scale_input=scale_input,
        dmol_col_indexes=dmol_col_indexes,

        # Join support.
        num_joined_tables=num_joined_tables,
        table_dropout=table_dropout,
        table_num_columns=table_num_columns,
        table_column_types=table_column_types,
        table_indexes=table_indexes,
        table_primary_index=table_primary_index,
    ).to(train_utils.get_device())

    if special_orders > 0:
        orders = []

        if order_content_only:
            print('Leaving out virtual columns from orderings')
            cols = [c for c in cols_to_train if not c.name.startswith('__')]
            inds_cols = [c for c in cols_to_train if c.name.startswith('__in_')]
            num_indicators = len(inds_cols)
            num_content, num_virtual = len(cols), len(cols_to_train) - len(cols)

            # Data: { content }, { indicators }, { fanouts }.
            for i in range(special_orders):
                rng = np.random.RandomState(i + 1)
                content = rng.permutation(np.arange(num_content))
                inds = rng.permutation(
                    np.arange(num_content, num_content + num_indicators))
                fanouts = rng.permutation(
                    np.arange(num_content + num_indicators, len(cols_to_train)))

                if order_indicators_at_front:
                    # Model: { indicators }, { content }, { fanouts },
                    # permute each bracket independently.
                    order = np.concatenate(
                        (inds, content, fanouts)).reshape(-1,)
                else:
                    # Model: { content }, { indicators }, { fanouts }.
                    # permute each bracket independently.
                    order = np.concatenate(
                        (content, inds, fanouts)).reshape(-1,)
                assert len(np.unique(order)) == len(cols_to_train), order
                orders.append(order)
        else:
            # Permute content & virtual columns together.
            for i in range(special_orders):
                orders.append(
                    np.random.RandomState(i + 1).permutation(
                        np.arange(len(cols_to_train))))

        if factor_table:
            # Correct for subvar ordering.
            for i in range(special_orders):
                # This could have [..., 6, ..., 4, ..., 5, ...].
                # So we map them back into:
                # This could have [..., 4, 5, 6, ...].
                # Subvars have to be in order and also consecutive
                order = orders[i]
                for orig_col, sub_cols in factor_table.fact_col_mapping.items():
                    first_subvar_index = cols_to_train.index(sub_cols[0])
                    print('Before', order)
                    for j in range(1, len(sub_cols)):
                        subvar_index = cols_to_train.index(sub_cols[j])
                        order = np.delete(order,
                                          np.argwhere(order == subvar_index))
                        order = np.insert(
                            order,
                            np.argwhere(order == first_subvar_index)[0][0] + j,
                            subvar_index)
                    orders[i] = order
                    print('After', order)

        print('Special orders', np.array(orders))

        if inv_order:
            for i, order in enumerate(orders):
                orders[i] = np.asarray(utils.InvertOrder(order))
            print('Inverted special orders:', orders)

        model.orderings = orders

    return model


class NeuroCard(tune.Trainable):

    def _setup(self, config):
        self.config = config
        print('NeuroCard config:')
        pprint.pprint(config)
        os.chdir(config['cwd'])
        for k, v in config.items():
            setattr(self, k, v)

        if config['__gpu'] == 0:
            torch.set_num_threads(config['__cpu'])

        # W&B.
        # Do wandb.init() after the os.chdir() above makes sure that the Git
        # diff file (diff.patch) is w.r.t. the directory where this file is in,
        # rather than w.r.t. Ray's package dir.

        # wandb_project = config['__run']
        # wandb.init(name=os.path.basename(
        #     self.logdir if self.logdir[-1] != '/' else self.logdir[:-1]),
        #            sync_tensorboard=True,
        #            config=config,
        #            project=wandb_project)

        self.epoch = 0

        if isinstance(self.join_tables, int):
            # Hack to support training single-model tables.
            sorted_table_names = sorted(
                list(datasets.JoinOrderBenchmark.GetJobLightJoinKeys().keys()))
            self.join_tables = [sorted_table_names[self.join_tables]]

        # Try to make all the runs the same, except for input orderings.
        torch.manual_seed(0)
        np.random.seed(0)

        # Common attributes.
        self.loader = None
        self.join_spec = None
        join_iter_dataset = None
        table_primary_index = None

        # New datasets should be loaded here.
        assert self.dataset in ['imdb']
        if self.dataset == 'imdb':
            print('Training on Join({})'.format(self.join_tables))
            loaded_tables = []
            for t in self.join_tables:
                print('Loading', t)
                table = datasets.LoadImdb(t, use_cols=self.use_cols)
                table.data.info()
                loaded_tables.append(table)
            if len(self.join_tables) > 1:
                join_spec, join_iter_dataset, loader, table = self.MakeSamplerDatasetLoader(
                    loaded_tables)

                self.join_spec = join_spec
                self.train_data = join_iter_dataset
                self.loader = loader

                table_primary_index = [t.name for t in loaded_tables
                                      ].index('title')

                table.cardinality = datasets.JoinOrderBenchmark.GetFullOuterCardinalityOrFail(
                    self.join_tables)
                self.train_data.cardinality = table.cardinality

                print('rows in full join', table.cardinality,
                      'cols in full join', len(table.columns), 'cols:', table)
            else:
                # Train on a single table.
                table = loaded_tables[0]

            self.loaded_tables = loaded_tables
        if self.dataset != 'imdb' or len(self.join_tables) == 1:
            table.data.info()
            self.train_data = self.MakeTableDataset(table)

        self.table = table
        # Provide true cardinalities in a file or implement an oracle CardEst.
        self.oracle = None
        self.table_bits = 0
        # A fixed ordering?
        self.fixed_ordering = self.MakeOrdering(table.columns)

        model = self.MakeModel(self.table,
                               self.train_data,
                               table_primary_index=table_primary_index)

        self.content_dics, self.indicator_dics, self.fanout_dics = self.MakeIndexRecords(self.table,
                                                                          self.train_data,
                                                                          table_primary_index=table_primary_index)
        print (self.content_dics)
        # NOTE: ReportModel()'s returned value is the true model size in
        # megabytes containing all all *trainable* parameters.  As impl
        # convenience, the saved ckpts on disk have slightly bigger footprint
        # due to saving non-trainable constants (the masks in each layer) as
        # well.  They can be deterministically reconstructed based on RNG seeds
        # and so should not be counted as model size.
        self.mb = train_utils.ReportModel(model)
        if not isinstance(model, transformer.Transformer):
            print('applying train_utils.weight_init()')
            model.apply(train_utils.weight_init)
        self.model = model

        if self.use_data_parallel:
            self.model = DataParallelPassthrough(self.model)

        # wandb.watch(model, log='all')

        if self.use_transformer:
            opt = torch.optim.Adam(
                list(model.parameters()),
                2e-4,
                # betas=(0.9, 0.98),  # B in Lingvo; in Trfmr paper.
                betas=(0.9, 0.997),  # A in Lingvo.
                eps=1e-9,
            )
        else:
            if self.optimizer == 'adam':
                opt = torch.optim.Adam(list(model.parameters()), 2e-4)
            else:
                print('Using Adagrad')
                opt = torch.optim.Adagrad(list(model.parameters()), 2e-4)
        print('Optimizer:', opt)
        self.opt = opt

        total_steps = self.epochs * self.max_steps
        if self.lr_scheduler == 'CosineAnnealingLR':
            # Starts decaying to 0 immediately.
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, total_steps)
        elif self.lr_scheduler == 'OneCycleLR':
            # Warms up to max_lr, then decays to ~0.
            self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                opt, max_lr=2e-3, total_steps=total_steps)
        elif self.lr_scheduler is not None and self.lr_scheduler.startswith(
                'OneCycleLR-'):
            warmup_percentage = float(self.lr_scheduler.split('-')[-1])
            # Warms up to max_lr, then decays to ~0.
            self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                opt,
                max_lr=2e-3,
                total_steps=total_steps,
                pct_start=warmup_percentage)
        elif self.lr_scheduler is not None and self.lr_scheduler.startswith(
                'wd_'):
            # Warmups and decays.
            splits = self.lr_scheduler.split('_')
            assert len(splits) == 3, splits
            lr, warmup_fraction = float(splits[1]), float(splits[2])
            self.custom_lr_lambda = train_utils.get_cosine_learning_rate_fn(
                total_steps,
                learning_rate=lr,
                min_learning_rate_mult=1e-5,
                constant_fraction=0.,
                warmup_fraction=warmup_fraction)
        else:
            assert self.lr_scheduler is None, self.lr_scheduler

        self.tbx_logger = tune_logger.TBXLogger(self.config, self.logdir)

        if self.checkpoint_to_load:
            self.LoadCheckpoint()

        self.loaded_queries = None
        self.oracle_cards = None

        self.loaded_job_light_queries = None
        self.job_light_oracle_cards = None

        if self.dataset == 'imdb' and len(self.join_tables) > 1:
            queries_job_format = utils.JobToQuery(self.queries_csv)

            self.loaded_queries, self.oracle_cards = utils.UnpackQueries(
                    self.table, queries_job_format)

        if config['__gpu'] == 0:
            print('CUDA not available, using # cpu cores for intra-op:',
                  torch.get_num_threads(), '; inter-op:',
                  torch.get_num_interop_threads())


        # For sampled data
        self.sampled_tables = {}
        self.sampled_views = []
        for i in range(len(self.join_tables)-1):
            self.sampled_views.append({})

        self.sampled_table_idx = table_primary_index
        self.sampled_view_idx = []

        self.sampled_full_view = {}
        self.sampled_full_view_fanout = {}
        self.total_tuple_sampled = 0
        self.sampled_group_dict = {}

        for i in range(len(self.join_tables)):
            if i != table_primary_index:
                self.sampled_view_idx.append([table_primary_index, i])

        self.gt_caches = {}
        self.unique_rows = None

        self.sampled_table_nums = [0] * len(self.join_tables)


    def LoadCheckpoint(self):
        all_ckpts = glob.glob(self.checkpoint_to_load)
        msg = 'No ckpt found or use tune.grid_search() for >1 ckpts.'
        assert len(all_ckpts) == 1, msg
        loaded = torch.load(all_ckpts[0], map_location=torch.device('cpu'))
        try:
            self.model.load_state_dict(loaded)
        except RuntimeError as e:
            # Backward compatibility: renaming.
            def Rename(state_dict):
                new_state_dict = collections.OrderedDict()
                for key, value in state_dict.items():
                    new_key = key
                    if key.startswith('embedding_networks'):
                        new_key = key.replace('embedding_networks',
                                              'embeddings')
                    new_state_dict[new_key] = value
                return new_state_dict

            loaded = Rename(loaded)

            modules = list(self.model.net.children())
            if len(modules) < 2 or type(modules[-2]) != nn.ReLU:
                raise e
            # Try to load checkpoints created prior to a 7/28/20 fix where
            # there's an activation missing.
            print('Try loading without ReLU before output layer.')
            modules.pop(-2)
            self.model.net = nn.Sequential(*modules)
            self.model.load_state_dict(loaded)

        print('Loaded ckpt from', all_ckpts[0])

    def MakeTableDataset(self, table):
        train_data = common.TableDataset(table)
        if self.factorize:
            train_data = common.FactorizedTable(
                train_data, word_size_bits=self.word_size_bits)
        return train_data

    def MakeSamplerDatasetLoader(self, loaded_tables):
        assert self.sampler in ['fair_sampler',
                                'factorized_sampler'], self.sampler
        join_spec = join_utils.get_join_spec(self.__dict__)
        if self.sampler == 'fair_sampler':
            klass = fair_sampler.FairSamplerIterDataset
        else:
            klass = factorized_sampler.FactorizedSamplerIterDataset
        join_iter_dataset = klass(
            loaded_tables,
            join_spec,
            sample_batch_size=self.sampler_batch_size,
            disambiguate_column_names=True,
            # Only initialize the sampler if training.
            initialize_sampler=self.checkpoint_to_load is None,
            save_samples=self._save_samples,
            load_samples=self._load_samples)

        table = common.ConcatTables(loaded_tables,
                                    self.join_keys,
                                    sample_from_join_dataset=join_iter_dataset)

        if self.factorize:
            join_iter_dataset = common.FactorizedSampleFromJoinIterDataset(
                join_iter_dataset,
                base_table=table,
                factorize_blacklist=self.dmol_cols if self.num_dmol else
                self.factorize_blacklist if self.factorize_blacklist else [],
                word_size_bits=self.word_size_bits,
                factorize_fanouts=self.factorize_fanouts)

        loader = data.DataLoader(join_iter_dataset,
                                 batch_size=self.bs,
                                 num_workers=self.loader_workers,
                                 worker_init_fn=lambda worker_id: np.random.
                                 seed(np.random.get_state()[1][0] + worker_id),
                                 pin_memory=True)
        return join_spec, join_iter_dataset, loader, table

    # def MakeOrdering(self, columns):
    #     fixed_ordering = None
    #     if self.dataset != 'imdb' and self.special_orders <= 1:
    #         fixed_ordering = list(range(len(columns)))
    #         return fixed_ordering
    #
    #     if self.order is not None:
    #         print('Using passed-in order:', self.order)
    #         fixed_ordering = self.order
    #         return fixed_ordering
    #
    #     cols = [c for c in columns if not c.name.startswith('__')]
    #     inds_cols = [c for c in columns if c.name.startswith('__in_')]
    #     num_indicators = len(inds_cols)
    #     num_content, num_virtual = len(cols), len(columns) - len(cols)
    #
    #     # Data: { content }, { indicators }, { fanouts }.
    #
    #     content = np.arange(num_content)
    #     inds = np.arange(num_content, num_content + num_indicators)
    #     fanouts = np.arange(num_content + num_indicators, len(columns))
    #
    #     if order_indicators_at_front:
    #         # Model: { indicators }, { content }, { fanouts },
    #         order = np.concatenate(
    #             (inds, content, fanouts)).reshape(-1, )
    #
    #     assert len(np.unique(order)) == len(cols_to_train), order
    #     orders.append(order)
    #     return fixed_ordering

    def MakeOrdering(self, table):
        fixed_ordering = None
        if self.dataset != 'imdb' and self.special_orders <= 1:
            fixed_ordering = list(range(len(table.columns)))

        if self.order is not None:
            print('Using passed-in order:', self.order)
            fixed_ordering = self.order

        if self.order_seed is not None:
            if self.order_seed == 'reverse':
                fixed_ordering = fixed_ordering[::-1]
            else:
                rng = np.random.RandomState(self.order_seed)
                rng.shuffle(fixed_ordering)
            print('Using generated order:', fixed_ordering)
        return fixed_ordering

    def MakeIndexRecords(self, table, train_data, table_primary_index=None):
        cols_to_train = table.columns
        if self.factorize:
            cols_to_train = train_data.columns

        num_tables = len(self.join_tables)

        fanout_dics = []
        for i in range (num_tables):
            fanout_dics.append([])
        indicator_dics = [None] * num_tables
        content_dics = []
        for i in range (num_tables):
            content_dics.append([])

        table_num_columns = table_column_types = table_indexes = None
        if isinstance(train_data, (common.SamplerBasedIterDataset,
                                   common.FactorizedSampleFromJoinIterDataset)):
            table_num_columns = train_data.table_num_columns
            table_column_types = train_data.combined_columns_types
            table_indexes = train_data.table_indexes
            print('table_num_columns', table_num_columns)
            print('table_column_types', table_column_types)
            print('table_indexes', table_indexes)
            print('table_primary_index', table_primary_index)

            print(cols_to_train)
            print(len(cols_to_train))
            for col_id, col in enumerate(cols_to_train):
                col_type = table_column_types[col_id]
                table_id = int(table_indexes[col_id])

                table_name = self.join_tables[table_id]
                table_key = table_name + '.csv'
                cols_candidate = datasets.JoinOrderBenchmark.BASE_TABLE_PRED_COLS[table_key]

                if col_type == common.TYPE_NORMAL_ATTR:
                    if table_id == 4 and col.name.split(':')[-1] in ['kind_id', 'production_year']:
                        content_dics[table_id].append(col_id)
                    elif table_id == 0 and col.name.split(':')[-1] in ['role_id', 'person_id_fact_0', 'person_id_fact_1']:
                        content_dics[table_id].append(col_id)
                    elif table_id == 1 and col.name.split(':')[-1] in ['company_id_fact_0', 'company_id_fact_1', 'company_type_id']:
                        content_dics[table_id].append(col_id)
                    elif table_id == 2 and col.name.split(':')[-1] == 'info_type_id':
                        content_dics[table_id].append(col_id)
                    elif table_id == 3 and col.name.split(':')[-1] in ['keyword_id_fact_0', 'keyword_id_fact_1']:
                        content_dics[table_id].append(col_id)
                    elif table_id == 5 and col.name.split(':')[-1] == 'info_type_id':
                        content_dics[table_id].append(col_id)
                elif col_type == common.TYPE_INDICATOR:
                    indicator_dics[table_id] = col_id
                else:
                    fanout_dics[table_id].append(col_id)

            # print("printing all table fanout columns: ")
            # for table_id in range(num_tables):
            #     print("table id: {}, fanout columns: {}".format(table_id, fanout_dics[table_id]))

        return content_dics, indicator_dics, fanout_dics

    def ComputeCE(self, gt_table, gen_table, gt_caches, eps=1e-9):
        col_names = gt_table.columns.tolist()
        unique_rows = list(gt_table.groupby(col_names).groups)
        ce = 0.

        if not len(gt_caches):
            gt_counts_df = gt_table.groupby(col_names).size().reset_index(name='counts')
        gen_counts_df = gen_table.groupby(col_names).size().reset_index(name='counts')


        for row in unique_rows:
            value = list(row)
            value_str = ','.join(value)

            if value_str in gt_caches:
                gt_prob = gt_caches[value_str]
            else:
                gt_prob = gt_counts_df[gt_counts_df[col_names[0]] == value[0]]
                for i in range(len(col_names) - 1):
                    gt_prob = gt_prob[gt_prob[col_names[i + 1]] == value[i + 1]]
                gt_prob = gt_prob.iloc[0]['counts'] / len(gt_table)

                gt_caches[value_str] = gt_prob

            gen_prob = gen_counts_df[gen_counts_df[col_names[0]] == value[0]]
            for i in range(len(col_names) - 1):
                gen_prob = gen_prob[gen_prob[col_names[i + 1]] == value[i + 1]]
            if len(gen_prob) > 0:
                gen_prob = gen_prob.iloc[0]['counts'] / len(gen_table)
            else:
                gen_prob = eps

            ce -= gt_prob * np.log(gen_prob)

        return ce

    def AR_ComputeCE(self, gt_table, gen_table_dics, gen_total_num, gt_caches, eps=1e-9):
        col_names = ['production_year', 'kind_id']
        gt_table = gt_table.fillna(-1)
        print ('start group by')
        if self.unique_rows is None:
            self.unique_rows = list(gt_table.groupby(col_names).groups)
        ce = 0.

        if not len(gt_caches):
            print ('start group by for gt counts')
            gt_counts_df = gt_table.groupby(col_names).size().reset_index(name='counts')

        for row in self.unique_rows:
            value = list(row)
            value_str = [str(float(i)) for i in value]
            value_str = ','.join(value_str)


            if value_str in gt_caches:
                gt_prob = gt_caches[value_str]
            else:
                gt_prob = gt_counts_df[gt_counts_df[col_names[0]] == value[0]]
                for i in range(len(col_names) - 1):
                    gt_prob = gt_prob[gt_prob[col_names[i + 1]] == value[i + 1]]
                gt_prob = gt_prob.iloc[0]['counts'] / len(gt_table)

                gt_caches[value_str] = gt_prob

            # value_id_format = []
            # for i, col_value in enumerate(value):
            #     value_id_format.append(look_up_list[i][col_value])
            # value_id_format = ','.join(value_id_format)
            if value_str in gen_table_dics:
                gen_prob = gen_table_dics[value_str] / gen_total_num
            else:
                gen_prob = eps

            ce -= gt_prob * np.log(gen_prob)

        return ce

    def MakeModel(self, table, train_data, table_primary_index=None):
        cols_to_train = table.columns
        if self.factorize:
            cols_to_train = train_data.columns

        fixed_ordering = self.MakeOrdering(cols_to_train)

        table_num_columns = table_column_types = table_indexes = None
        if isinstance(train_data, (common.SamplerBasedIterDataset,
                                   common.FactorizedSampleFromJoinIterDataset)):
            table_num_columns = train_data.table_num_columns
            table_column_types = train_data.combined_columns_types
            table_indexes = train_data.table_indexes
            print('table_num_columns', table_num_columns)
            print('table_column_types', table_column_types)
            print('table_indexes', table_indexes)
            print('table_primary_index', table_primary_index)

        if self.use_transformer:
            args = {
                'num_blocks': 4,
                'd_ff': 128,
                'd_model': 32,
                'num_heads': 4,
                'd_ff': 64,
                'd_model': 16,
                'num_heads': 2,
                'nin': len(cols_to_train),
                'input_bins': [c.distribution_size for c in cols_to_train],
                'use_positional_embs': False,
                'activation': 'gelu',
                'fixed_ordering': self.fixed_ordering,
                'dropout': self.dropout,
                'per_row_dropout': self.per_row_dropout,
                'seed': None,
                'join_args': {
                    'num_joined_tables': len(self.join_tables),
                    'table_dropout': self.table_dropout,
                    'table_num_columns': table_num_columns,
                    'table_column_types': table_column_types,
                    'table_indexes': table_indexes,
                    'table_primary_index': table_primary_index,
                }
            }
            args.update(self.transformer_args)
            model = transformer.Transformer(**args).to(train_utils.get_device())
        else:
            model = MakeMade(
                table=table,
                scale=self.fc_hiddens,
                layers=self.layers,
                cols_to_train=cols_to_train,
                seed=self.seed,
                factor_table=train_data if self.factorize else None,
                fixed_ordering=fixed_ordering,
                special_orders=self.special_orders,
                order_content_only=self.order_content_only,
                order_indicators_at_front=self.order_indicators_at_front,
                inv_order=True,
                residual=self.residual,
                direct_io=self.direct_io,
                input_encoding=self.input_encoding,
                output_encoding=self.output_encoding,
                embed_size=self.embed_size,
                dropout=self.dropout,
                per_row_dropout=self.per_row_dropout,
                grouped_dropout=self.grouped_dropout
                if self.factorize else False,
                fixed_dropout_ratio=self.fixed_dropout_ratio,
                input_no_emb_if_leq=self.input_no_emb_if_leq,
                embs_tied=self.embs_tied,
                resmade_drop_prob=self.resmade_drop_prob,
                # DMoL:
                num_dmol=self.num_dmol,
                scale_input=self.scale_input if self.num_dmol else False,
                dmol_cols=self.dmol_cols if self.num_dmol else [],
                # Join specific:
                num_joined_tables=len(self.join_tables),
                table_dropout=self.table_dropout,
                table_num_columns=table_num_columns,
                table_column_types=table_column_types,
                table_indexes=table_indexes,
                table_primary_index=table_primary_index,
            )
        return model

    def MakeProgressiveSamplers(self,
                                model,
                                train_data,
                                do_fanout_scaling=False):
        estimators = []
        dropout = self.dropout or self.per_row_dropout
        for n in self.eval_psamples:
            if self.factorize:
                estimators.append(
                    estimators_lib.FactorizedProgressiveSampling(
                        model,
                        train_data,
                        n,
                        self.join_spec,
                        device=train_utils.get_device(),
                        shortcircuit=dropout,
                        do_fanout_scaling=do_fanout_scaling))
            else:
                estimators.append(
                    estimators_lib.ProgressiveSampling(
                        model,
                        train_data,
                        n,
                        self.join_spec,
                        device=train_utils.get_device(),
                        shortcircuit=dropout,
                        do_fanout_scaling=do_fanout_scaling))
        return estimators

    def MakeProgressiveSampler_train(self,
                                    model,
                                    train_data,
                                    do_fanout_scaling=False, train_virtual_cols=True):
        dropout = self.dropout or self.per_row_dropout
        if self.factorize:
            res = estimators_lib.DifferentialbleFactorizedProgressiveSampling(
                    model,
                    train_data,
                    self.train_sample_num,
                    self.join_spec,
                    device=train_utils.get_device(),
                    shortcircuit=dropout,
                    do_fanout_scaling=do_fanout_scaling,
                    train_virtual_cols=train_virtual_cols)
        else:
             res = estimators_lib.DifferentialbleProgressiveSampling(
                    model,
                    train_data,
                    self.train_sample_num,
                    self.join_spec,
                    device=train_utils.get_device(),
                    shortcircuit=dropout,
                    do_fanout_scaling=do_fanout_scaling,
                    train_virtual_cols=train_virtual_cols)
        return res

    def ProcessSampled(self, sampled, look_up_list_dict):
        sampled = sampled.cpu().numpy()

        fk_table_idx = list(range(len(self.join_tables)))
        fk_table_idx.remove(self.sampled_table_idx)
        indicator_idx = [self.indicator_dics[i] for i in range(len(self.join_tables))]

        indicator_count = 0
        for sample in sampled:

            invalid_sample = False
            for view_id in range(len(fk_table_idx)):
                joined_table_id = fk_table_idx[view_id]

                look_up_list = look_up_list_dict[view_id]

                sampled_idxs = [int(float(sample[content_id])) for content_id in self.content_dics[joined_table_id]]
                # insert placeholder value for pk table sample
                sampled_idxs.insert(0, 0)
                sampled_idxs.insert(0, 0)
                if view_id == 0 or view_id == 1:
                    col_id_fact_0 = self.content_dics[view_id][1]
                    col_id_fact_1 = self.content_dics[view_id][2]

                    sampled_idx_fact_0 = int(
                        self.train_data.columns[col_id_fact_0].all_distinct_values[int(float(sampled_idxs[3]))]) \
                                        << self.train_data.columns[col_id_fact_0].bit_offset

                    sampled_idx_fact_1 = int(self.train_data.columns[col_id_fact_1].all_distinct_values[
                                                int(float(sampled_idxs[4]))]) \
                                        << self.train_data.columns[col_id_fact_1].bit_offset

                    sampled_idx_final = sampled_idx_fact_0 + sampled_idx_fact_1

                    if view_id == 0:
                        original_size = self.loaded_tables[0]['person_id'].distribution_size
                    else:
                        original_size = self.loaded_tables[1]['company_id'].distribution_size

                    if sampled_idx_final >= original_size:
                        invalid_sample = True
                        break

                elif view_id == 3:
                    col_id_fact_0 = self.content_dics[view_id][0]
                    col_id_fact_1 = self.content_dics[view_id][1]

                    sampled_idx_fact_0 = int(
                        self.train_data.columns[col_id_fact_0].all_distinct_values[int(float(sampled_idxs[2]))]) \
                                        << self.train_data.columns[col_id_fact_0].bit_offset

                    sampled_idx_fact_1 = int(self.train_data.columns[col_id_fact_1].all_distinct_values[
                                                int(float(sampled_idxs[3]))]) \
                                        << self.train_data.columns[col_id_fact_1].bit_offset

                    sampled_idx_final = sampled_idx_fact_0 + sampled_idx_fact_1

                    original_size = self.loaded_tables[3]['keyword_id'].distribution_size

                    if sampled_idx_final >= original_size:
                        invalid_sample = True
                        break

            if invalid_sample:
                continue

            all_indicator = True
            for idx in indicator_idx:
                if sample[idx] == 0:
                    all_indicator = False
                    break
            
            if all_indicator:
                indicator_count += 1
            else:
                continue

            primary_id = self.sampled_table_idx
            pri_indicator_id = self.indicator_dics[primary_id]

            if sample[pri_indicator_id] != 0:
                # save sample from full outer join
                weight = 1.
                content = []
                for i in fk_table_idx:
                    if sample[self.indicator_dics[i]] != 0:
                        fanout_id = self.fanout_dics[i][0]
                        fanout = sample[fanout_id]
                        if fanout <= 1:
                            fanout = 1.
                            sample[fanout_id] = 1
                        else:
                            fanout = float(fanout - 1)
                            
                        weight = weight / fanout
                for table_id in range(len(self.join_tables)):
                    for content_id in self.content_dics[table_id]:
                        content.append(str(sample[content_id]))
               
                for idx in indicator_idx:
                    content.append(str(sample[idx]))

                content_group = []
                for content_id in self.content_dics[self.sampled_table_idx]:
                    content_group.append(str(sample[content_id]))

                content_str = ','.join(content)
                content_with_fanout = content
                for i in fk_table_idx:
                    fanout_id = self.fanout_dics[i][0]
                    content_with_fanout.append(str(sample[fanout_id]))
                    content_group.append(str(sample[fanout_id]))

                content_with_fanout_str = ','.join(content_with_fanout)

                if not (content_str in self.sampled_full_view):
                    self.sampled_full_view[content_str] = {"sample": [sample]}
                    self.sampled_full_view[content_str][self.sampled_table_idx] = weight
                    for idx in fk_table_idx:
                        if sample[self.indicator_dics[idx]] != 0:
                            self.sampled_full_view[content_str][idx] = weight*sample[self.fanout_dics[idx][0]]
                    
                else:
                    # self.sampled_full_view[content]["sample"].append(tuple)
                    self.sampled_full_view[content_str][self.sampled_table_idx] += weight
                    for idx in fk_table_idx:
                        if sample[self.indicator_dics[idx]] != 0:
                            self.sampled_full_view[content_str][idx] += (weight*sample[self.fanout_dics[idx][0]])

                if not (content_with_fanout_str in self.sampled_full_view_fanout):
                        self.sampled_full_view_fanout[content_with_fanout_str] = {"sample": [sample]}
                        self.sampled_full_view_fanout[content_with_fanout_str][self.sampled_table_idx] = weight
                        for idx in fk_table_idx:
                            if sample[self.indicator_dics[idx]] != 0:
                                self.sampled_full_view_fanout[content_with_fanout_str][idx] = weight*sample[self.fanout_dics[idx][0]]
                    
                else:
                    # self.sampled_full_view[content]["sample"].append(tuple)
                    self.sampled_full_view_fanout[content_with_fanout_str][self.sampled_table_idx] += weight
                    for idx in fk_table_idx:
                        if sample[self.indicator_dics[idx]] != 0:
                            self.sampled_full_view_fanout[content_with_fanout_str][idx] += (weight*sample[self.fanout_dics[idx][0]])

                content_group_str = ','.join(content_group)
                if not content_group_str in self.sampled_group_dict:
                    self.sampled_group_dict[content_group_str] = [content_with_fanout_str]
                else:
                    self.sampled_group_dict[content_group_str].append(content_with_fanout_str)

        # print("number of samples with all indicators: {}".format(indicator_count))
        self.total_tuple_sampled += indicator_count
        # print("total number of samples obtained: {}".format(self.total_tuple_sampled))
        # print("total number of pk+fanout groups: {}".format(len(self.sampled_group_dict)))

    def _simple_save(self):
        semi_str = 'usesemi' if self.semi_train else 'nosemi'
        path = os.path.join(
            wandb.run.dir, 'model-{}-{}-{}-{}.h5'.format(self.epoch,
                                                   '-'.join(self.join_tables), semi_str, self.q_weight))
        torch.save(self.model.state_dict(), path)
        wandb.save(path)
        return path

    def _train(self):
        final_time_start = time.time()
        fk_table_idx = list(range(len(self.join_tables)))
        fk_table_idx.remove(self.sampled_table_idx)

        if self.checkpoint_to_load or self.eval_join_sampling:
            model = self.model

            batch_size = 100000
            print("join cardinality: {}".format(self.table.cardinality))

            pk_look_up_list = []
            # generate title table
            pk_look_up_list.append(self.loaded_tables[self.sampled_table_idx]['production_year'].all_distinct_values)
            pk_look_up_list.append(self.loaded_tables[self.sampled_table_idx]['kind_id'].all_distinct_values)

            look_up_list_dict = {}
            for view_id in range(len(fk_table_idx)):
                if view_id == 0:
                    look_up_list_dict[view_id] = []
                    look_up_list_dict[view_id].append(self.loaded_tables[0]['role_id'].all_distinct_values)
                    look_up_list_dict[view_id].append(self.loaded_tables[0]['person_id'].all_distinct_values)
                elif view_id == 1:
                    look_up_list_dict[view_id] = []
                    look_up_list_dict[view_id].append(self.loaded_tables[1]['company_type_id'].all_distinct_values)
                    look_up_list_dict[view_id].append(self.loaded_tables[1]['company_id'].all_distinct_values)
                elif view_id == 2:
                    look_up_list_dict[view_id] = []
                    look_up_list_dict[view_id].append(self.loaded_tables[2]['info_type_id'].all_distinct_values)
                elif view_id == 3:
                    look_up_list_dict[view_id] = []
                    look_up_list_dict[view_id].append(self.loaded_tables[3]['keyword_id'].all_distinct_values)
                elif view_id == 4:
                    look_up_list_dict[view_id] = []
                    look_up_list_dict[view_id].append(self.loaded_tables[5]['info_type_id'].all_distinct_values)

            for iter_num in range(2001):
                
                self.sampled_table_nums = [0] * len(self.join_tables)
                # begin_time = t1 = time.time()
                sampled = model.sample(num=batch_size, device=train_utils.get_device())
                # dur = time.time()-t1
                # print("sample time {}ms".format(dur * 1000))

                # t1 = time.time()
                self.ProcessSampled(sampled, look_up_list_dict)
                # dur = time.time() - t1
                # print("process time {}ms".format(dur * 1000))

                if iter_num % 100 == 0:
                    print("iter_num = {}".format(iter_num+1))
                    ####### assign key to all tables ######
                    title_num = 0
                    total_weight = 0
                    for val in self.sampled_full_view:
                        total_weight += self.sampled_full_view[val][self.sampled_table_idx]

                    table_weight_sum = {}
                    for i in range(len(self.join_tables)):
                        table_weight_sum[i] = 0
                    for val in self.sampled_full_view:
                        for i in range(len(self.join_tables)):
                            if i in self.sampled_full_view[val]:
                                table_weight_sum[i] += self.sampled_full_view[val][i]
                    
                    scale_values = {}
                    for i in range(len(self.join_tables)):
                        scale_values[i] = self.loaded_tables[i].cardinality / table_weight_sum[i]
                    print("scale value: {}".format(scale_values))
                    print("table weight sum: {}".format(table_weight_sum))

                    actual_scale = self.table.cardinality / (batch_size*(iter_num+1))
                    print("theoretical scale value: {}".format(actual_scale))
                    for i in range(len(self.join_tables)):
                        print("number of tuple generated for {}: {}".format(self.join_tables[i], round(actual_scale*table_weight_sum[i])))
                    
                    print("number of distinct groups: {}".format(len(self.sampled_full_view)))
                    print("number of distinct groups with fanout: {}".format(len(self.sampled_full_view_fanout)))

                    counter = 0
                    generated_counter_group = 0
                    sampled_fanout_group_pk = {}
                    for group_val in self.sampled_group_dict:
                        self.sampled_group_dict[group_val].sort()
                        pk_sum = 0
                        for val in self.sampled_group_dict[group_val]:
                            pk_sum += self.sampled_full_view_fanout[val][self.sampled_table_idx] * scale_values[self.sampled_table_idx]
                    
                        if pk_sum > 0.5:
                            generated_counter_group += 1
                            current_idx = 0
                            group_counter = 0
                            current_group = []
                            for val in self.sampled_group_dict[group_val]:
                                group_counter += self.sampled_full_view_fanout[val][self.sampled_table_idx] * scale_values[self.sampled_table_idx]
                                current_group.append(val)
                                if int(group_counter) != current_idx:
                                    new_idx_list = list(range(current_idx+counter, int(group_counter)+counter))
                                    new_idx_list_str = ','.join([str(item) for item in new_idx_list])
                                    sampled_fanout_group_pk[new_idx_list_str] = current_group
                                    current_group = []
                                    current_idx = int(group_counter)
                            
                            if current_group:
                                new_idx_list = list(range(current_idx+counter, current_idx+counter+1))
                                new_idx_list_str = ','.join([str(item) for item in new_idx_list])
                                sampled_fanout_group_pk[new_idx_list_str] = current_group
                                current_idx += 1


                            num_pk = current_idx
                            counter = counter + num_pk

                    print("number of pk assigned group by fanout: {}".format(counter))
                    print("count number of group with title tuple generated group by fanout: {}".format(generated_counter_group))

                    print("generating title table combine by fanout...")
                    sampled_table = {}
                    for pk_str in sampled_fanout_group_pk:
                        val = sampled_fanout_group_pk[pk_str][0]
                        pk_list = pk_str.split(',')
                        current_val = self.sampled_full_view_fanout[val]

                        value_str = []
                        for i in range(2):
                            sampled_idx = int(float(current_val['sample'][0][self.content_dics[self.sampled_table_idx][i]]))
                            sampled_value = pk_look_up_list[i][sampled_idx]
                            if np.isnan(sampled_value):
                                value_str.append('')
                            else:
                                value_str.append(str(sampled_value))
                    
                        value_str = ','.join(value_str)
                        if not (value_str in sampled_table):
                            sampled_table[value_str] = pk_list
                        else: 
                            sampled_table[value_str] += pk_list


                    print("generating fk tables combined by fanout...")
                    # generate fk table
                    # count = 0
                    sampled_fk_tables = {}
                    for view_id in range(len(fk_table_idx)):
                            joined_table_id = fk_table_idx[view_id]
                            sampled_fk_tables[joined_table_id] = {}

                    unrej = [0 for i in range(len(fk_table_idx))]
                    for pk_str in sampled_fanout_group_pk:
                        val = sampled_fanout_group_pk[pk_str][0]
                        pk_list = pk_str.split(',')
                
                        weight_dict = {}
                        for view_id in range(len(fk_table_idx)):
                            weight_dict[view_id] = {}

                        for val in sampled_fanout_group_pk[pk_str]:
                            current_val = self.sampled_full_view_fanout[val]
                            # if count % 5000 == 0:
                            #     print("processed {} vals".format(count))
                            # count += 1

                            for view_id in range(len(fk_table_idx)):
                                joined_table_id = fk_table_idx[view_id]
                                if not (joined_table_id in current_val):
                                    continue

                                look_up_list = look_up_list_dict[view_id]

                                sampled_idxs = [int(float(current_val['sample'][0][content_id])) for content_id in self.content_dics[joined_table_id]]
                                # insert placeholder value for pk table sample
                                sampled_idxs.insert(0, 0)
                                sampled_idxs.insert(0, 0)
                                if view_id == 0 or view_id == 1:
                                    col_id_fact_0 = self.content_dics[view_id][1]
                                    col_id_fact_1 = self.content_dics[view_id][2]

                                    sampled_idx_fact_0 = int(
                                        self.train_data.columns[col_id_fact_0].all_distinct_values[int(float(sampled_idxs[3]))]) \
                                                        << self.train_data.columns[col_id_fact_0].bit_offset

                                    sampled_idx_fact_1 = int(self.train_data.columns[col_id_fact_1].all_distinct_values[
                                                                int(float(sampled_idxs[4]))]) \
                                                        << self.train_data.columns[col_id_fact_1].bit_offset

                                    sampled_idx_final = sampled_idx_fact_0 + sampled_idx_fact_1

                                    if view_id == 0:
                                        original_size = self.loaded_tables[0]['person_id'].distribution_size
                                    else:
                                        original_size = self.loaded_tables[1]['company_id'].distribution_size

                                    if sampled_idx_final >= original_size:
                                        exit
                                        continue

                                elif view_id == 3:
                                    col_id_fact_0 = self.content_dics[view_id][0]
                                    col_id_fact_1 = self.content_dics[view_id][1]

                                    sampled_idx_fact_0 = int(
                                        self.train_data.columns[col_id_fact_0].all_distinct_values[int(float(sampled_idxs[2]))]) \
                                                        << self.train_data.columns[col_id_fact_0].bit_offset

                                    sampled_idx_fact_1 = int(self.train_data.columns[col_id_fact_1].all_distinct_values[
                                                                int(float(sampled_idxs[3]))]) \
                                                        << self.train_data.columns[col_id_fact_1].bit_offset

                                    sampled_idx_final = sampled_idx_fact_0 + sampled_idx_fact_1

                                    original_size = self.loaded_tables[3]['keyword_id'].distribution_size

                                    if sampled_idx_final >= original_size:
                                        exit
                                        continue

                                value_str = []

                                for i, sampled_idx in enumerate(sampled_idxs):
                                    if i < 2:
                                        continue
                                    look_up_id = i - 2
                                    if view_id == 0 or view_id == 1:
                                        if look_up_id == 0:
                                            sampled_value = look_up_list[look_up_id][int(float(sampled_idx))]
                                            if np.isnan(sampled_value):
                                                value_str.append('')
                                            else:
                                                value_str.append(str(sampled_value))
                                        elif look_up_id == 1:
                                            col_id_fact_0 = self.content_dics[view_id][1]
                                            col_id_fact_1 = self.content_dics[view_id][2]

                                            sampled_idx_fact_0 = int(self.train_data.columns[col_id_fact_0].all_distinct_values[int(float(sampled_idx))]) \
                                                                << self.train_data.columns[col_id_fact_0].bit_offset

                                            sampled_idx_fact_1 = int(self.train_data.columns[col_id_fact_1].all_distinct_values[int(float(sampled_idxs[i+1]))]) \
                                                                << self.train_data.columns[col_id_fact_1].bit_offset

                                            sampled_idx_final = sampled_idx_fact_0 + sampled_idx_fact_1

                                            sampled_value = look_up_list[look_up_id][sampled_idx_final]
                                            if np.isnan(sampled_value):
                                                value_str.append('')
                                            else:
                                                value_str.append(str(sampled_value))
                                    elif view_id == 2 or view_id == 4:
                                        sampled_value = look_up_list[look_up_id][int(float(sampled_idx))]
                                        if np.isnan(sampled_value):
                                            value_str.append('')
                                        else:
                                            value_str.append(str(sampled_value))
                                    elif view_id == 3:
                                        if look_up_id == 0:
                                            col_id_fact_0 = self.content_dics[view_id][0]
                                            col_id_fact_1 = self.content_dics[view_id][1]

                                            sampled_idx_fact_0 = int(self.train_data.columns[col_id_fact_0].all_distinct_values[
                                                                    int(float(sampled_idx))]) \
                                                                << self.train_data.columns[col_id_fact_0].bit_offset

                                            sampled_idx_fact_1 = int(self.train_data.columns[col_id_fact_1].all_distinct_values[
                                                                    int(float(sampled_idxs[i + 1]))])\
                                                                << self.train_data.columns[col_id_fact_1].bit_offset

                                            sampled_idx_final = sampled_idx_fact_0 + sampled_idx_fact_1

                                            sampled_value = look_up_list[look_up_id][sampled_idx_final]
                                            if np.isnan(sampled_value):
                                                value_str.append('')
                                            else:
                                                value_str.append(str(sampled_value))

                                value_str = ','.join(value_str)

                                if not value_str in weight_dict[view_id]:
                                    weight_dict[view_id][value_str] = scale_values[joined_table_id] * current_val[joined_table_id]
                                else:
                                    weight_dict[view_id][value_str] += scale_values[joined_table_id] * current_val[joined_table_id]
                                unrej[view_id] += 1


                        for view_id in range(len(fk_table_idx)):
                            joined_table_id = fk_table_idx[view_id]
                            for value_str in weight_dict[view_id]:
                                num_tuples = int(round(weight_dict[view_id][value_str]))
                                if num_tuples == 0:
                                    num_tuples = 1
                                fk_list = pk_list*(num_tuples // len(pk_list)) + pk_list[:num_tuples%len(pk_list)]
                                if not (value_str in sampled_fk_tables[fk_table_idx[view_id]]):
                                    sampled_fk_tables[joined_table_id][value_str] = fk_list
                                else: 
                                    sampled_fk_tables[joined_table_id][value_str] += fk_list
                    
                    
                    print("number of unrejected samples: {}".format(unrej))

                    folder_name = self.folder_name + '_fanout'
                    print("saving generated tables...")
                    res_file = open('./{}/title_{}.csv'.format(folder_name, str(iter_num)), 'w', encoding="utf8")
                    res_file.write('id,production_year,kind_id\n')
                    for val in sampled_table:
                        for pk in sampled_table[val]:
                            res_file.write(str(pk) + ',' + val + '\n')
                    res_file.close()

                    for i in range(len(fk_table_idx)):
                        fk_table_id = fk_table_idx[i]
                        sampled_fk_table = sampled_fk_tables[fk_table_id]

                        if i == 0:
                            fk_res_file = open('./{}/cast_info_{}.csv'.format(folder_name, str(iter_num)), 'w', encoding="utf8")
                            fk_res_file.write('movie_id,role_id,person_id\n')
                        elif i == 1:
                            fk_res_file = open('./{}/movie_companies_{}.csv'.format(folder_name, str(iter_num)), 'w', encoding="utf8")
                            fk_res_file.write('movie_id,company_type_id,company_id\n')
                        elif i == 2:
                            fk_res_file = open('./{}/movie_info_{}.csv'.format(folder_name, str(iter_num)), 'w', encoding="utf8")
                            fk_res_file.write('movie_id,info_type_id\n')
                        elif i == 3:
                            fk_res_file = open('./{}/movie_keyword_{}.csv'.format(folder_name, str(iter_num)), 'w', encoding="utf8")
                            fk_res_file.write('movie_id,keyword_id\n')
                        elif i == 4:
                            fk_res_file = open('./{}/movie_info_idx_{}.csv'.format(folder_name, str(iter_num)), 'w', encoding="utf8")
                            fk_res_file.write('movie_id,info_type_id\n')

                        total_num = 0
                        for val in sampled_fk_table:
                            total_num += len(sampled_fk_table[val])
                            for fk in sampled_fk_table[val]:
                                fk_res_file.write(str(fk) + ',' + val + '\n')
                        print("generated total number of {} tuples: {}".format(self.join_tables[fk_table_id], total_num))
                        res_file.close()

                    print("generated number of distinct title tuples: {}".format(len(sampled_table)))
                    total_num = 0
                    for val in sampled_table:
                        total_num += len(sampled_table[val])
                    print("generated total number of title tuples: {}".format(total_num))
                    gt_table = self.loaded_tables[self.sampled_table_idx].data
                    gt_table = gt_table[['production_year', 'kind_id']]
                    sampled_table_count = {}
                    for val in sampled_table:
                        sampled_table_count[val] = len(sampled_table[val])
                    ce = self.AR_ComputeCE(gt_table, sampled_table_count, total_num, self.gt_caches)

                    print("generated ce: {}".format(ce))

                    current_time = time.time()
                    print("time lapsed so far: {}".format(current_time - final_time_start))

            self.model.model_bits = 0
            results = 0

            # results = self.evaluate(self.num_eval_queries_at_checkpoint_load,
            #                         done=True)
            # self._maybe_check_asserts(results, returns=None)
            return {
                'epoch': 0,
                'done': True,
                'results': results,
            }


    def _maybe_check_asserts(self, results, returns):
        if self.asserts:
            # asserts = {key: val, ...} where key either exists in "results"
            # (returned by evaluate()) or "returns", both defined above.
            error = False
            message = []
            for key, max_val in self.asserts.items():
                if key in results:
                    if results[key] >= max_val:
                        error = True
                        message.append(str((key, results[key], max_val)))
                elif returns[key] >= max_val:
                    error = True
                    message.append(str((key, returns[key], max_val)))
            assert not error, '\n'.join(message)

    def _save(self, tmp_checkpoint_dir):
        if self.checkpoint_to_load or not self.save_checkpoint_at_end:
            return {}
        semi_str = 'usesemi' if self.semi_train else 'nosemi'
        # NOTE: see comment at ReportModel() call site about model size.
        if self.fixed_ordering is None:
            if self.seed is not None:
                PATH = 'models/{}-{:.1f}MB-model{:.3f}-{}-{}epochs-seed{}-{}-q-{}.pt'.format(
                    self.dataset, self.mb, self.model.model_bits,
                    self.model.name(), self.epoch, self.seed, semi_str, self.q_weight)
            else:
                PATH = 'models/{}-{:.1f}MB-model{:.3f}-{}-{}epochs-seed{}-{}-{}-q-{}.pt'.format(
                    self.dataset, self.mb, self.model.model_bits,
                    self.model.name(), self.epoch, self.seed, time.time(),semi_str,self.q_weight)
        else:
            PATH = 'models/{}-{:.1f}MB-model{:.3f}-{}-{}epochs-seed{}-order{}-{}-q-{}.pt'.format(
                self.dataset, self.mb, self.model.model_bits, self.model.name(),
                self.epoch, self.seed,
                str(self.order_seed) if self.order_seed is not None else
                '_'.join(map(str, self.fixed_ordering))[:60], semi_str,self.q_weight)

        if self.dataset == 'imdb':
            tuples_seen = self.bs * self.max_steps * self.epochs
            PATH = PATH.replace(
                '-seed', '-{}tups-seed'.format(utils.HumanFormat(tuples_seen)))

            if len(self.join_tables) == 1:
                PATH = PATH.replace('imdb',
                                    'indep-{}'.format(self.join_tables[0]))

        torch.save(self.model.state_dict(), PATH)
        # wandb.save(PATH)
        print('Saved to:', PATH)
        return {'path': PATH}

    def stop(self):
        self.tbx_logger.flush()
        self.tbx_logger.close()

    def _log_result(self, results):
        psamples = {}
        # When we run > 1 epoch in one tune "iter", we want TensorBoard x-axis
        # to show our real epoch numbers.
        results['iterations_since_restore'] = results[
            'training_iteration'] = self.epoch
        for k, v in results['results'].items():
            if 'psample' in k:
                psamples[k] = v
        # wandb.log(results)
        self.tbx_logger.on_result(results)
        self.tbx_logger._file_writer.add_custom_scalars_multilinechart(
            map(lambda s: 'ray/tune/results/{}'.format(s), psamples.keys()),
            title='psample')

    def ErrorMetric(self, est_card, card):
        if card == 0 and est_card != 0:
            return est_card
        if card != 0 and est_card == 0:
            return card
        if card == 0 and est_card == 0:
            return 1.0

        return max(est_card / card, card / est_card)

    def Query(self,
              estimators,
              oracle_card=None,
              query=None,
              table=None,
              oracle_est=None):
        assert query is not None
        cols, ops, vals = query
        card = oracle_est.Query(cols, ops,
                                vals) if oracle_card is None else oracle_card
        print('Q(', end='')
        for c, o, v in zip(cols, ops, vals):
            print('{} {} {}, '.format(c.name, o, str(v)), end='')
        print('): ', end='')
        print('\n  actual {} ({:.3f}%) '.format(card,
                                                card / table.cardinality * 100),
              end='')
        for est in estimators:
            est_card = est.Query(cols, ops, vals)
            err = self.ErrorMetric(est_card, card)
            est.AddError(err, est_card, card)
            print('{} {} (err={:.3f}) '.format(str(est), est_card, err), end='')

        print()

    def evaluate(self, num_queries, done, estimators=None):
        model = self.model
        if isinstance(model, DataParallelPassthrough):
            model = model.module
        model.eval()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        results = {}
        if num_queries:
            if estimators is None:
                estimators = self.MakeProgressiveSamplers(
                    model,
                    self.train_data if self.factorize else self.table,
                    do_fanout_scaling=(self.dataset == 'imdb'))
                if self.eval_join_sampling:  # None or an int.
                    estimators = [
                        estimators_lib.JoinSampling(self.train_data, self.table,
                                                    self.eval_join_sampling)
                    ]

            assert self.loaded_queries is not None
            num_queries = min(len(self.loaded_queries), num_queries)
            for i in range(self.train_queries, self.train_queries+self.test_queries):
                print('Query {}:'.format(i), end=' ')
                query = self.loaded_queries[i]
                self.Query(estimators,
                           oracle_card=None if self.oracle_cards is None else
                           self.oracle_cards[i],
                           query=query,
                           table=self.table,
                           oracle_est=self.oracle)
                if i % 100 == 0:
                    for est in estimators:
                        est.report()

            for est in estimators:
                results[str(est) + '_max'] = np.max(est.errs)
                results[str(est) + '_p99'] = np.quantile(est.errs, 0.99)
                results[str(est) + '_p95'] = np.quantile(est.errs, 0.95)
                results[str(est) + '_median'] = np.median(est.errs)
                est.report()

                series = pd.Series(est.query_dur_ms)
                print(series.describe())
                series.to_csv(str(est) + '.csv', index=False, header=False)

            # estimate job-light queries
            print("start estimating job-light queries")

            for est in estimators:
                est.ClearRecords()

            for i in range(len(self.loaded_job_light_queries)):
                print('Query {}:'.format(i), end=' ')

                query = self.loaded_job_light_queries[i]
                self.Query(estimators,
                           oracle_card=None if self.job_light_oracle_cards is None else
                           self.job_light_oracle_cards[i],
                           query=query,
                           table=self.table,
                           oracle_est=self.oracle)
                if i % 30 == 0:
                    for est in estimators:
                        est.report()

                for est in estimators:
                    results[str(est) + '_light_max'] = np.max(est.errs)
                    results[str(est) + '_light_p99'] = np.quantile(est.errs, 0.99)
                    results[str(est) + '_light_p95'] = np.quantile(est.errs, 0.95)
                    results[str(est) + '_light_median'] = np.median(est.errs)
                    est.report()

                    series = pd.Series(est.query_dur_ms)
                    print(series.describe())

        return results

if __name__ == '__main__':
    ray.init(ignore_reinit_error=True)

    for k in args.run:
        assert k in experiments.EXPERIMENT_CONFIGS, 'Available: {}'.format(
            list(experiments.EXPERIMENT_CONFIGS.keys()))

    num_gpus = args.gpus if torch.cuda.is_available() else 0
    num_cpus = args.cpus

    tune.run_experiments(
        {
            k: {
                'run': NeuroCard,
                'checkpoint_at_end': True,
                'resources_per_trial': {
                    'gpu': num_gpus,
                    'cpu': num_cpus,
                },
                'config': dict(
                    experiments.EXPERIMENT_CONFIGS[k], **{
                        '__run': k,
                        '__gpu': num_gpus,
                        '__cpu': num_cpus
                    }),
            } for k in args.run
        },
        concurrent=True,
    )
