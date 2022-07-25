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
import multiprocessing as mp

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

os.environ['RAY_DEBUG_DISABLE_MEMORY_MONITOR']= '0.999'
parser = argparse.ArgumentParser()
# os.environ["CUDA_VISIBLE_DEVICES"]= "0,4"
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


class SAM(tune.Trainable):

    def _setup(self, config):
        self.config = config
        print('SAM config:')
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
                                      ].index(self.pk_table)

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

        # set the columns to generate
        self.content_cols, self.indicator_cols, self.fanout_cols = self.MakeIndexRecords(self.table,
                                                                          self.train_data,
                                                                          table_primary_index=table_primary_index)
        print (self.content_cols)
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

        self.pk_table_id = table_primary_index
        self.sampled_view_idx = []

        self.sampled_full_view = {}
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
        """
        Extract the index of each table's columns in the AR model
        """
        cols_to_train = table.columns
        if self.factorize:
            cols_to_train = train_data.columns

        num_tables = len(self.join_tables)

        fanout_cols = []
        for i in range (num_tables):
            fanout_cols.append([])
        indicator_cols = [None] * num_tables
        
        content_cols = []
        for i in range (num_tables):
            content_cols.append({})

        table_num_columns = table_column_types = table_indexes = None
        if isinstance(train_data, (common.SamplerBasedIterDataset,
                                   common.FactorizedSampleFromJoinIterDataset)):
            table_num_columns = train_data.table_num_columns
            table_column_types = train_data.combined_columns_types
            table_indexes = train_data.table_indexes
            print('table_num_columns:', table_num_columns)
            print('table_column_types:', table_column_types)
            print('table_indexes:', table_indexes)
            print('table_primary_index:', table_primary_index)

            print('number of columns in AR model:', len(cols_to_train))
            for col_id, col in enumerate(cols_to_train):
                col_type = table_column_types[col_id]
                table_id = int(table_indexes[col_id])

                table_name = self.join_tables[table_id]
                table_key = table_name + '.csv'
                cols_candidate = self.generation_cols[table_key]

                if col_type == common.TYPE_NORMAL_ATTR:
                    col_name = col.name.split(':')[-1]
                    # print(col_name)
                    if col_name[-6:-2] == "fact":
                        # print(col_name)
                        if col_name[:-7] in cols_candidate:
                            if col_name[:-7] in content_cols[table_id]:
                                content_cols[table_id][col_name[:-7]].append(col_id)
                            else:
                                content_cols[table_id][col_name[:-7]] = [col_id]
                    elif col_name in cols_candidate:
                        if col_name in content_cols[table_id]:
                            content_cols[table_id][col_name].append(col_id)
                        else:
                            content_cols[table_id][col_name] = [col_id]

                elif col_type == common.TYPE_INDICATOR:
                    indicator_cols[table_id] = col_id
                else:
                    fanout_cols[table_id].append(col_id)

        return content_cols, indicator_cols, fanout_cols

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

    def AR_ComputeCE(self, col_names, gt_table, gen_table_dics, gen_total_num, gt_caches, eps=1e-9):
        gt_table = gt_table.fillna(-1)
        # print ('start group by')
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

    def ProcessSampled(self, sampled):
        sampled = sampled.cpu().numpy()

        fk_table_idx = list(range(len(self.join_tables)))
        fk_table_idx.remove(self.pk_table_id)
        indicator_idx = [self.indicator_cols[i] for i in range(len(self.join_tables))]

        indicator_count = 0
        for sample in sampled:

            # check if the sampled value for factorized column is invalid
            invalid_sample = False

            for view_id in range(len(fk_table_idx)):
                joined_table_id = fk_table_idx[view_id]

                # reconstruct column values from factorized columns

                for col_name in self.content_cols[joined_table_id]:
                    # only check if the column is factorized
                    col_ids = self.content_cols[joined_table_id][col_name]
                    if len(col_ids) > 1:
                        sampled_idx_value = 0
                        for col_id in col_ids:
                            current_idx_value = int(self.train_data.columns[col_id].all_distinct_values[int(float(sample[col_id]))]) \
                                                << self.train_data.columns[col_id].bit_offset
                            sampled_idx_value += current_idx_value
                        
                        original_size = self.loaded_tables[joined_table_id][col_name].distribution_size
                        if sampled_idx_value >= original_size:
                            invalid_sample = True
                            break

            # if the reconstructed value is out-of-range, discard the sample
            if invalid_sample:
                continue

            # only use samples where all indicator column values are one
            all_indicator = True
            for idx in indicator_idx:
                if sample[idx] == 0:
                    all_indicator = False
                    break
            
            if all_indicator:
                indicator_count += 1
            else:
                continue

            primary_id = self.pk_table_id
            pri_indicator_id = self.indicator_cols[primary_id]

            if sample[pri_indicator_id] != 0:
                # save sample from full outer join
                weight = 1.
                content = []
                for i in fk_table_idx:
                    if sample[self.indicator_cols[i]] != 0:
                        fanout_id = self.fanout_cols[i][0]
                        fanout = sample[fanout_id]
                        if fanout <= 1:
                            fanout = 1.
                            sample[fanout_id] = 1
                        else:
                            fanout = float(fanout - 1)
                            
                        weight = weight / fanout

                content_col_ids = []
                for table_id in range(len(self.join_tables)):
                    for col_name in self.content_cols[table_id]:
                        content_col_ids += self.content_cols[table_id][col_name]    
                
                for col_id in content_col_ids:
                    content.append(str(sample[col_id]))
               
                for idx in indicator_idx:
                    content.append(str(sample[idx]))

                pk_content_col_ids = []
                for col_name in self.content_cols[self.pk_table_id]:
                    pk_content_col_ids += self.content_cols[self.pk_table_id][col_name]   

                content_group = []
                for col_id in pk_content_col_ids:
                    content_group.append(str(sample[col_id])) 

                content_str = ','.join(content)
                content_with_fanout = content
                for i in fk_table_idx:
                    fanout_id = self.fanout_cols[i][0]
                    content_with_fanout.append(str(sample[fanout_id]))
                    content_group.append(str(sample[fanout_id]))

                content_with_fanout_str = ','.join(content_with_fanout)

                if not (content_with_fanout_str in self.sampled_full_view):
                        self.sampled_full_view[content_with_fanout_str] = {"sample": [sample]}
                        self.sampled_full_view[content_with_fanout_str][self.pk_table_id] = weight
                        for idx in fk_table_idx:
                            if sample[self.indicator_cols[idx]] != 0:
                                self.sampled_full_view[content_with_fanout_str][idx] = weight*sample[self.fanout_cols[idx][0]]
                    
                else:
                    self.sampled_full_view[content_with_fanout_str][self.pk_table_id] += weight
                    for idx in fk_table_idx:
                        if sample[self.indicator_cols[idx]] != 0:
                            self.sampled_full_view[content_with_fanout_str][idx] += (weight*sample[self.fanout_cols[idx][0]])

                content_group_str = ','.join(content_group)
                if not content_group_str in self.sampled_group_dict:
                    self.sampled_group_dict[content_group_str] = [content_with_fanout_str]
                else:
                    self.sampled_group_dict[content_group_str].append(content_with_fanout_str)

        self.total_tuple_sampled += indicator_count

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
        fk_table_idx.remove(self.pk_table_id)

        if self.checkpoint_to_load or self.eval_join_sampling:
            model = self.model

            batch_size = 100000
            print("join cardinality: {}".format(self.table.cardinality))

            # record distinct value of pk tables columns
            look_up_list = []
            for table_id in range(len(self.join_tables)):
                column_values_dict = {}
                for col_name in self.content_cols[table_id]:
                    column_values_dict[col_name]=(self.loaded_tables[table_id][col_name].all_distinct_values)
                look_up_list.append(column_values_dict)

            for iter_num in range(self.total_iterations + 1):
                
                self.sampled_table_nums = [0] * len(self.join_tables)
                sampled = model.sample(num=batch_size, device=train_utils.get_device())

                self.ProcessSampled(sampled)

                if iter_num % self.save_frequency == 0:
                    print("iter_num = {}".format(iter_num+1))

                    pk_total_weight = 0

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
                    print("table weight sum: {}".format(table_weight_sum))
                    print("scale value: {}".format(scale_values))
                    
                    print("number of groups: {}".format(len(self.sampled_full_view)))

                    # Group-and-Merge algorithm
                    pk_count = 0
                    generated_group_count = 0
                    sampled_fanout_group_pk = {}
                    for group_val in self.sampled_group_dict:
                        self.sampled_group_dict[group_val].sort()

                        # Calculate sum of pk relation weight of the group
                        pk_sum = 0
                        for val in self.sampled_group_dict[group_val]:
                            pk_sum += self.sampled_full_view[val][self.pk_table_id] * scale_values[self.pk_table_id]
                    
                        # Generate pk from the group if the sum of pk relation weight is greater then 0.5
                        if pk_sum > 0.5:
                            generated_group_count += 1
                            current_idx = 0
                            tuple_count = 0
                            current_set = []
                            for val in self.sampled_group_dict[group_val]:
                                tuple_count += self.sampled_full_view[val][self.pk_table_id] * scale_values[self.pk_table_id]
                                current_set.append(val)
                                if int(tuple_count) != current_idx:
                                    new_idx_list = list(range(current_idx+pk_count, int(tuple_count)+pk_count))
                                    new_idx_list_str = ','.join([str(item) for item in new_idx_list])
                                    sampled_fanout_group_pk[new_idx_list_str] = current_set
                                    current_set = []
                                    current_idx = int(tuple_count)
                            
                            if current_set:
                                new_idx_list = list(range(current_idx+pk_count, current_idx+pk_count+1))
                                new_idx_list_str = ','.join([str(item) for item in new_idx_list])
                                sampled_fanout_group_pk[new_idx_list_str] = current_set
                                current_idx += 1


                            num_pk = current_idx
                            pk_count = pk_count + num_pk

                    print("number of pk assigned: {}".format(pk_count))
                    print("number of groups with pk assigned: {}".format(generated_group_count))

                    print("Generating pk table...")
                    generated_pk_table = {}
                    for pk_str in sampled_fanout_group_pk:
                        val = sampled_fanout_group_pk[pk_str][0]
                        pk_list = pk_str.split(',')
                        current_val = self.sampled_full_view[val]

                        value_str = []

                        for col_name in self.content_cols[self.pk_table_id]:
                            # only check if the column is factorized
                            col_ids = self.content_cols[self.pk_table_id][col_name]
                            if len(col_ids) > 1:
                                sampled_idx_value = 0
                                for col_id in col_ids:
                                    current_idx_value = int(self.train_data.columns[col_id].all_distinct_values[int(float(current_val['sample'][0][col_id]))]) \
                                                        << self.train_data.columns[col_id].bit_offset
                                    sampled_idx_value += current_idx_value
                            else:
                                sampled_idx_value = int(float(current_val['sample'][0][col_ids[0]]))
                            
                            sampled_value = look_up_list[self.pk_table_id][col_name][sampled_idx_value]
                            if np.isnan(sampled_value):
                                value_str.append('')
                            else:
                                value_str.append(str(sampled_value))
                    
                        value_str = ','.join(value_str)
                        if not (value_str in generated_pk_table):
                            generated_pk_table[value_str] = pk_list
                        else: 
                            generated_pk_table[value_str] += pk_list


                    print("Generating fk tables...")
                    generated_fk_tables = {}
                    for table_id in fk_table_idx:
                        generated_fk_tables[table_id] = {}

                    for pk_str in sampled_fanout_group_pk:
                        val = sampled_fanout_group_pk[pk_str][0]
                        pk_list = pk_str.split(',')
                
                        weight_dict = {}
                        for table_id in fk_table_idx:
                            weight_dict[table_id] = {}

                        for val in sampled_fanout_group_pk[pk_str]:
                            current_val = self.sampled_full_view[val]

                            for table_id in fk_table_idx:
                                if not (table_id in current_val):
                                    continue

                                # reconstruct column values from factorized columns

                                value_str = []

                                for col_name in self.content_cols[table_id]:
                                    # only check if the column is factorized
                                    col_ids = self.content_cols[table_id][col_name]
                                    if len(col_ids) > 1:
                                        sampled_idx_value = 0
                                        for col_id in col_ids:
                                            current_idx_value = int(self.train_data.columns[col_id].all_distinct_values[int(float(current_val['sample'][0][col_id]))]) \
                                                                << self.train_data.columns[col_id].bit_offset
                                            sampled_idx_value += current_idx_value
                                    else:
                                        sampled_idx_value = int(float(current_val['sample'][0][col_ids[0]]))
                                    
                                    sampled_value = look_up_list[table_id][col_name][sampled_idx_value]
                                    if np.isnan(sampled_value):
                                        value_str.append('')
                                    else:
                                        value_str.append(str(sampled_value))

                                value_str = ','.join(value_str)

                                if not value_str in weight_dict[table_id]:
                                    weight_dict[table_id][value_str] = scale_values[table_id] * current_val[table_id]
                                else:
                                    weight_dict[table_id][value_str] += scale_values[table_id] * current_val[table_id]


                        for table_id in fk_table_idx:
                            for value_str in weight_dict[table_id]:
                                num_tuples = int(round(weight_dict[table_id][value_str]))
                                if num_tuples == 0:
                                    num_tuples = 1
                                fk_list = pk_list*(num_tuples // len(pk_list)) + pk_list[:num_tuples%len(pk_list)]
                                if not (value_str in generated_fk_tables[table_id]):
                                    generated_fk_tables[table_id][value_str] = fk_list
                                else: 
                                    generated_fk_tables[table_id][value_str] += fk_list

                    folder_name = self.folder_name
                    print("saving generated tables...")
                    res_file = open('./{}/{}_{}.csv'.format(folder_name, self.join_tables[self.pk_table_id], str(iter_num)), 'w', encoding="utf8")
                    header_str = self.generation_cols[self.join_tables[self.pk_table_id]+".csv"][0]
                    for col_name in self.content_cols[self.pk_table_id]:
                        header_str = header_str+",{}".format(col_name)
                    res_file.write(header_str)
                    res_file.write("\n")
                    for val in generated_pk_table:
                        values = val.split(",")
                        for i in range(len(values)):
                            if values[i].replace('.','',1).isdigit():
                                values[i] = str(int(float(values[i])))
                        for pk in generated_pk_table[val]:
                            res_file.write(str(pk) + ',' + ','.join(values) + '\n')
                    res_file.close()
                    print("generated number of distinct pk tuples: {}".format(len(generated_pk_table)))
                    total_num = 0
                    for val in generated_pk_table:
                        total_num += len(generated_pk_table[val])
                    print("generated total number of {} tuples: {}".format(self.join_tables[self.pk_table_id], total_num))

                    for table_id in fk_table_idx:
                        generated_fk_table = generated_fk_tables[table_id]

                        res_file = open('./{}/{}_{}.csv'.format(folder_name, self.join_tables[table_id], str(iter_num)), 'w', encoding="utf8")
                        header_str = self.generation_cols[self.join_tables[table_id]+".csv"][0]
                        for col_name in self.content_cols[table_id]:
                            header_str = header_str+",{}".format(col_name)
                        res_file.write(header_str)
                        res_file.write("\n")

                        total_num = 0
                        for val in generated_fk_table:
                            total_num += len(generated_fk_table[val])
                            values = val.split(",")
                            for i in range(len(values)):
                                if values[i].replace('.','',1).isdigit():
                                    values[i] = str(int(float(values[i])))
                            for fk in generated_fk_table[val]:
                                res_file.write(str(fk) + ',' + ",".join(values) + '\n')
                        print("generated total number of {} tuples: {}".format(self.join_tables[table_id], total_num))
                        res_file.close()

                    pk_column_list = []
                    for col_name in self.content_cols[self.pk_table_id]:
                        pk_column_list.append(col_name)
                    gt_table = self.loaded_tables[self.pk_table_id].data
                    gt_table = gt_table[pk_column_list]
                    pk_table_count = {}
                    for val in generated_pk_table:
                        pk_table_count[val] = len(generated_pk_table[val])
                    ce = self.AR_ComputeCE(pk_column_list, gt_table, pk_table_count, total_num, self.gt_caches)

                    print("Cross Entropy between original table and generated table: {}".format(ce))

                    current_time = time.time()
                    print("time lapsed so far: {}".format(current_time - final_time_start))

            self.model.model_bits = 0
            results = None

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
        return {}

    def stop(self):
        self.tbx_logger.flush()
        self.tbx_logger.close()

    def _log_result(self, results):
        pass

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
                'run': SAM,
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
