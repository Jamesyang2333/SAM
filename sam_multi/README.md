# SAM for multi-relation database generation
### Instruction

**Datasets** For multi-relation database, we conduct our experiments on IMDB. User can download the dataset by running the script
```
bash scripts/download_imdb.sh
```
**Pretrained Models** We have provided two pretrained model for IMDB dataset (Job-light-ranges schema). 

[`./models/uaeq-mscn.pt`](./models/uaeq-mscn.pt): Trained from the full [MSCN](https://github.com/andreaskipf/learnedcardinalities) dataset ([`./queries/mscn_full.csv`](./queries/mscn_full.csv))

[`./models/uaeq-mscn-400.pt`](./models/uaeq-mscn.pt): Trained from 400 queries in dataset ([`./queries/mscn_400.csv`](./queries/mscn_400.csv))

**Generation from pretrained model** To generate database from pretrained models using SAM, use the following commands.
```
python run_dbgen.py --run job-light-ranges-reload 
```
By default, this generates the database using the model [`./models/uaeq-mscn-400.pt`](./models/uaeq-mscn.pt). The generation process runs for 100 iterations.

### Configuration for database generation

All configuration of SAM, including model training and database generation, can be set in [`experiments.py`](./experiments.py). To configure the generation process, locate the configure for [`'data-generation-job-light-MSCN-worklod'`], where you can set the autoregressive model to load, the number of iteration to run, as well as the schema of the generated database.

To reproduce our resutls and generate database from the pretrained model from the full MSCN dataset, set the following in [`experiments.py`](./experiments.py)
```
'checkpoint_to_load': 'models/uaeq-mscn.pt'
'total_iterations': 1000
```

*To speedup the generation process, use a larger [`'save_frequency'`].

### SAM model training
SAM uses [UAEQ](https://github.com/pagegitss/UAE) to train a deep autoregressive model from query workloads, 

To train the model from the full MSCN dataset
```
python run_uae.py --run job-light-ranges-mscn-workload
```

To test the model on sub-queries of JOB-light
```
python run_uae.py --run uae-job-light-ranges-reload
```
