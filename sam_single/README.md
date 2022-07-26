# SAM for single-relation database generation
### Getting Started

**Datasets** For single-relation database, we conduct our experiments on two datasets, Census and DMV. We have uploaded Census at [`./datasets/census.csv`](./datasets/census.csv). You can download the DMV dataset by running the script.
```
bash scripts/download_dmv.sh
```
**Pretrained Models** We have provided a pretrained model for each dataset.
[`./models/census_pretrained.pt`](./models/census_pretrained.pt): Trained from 20000 queries in the generated workload ([`./queries/census_21000.txt`](./queries/census_21000.txt)).

[`./models/dmv_pretrained.pt`](./models/dmv_pretrained.pt): Trained from 20000 queries in the generated workload ([`./queries/dmv_21000.txt`](./queries/dmv_21000.txt)).

**Database Generation** To generate database from trained models using SAM, use the following commands.
```
python gen_data_model.py --dataset census --residual --layers=2 --fc-hiddens=128 --direct-io --column-masking --glob census_pretrained.pt --save-name census
python gen_data_model.py --dataset dmv --residual --layers=2 --fc-hiddens=128 --direct-io --column-masking --glob dmv_pretrained.pt --save-name dmv
```
The generated relation is saved at `./generated_data_tables`.

**Test the generated database** 


### SAM model training
SAM uses [UAE-Q](https://github.com/pagegitss/UAE) to train a deep autoregressive model from query workloads, 

To train the model from the full MSCN dataset
```
python run_uae.py --run job-light-ranges-mscn-workload
```

To test the model on sub-queries of JOB-light
```
python run_uae.py --run uae-job-light-ranges-reload
```

