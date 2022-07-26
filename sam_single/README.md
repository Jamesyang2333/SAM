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

**Test the generated database** Run 1000 test queries on the generated database. The last 1000 queries in the generated workload are test queries.
```
python query_execute_single.py --dataset census
python query_execute_single.py --dataset dmv
```


### SAM model training
SAM uses [UAE-Q](https://github.com/pagegitss/UAE) to train a deep autoregressive model from query workloads, 

To train the model from the full MSCN dataset
```
python train_uae.py --num-gpus=1 --dataset=census --epochs=50 --constant-lr=5e-4 --run-uaeq  --residual --layers=2 --fc-hiddens=128 --direct-io --column-masking --workload-size 20000 --q-bs 200
python train_uae.py --num-gpus=1 --dataset=census --epochs=50 --constant-lr=5e-4 --run-uaeq  --residual --layers=2 --fc-hiddens=128 --direct-io --column-masking --workload-size 20000 --q-bs 200
```

To test the model
```
python eval_model.py --dataset census --residual --layers=2 --fc-hiddens=128 --direct-io --column-masking --glob dmv_pretrained.pt
python eval_model.py --dataset dmv --residual --layers=2 --fc-hiddens=128 --direct-io --column-masking --glob census_pretained.pt
```

