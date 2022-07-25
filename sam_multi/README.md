# SAM for multi-relation database generation
## Dataset
For multi-relation database, we conduct our experiments on IMDB. User can download the dataset by running the script
```
bash scripts/download_imdb.sh
```
## Instruction
We have provided two pretrained model for IMDB dataset. 
[`./models/uaeq-mscn.pt`](./models/uaeq-mscn.pt): Trained from the full [MSCN](https://github.com/andreaskipf/learnedcardinalities) dataset at  [`./queries/mscn_full.csv`](./queries/mscn_full.csv)
[`./models/uaeq-mscn-400.pt`](./models/uaeq-mscn.pt): Trained from 400 queries in [MSCN](https://github.com/andreaskipf/learnedcardinalities) dataset at [`./queries/mscn_400.csv`](./queries/mscn_400.csv)

[`./sam_multi`](./sam_multi): SAM for multi-relation database generation


To generate database from pretrained models using SAM, use the following commands.
```
python run_dbgen.py --run job-light-ranges-reload 
```

One can modify `experiments.py` to select the model to be used.
