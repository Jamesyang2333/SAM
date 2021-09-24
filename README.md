# SAM
Source code of SAM: Database Generation from Query Workload with Supervised Autoregressive Model

### Main Directories
[`./sam_single`](./sam_single): the code of sam for single-relation database generation

[`./sam_multi`](./sam_multi): the code of sam for multiple-relation database generation

### Datasets
We conducted experiments on three datasets: Census, DMV and IMDB. Census is already uploaded in `datasets/census.csv`. Due to storage space limit, we didn't upload DMV and IMDB. Users can download the two datasets by executing `scripts/download_dmv.sh` and `scripts/downloa_imdb.sh` respectively.

### Constructing SAM
To construct and train the deep autoregressive model of SAM from query workloads, one can refer to the published code of UAE(https://github.com/pagegitss/UAE)
