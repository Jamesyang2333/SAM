# SAM
SAM is a learning-based method for high-fidelity database generation using deep autoregressive models.

Your can learn more about SAM in our SIGMOD 2022 paper, [SAM: Database Generation from Query Workloads with Supervised Autoregressive Models](https://dl.acm.org/doi/abs/10.1145/3514221.3526168).

## Getting Started
This project contains two main directories:

[`./sam_single`](./sam_single): SAM for single-relation database generation

[`./sam_multi`](./sam_multi): SAM for multi-relation database generation

Here we give a quick example of using SAM to generate the IMDB database from pre-trained autoregressive model. More detailed instructions on SAM can be found in the README of the respective directories.

Set up the conda environment for the project:
```
conda env create -f environment.yml
conda activate sam
```

Enter the directory and download the IMDB database:
```
cd sam_multi
bash scripts/download_imdb.sh
```

Generate the IMDB database using the pretrained model at [`./sam_multi/models/uaeq-mscn-400.pt`](./sam_multi/models/uaeq-mscn-400.pt). The model is trained from the first 400 queries in the MSCN workload. The generated data csv files are saved at `./sam_multi/generated_database/imdb`.
```
python run_dbgen.py --run data-generation-job-light-MSCN-worklod
```

To test the fidelity of generated database, import the files to a PostgreSQL database:
```sql
create table title (id int PRIMARY KEY, production_year int, kind_id int);
copy title from 'SAM/sam_multi/generated_database/imdb/title_100.csv' delimiter ',' header csv;

create table movie_keyword (movie_id int, keyword_id int);
copy movie_keyword from 'SAM/sam_multi/generated_database/imdb/movie_keyword_100.csv' delimiter ',' header csv;

create table movie_info_idx (movie_id int, info_type_id int);
copy movie_info_idx from 'SAM/sam_multi/generated_database/imdb/movie_info_idx_100.csv' delimiter ',' header csv;

create table movie_info (movie_id int, info_type_id int);
copy movie_info from 'SAM/sam_multi/generated_database/imdb/movie_info_100.csv' delimiter ',' header csv;

create table movie_companies (movie_id int, company_type_id int, company_id int);
copy movie_companies from 'SAM/sam_multi/generated_database/imdb/movie_companies_100.csv' delimiter ',' header csv;

create table cast_info (movie_id int, role_id int, person_id int);
copy cast_info from 'SAM/sam_multi/generated_database/imdb/cast_info_100.csv' delimiter ',' header csv;
```

Run the 400 training queries on the generated database and get the result Q-error:
```
python query_execute.py --queries ./queries/mscn_400.sql --cards ./queries/mscn_400_card.csv
```

## Citation
```bibtex
@inproceedings{
  title={SAM: Database Generation from Query Workloads with Supervised Autoregressive Models},
  author={Yang, Jingyi and Wu, Peizhi and Cong, Gao and Zhang, Tieying and He, Xiao},
  booktitle={Proceedings of the 2022 International Conference on Management of Data},
  pages={1542--1555},
  year={2022},
  location = {Philadelphia, PA, USA},
  publisher = {Association for Computing Machinery}
}
```

## Acknowledgements
This project builds on top of [UAE](https://github.com/pagegitss/UAE) and [NeuroCard](https://github.com/neurocard/neurocard).
