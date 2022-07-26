# SAM for multi-relation database generation
### Getting Started

**Datasets** For multi-relation database, we conduct our experiments on IMDB. User can download the dataset by running the script
```
bash scripts/download_imdb.sh
```
**Pretrained Models** We have provided two pretrained model for IMDB dataset (Job-light-ranges schema). 

[`./models/uaeq-mscn.pt`](./models/uaeq-mscn.pt): Trained from the full [MSCN](https://github.com/andreaskipf/learnedcardinalities) workload ([`./queries/mscn_full.csv`](./queries/mscn_full.csv)).

[`./models/uaeq-mscn-400.pt`](./models/uaeq-mscn.pt): Trained from the first 400 queries in the MSCN workload ([`./queries/mscn_400.csv`](./queries/mscn_400.csv)).

**Database Generation** To generate database from trained models using SAM, use the following commands.
```
python run_dbgen.py --run job-light-ranges-reload 
```
By default, this generates the database using the model [`./models/uaeq-mscn-400.pt`](./models/uaeq-mscn.pt). The generation process runs for 100 iterations. The generated data csv files are saved at `./generated_database/imdb`.

**Test the generated database** To test the fidelity of generated database, import the files to a PostgreSQL database:
```sql
create table title (id int PRIMARY KEY, production_year int, kind_id int);
copy title from '/SAM/sam_multi/generated_database/imdb/title_100.csv' delimiter ',' header csv;

create table movie_keyword (movie_id int, keyword_id int);
copy movie_keyword from '/SAM/sam_multi/generated_database/imdb/movie_keyword_100.csv' delimiter ',' header csv;

create table movie_info_idx (movie_id int, info_type_id int);
copy movie_info_idx from 'SAM/sam_multi/generated_database/imdb/movie_info_idx_100.csv' delimiter ',' header csv;

create table movie_info (movie_id int, info_type_id int);
copy movie_info from '/SAM/sam_multi/generated_database/imdb/movie_info_100.csv' delimiter ',' header csv;

create table movie_companies (movie_id int, company_type_id int, company_id int);
copy movie_companies from '/SAM/sam_multi/generated_database/imdb/movie_companies_100.csv' delimiter ',' header csv;

create table cast_info (movie_id int, role_id int, person_id int);
copy cast_info from '/SAM/sam_multi/generated_database/imdb/cast_info_100.csv' delimiter ',' header csv;
```

Run the 400 training queries ([`./queries/mscn_400.sql`](./queries/mscn_400.sql)) on the generated database and get the result Q-error:
```
python query_execute.py --queries ./queries/mscn_400.sql --cards ./queries/mscn_400_card.csv
```

### Configuration for database generation

All configuration of SAM, including model training and database generation, can be set in [`experiments.py`](./experiments.py). To configure the generation process, locate the configure for [`'data-generation-job-light-MSCN-worklod'`], where you can set the autoregressive model to load, the number of iteration to run, as well as the schema of the generated database.

To reproduce our results and generate database using the pretrained model from the full MSCN workload, set the following in [`experiments.py`](./experiments.py)
```
'checkpoint_to_load': 'models/uaeq-mscn.pt'
'total_iterations': 1000
```

*To speedup the generation process, use a larger `'save_frequency'`.

The sample test set of 1000 queries can be found at [`./queries/mscn_sample_1000.sql`](./queries/mscn_sample_1000.sql) and [`./queries/mscn_sample_100_card.csv`](./queries/mscn_sample_1000_card.csv)

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
