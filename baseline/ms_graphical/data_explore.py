import numpy as np
import torch
import pandas as pd
import json

import datasets
import common
from data_loader import load_dataset
from evaluate import Query
from utils import get_qerror

train_data_raw = load_dataset("census", "/home_nfs/jingyi/db_generation/queries/censusworkload_with_card_100000.txt")
print(len(train_data_raw['card']))



# data = {}
# data['query_list'] = []
# data['card_list'] = []

# file_name = '/home_nfs/jingyi/db_generation/queries/censusworkload_with_card_21000.txt'
# with open(file_name, 'r', encoding="utf8") as f:
#     workload_stats = json.load(f)
#     data['query_list'] = data['query_list'] + workload_stats['query_list'][:20000]
#     data['card_list'] = data['card_list'] + workload_stats['card_list'][:20000]
    
# file_name = '/home_nfs/jingyi/db_generation/queries/adultworkload_with_card_100000_start_id_21000.txt'
# with open(file_name, 'r', encoding="utf8") as f:
#     workload_stats = json.load(f)
#     data['query_list'] = data['query_list'] + workload_stats['query_list']
#     data['card_list'] = data['card_list'] + workload_stats['card_list']

# file_name = '/home_nfs/jingyi/db_generation/queries/adultworkload_with_card_101000_start_id_100000.txt'
# with open(file_name, 'r', encoding="utf8") as f:
#     workload_stats = json.load(f)
#     data['query_list'] = data['query_list'] + workload_stats['query_list']
#     data['card_list'] = data['card_list'] + workload_stats['card_list']

# print(len(data['card_list']))
# file_name = '/home_nfs/jingyi/db_generation/queries/censusworkload_with_card_100000.txt'
# with open(file_name, 'w') as f:
#     json.dump(data, f)
    