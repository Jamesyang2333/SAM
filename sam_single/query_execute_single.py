import numpy as np
import torch
import pandas as pd

import datasets
import common
from data_loader import load_dataset
from evaluate import Query
from utils import get_qerror

# train_data_raw = load_dataset("census", "/home_nfs/jingyi/db_generation/queries/adultworkload_with_card_21000.txt")
train_data_raw = load_dataset("dmv", "/home_nfs/peizhi/db_gen/queries/dmvworkload_with_card_21000.txt")
# idx_list_filter = []
# # get rid of queries with filter on both column 0 and 14
# for i in range(len(train_data_raw["card"])):
#     if not ((0 in train_data_raw["column"][i]) and (14 in train_data_raw["column"][i])):
#         idx_list_filter.append(i)
# train_data_raw["card"] = [train_data_raw["card"][i] for i in idx_list_filter]
# train_data_raw["column"] = [train_data_raw["column"][i] for i in idx_list_filter]
# train_data_raw["operator"] = [train_data_raw["operator"][i] for i in idx_list_filter]
# train_data_raw["val"] = [train_data_raw["val"][i] for i in idx_list_filter]

start_idx = 20000
test_num = 1000

sample_frame = pd.read_csv("/home_nfs/peizhi/db_gen/uae/generated_data_tables/dmv_7_36.csv", header=0, names=[0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
sample_frame = sample_frame.astype({0:'float64', 4:'float64', 10:'float64', 11:'float64', 12:'float64'})
print(sample_frame.head())
print(sample_frame[0])
sample_table = datasets.LoadCensus(sample_frame)
# csv_file = "/home_nfs/jingyi/db_generation/db_gen/baseline/ms_graphical/single_relation_results/dmv_num_7.csv"
# cols = [
#         'Record Type', 'Registration Class', 'State', 'County', 'Body Type',
#         'Fuel Type', 'Reg Valid Date', 'Color', 'Scofflaw Indicator',
#         'Suspension Indicator', 'Revocation Indicator'
#     ]
#     # Note: other columns are converted to objects/strings automatically.  We
#     # don't need to specify a type-cast for those because the desired order
#     # there is the same as the default str-ordering (lexicographical).
# type_casts = {'Reg Valid Date': np.datetime64}
# sample_table = common.CsvTable('DMV_gen', csv_file, cols, type_casts)

card_pred = []
for i in range(start_idx, start_idx + test_num):
    cols = [sample_table.columns[sample_table.ColumnIndex(col)] for col in train_data_raw['column'][i]]
    ops = train_data_raw['operator'][i]
    vals = train_data_raw['val'][i]
    est = Query(sample_table, cols, ops, vals)
    card_pred.append(est / sample_table.cardinality)

q_error_list = get_qerror(np.array(card_pred), np.array(train_data_raw['card'][start_idx:start_idx+test_num]))
print("q error on generated dataset:")
print("Max q error: {}".format(np.max(q_error_list)))
print("99 percentile q error: {}".format(np.percentile(q_error_list, 99)))
print("95 percentile q error: {}".format(np.percentile(q_error_list, 95)))
print("90 percentile q error: {}".format(np.percentile(q_error_list, 90)))
print("75 percentile q error: {}".format(np.percentile(q_error_list, 75)))
print("50 percentile q error: {}".format(np.percentile(q_error_list, 50)))
print("Average q error: {}".format(np.mean(q_error_list)))

# sample_frame.to_csv('./census_results/train_num_{}.csv'.format(train_num), index=False)
