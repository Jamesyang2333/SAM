import numpy as np
import torch
import pandas as pd

import datasets
import common
from data_loader import load_dataset
from evaluate import Query
from utils import get_qerror

# train_data_raw = load_dataset("census", "/home_nfs/jingyi/db_generation/queries/censusworkload_with_card_21000.txt")
train_data_raw = load_dataset("dmv", "/home_nfs/jingyi/db_generation/queries/dmvworkload_with_card_21000.txt")

start_idx = 0
test_num = 20000

# sample_frame = pd.read_csv("/home_nfs/jingyi/db_generation/db_gen/baseline/ms_graphical/single_relation_results/census_num_12.csv", header=0, names=[0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
# sample_frame = pd.read_csv("/home_nfs/jingyi/db_generation/db_gen/uae/generated_data_tables/census_40000_10.csv", header=0, names=[0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
# sample_frame = sample_frame.astype({0:'float64', 4:'float64', 10:'float64', 11:'float64', 12:'float64'})
# print(sample_frame.head())
# print(sample_frame[0])
# sample_table = datasets.LoadCensus(sample_frame)
# csv_file = "/home_nfs/jingyi/db_generation/db_gen/baseline/ms_graphical/single_relation_results/dmv_num_7.csv"
csv_file = "/home_nfs/jingyi/db_generation/db_gen/uae/generated_data_tables/dmv_20000_0005_20.csv"
cols = [
        'Record Type', 'Registration Class', 'State', 'County', 'Body Type',
        'Fuel Type', 'Reg Valid Date', 'Color', 'Scofflaw Indicator',
        'Suspension Indicator', 'Revocation Indicator'
    ]
    # Note: other columns are converted to objects/strings automatically.  We
    # don't need to specify a type-cast for those because the desired order
    # there is the same as the default str-ordering (lexicographical).
type_casts = {'Reg Valid Date': np.datetime64}
sample_table = common.CsvTable('DMV', csv_file, cols, type_casts)

card_pred = []
for i in range(start_idx, start_idx + test_num):
    if i % 1000 == 0:
        print(i)
    cols = [sample_table.columns[sample_table.ColumnIndex(col)] for col in train_data_raw['column'][i]]
    ops = train_data_raw['operator'][i]
    vals = train_data_raw['val'][i]
    est = Query(sample_table, cols, ops, vals)
    if est == 0:
        est = 1
    card_pred.append(est / sample_table.cardinality)

q_error_list = get_qerror(np.array(card_pred), np.array(train_data_raw['card'][start_idx:start_idx+test_num]))
print("q error on generated dataset:")
print("Max q error: {}".format(np.max(q_error_list)))
# idx = np.argmax(q_error_list)
# print("Index of max error query: {}".format(idx))
# print(train_data_raw['operator'][start_idx+idx])
# print(train_data_raw['val'][start_idx+idx])
# print(train_data_raw['card'][start_idx+idx])
# print(card_pred[idx])
print("99 percentile q error: {}".format(np.percentile(q_error_list, 99)))
print("95 percentile q error: {}".format(np.percentile(q_error_list, 95)))
print("90 percentile q error: {}".format(np.percentile(q_error_list, 90)))
print("75 percentile q error: {}".format(np.percentile(q_error_list, 75)))
print("50 percentile q error: {}".format(np.percentile(q_error_list, 50)))
print("Average q error: {}".format(np.mean(q_error_list)))

# sample_frame.to_csv('./census_results/train_num_{}.csv'.format(train_num), index=False)

