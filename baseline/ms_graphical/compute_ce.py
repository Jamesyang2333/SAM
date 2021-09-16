import time

import numpy as np
import pandas as pd
import common
import datasets


dataset = 'census'

def AR_ComputeCE(gt_table, gen_table_dics, gen_total_num, eps=1e-15):
    if dataset == 'dmv':
        col_names = [
            'Record Type', 'Registration Class', 'State', 'County', 'Body Type',
            'Fuel Type', 'Reg Valid Date', 'Color', 'Scofflaw Indicator',
            'Suspension Indicator', 'Revocation Indicator'
        ]
    elif dataset == 'census':
        col_names = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    gt_table = gt_table.fillna(-1)
    print ('start group by')
    unique_rows = list(gt_table.groupby(col_names).groups)
    ce = 0.


    print ('start group by for gt counts')
    gt_counts_df = gt_table.groupby(col_names).size().reset_index(name='counts')

    for row in unique_rows:
        value = list(row)
        value_str = [str(i) for i in value]
        value_str = ','.join(value_str)

        gt_prob = gt_counts_df[gt_counts_df[col_names[0]] == value[0]]
        for i in range(len(col_names) - 1):
            gt_prob = gt_prob[gt_prob[col_names[i + 1]] == value[i + 1]]
        gt_prob = gt_prob.iloc[0]['counts'] / len(gt_table)

        # value_id_format = []
        # for i, col_value in enumerate(value):
        #     value_id_format.append(look_up_list[i][col_value])
        # value_id_format = ','.join(value_id_format)
        if value_str in gen_table_dics:
            gen_prob = gen_table_dics[value_str] / gen_total_num + eps
        else:
            gen_prob = eps
        ce -= gt_prob * np.log(gen_prob)

    return ce


if dataset == 'dmv':
    gt_table = datasets.LoadDmv()
    table_card = gt_table.cardinality
    csv_file = './generated_tables/dmv.csv'
    cols = [
        'Record Type', 'Registration Class', 'State', 'County', 'Body Type',
        'Fuel Type', 'Reg Valid Date', 'Color', 'Scofflaw Indicator',
        'Suspension Indicator', 'Revocation Indicator'
    ]
    # Note: other columns are converted to objects/strings automatically.  We
    # don't need to specify a type-cast for those because the desired order
    # there is the same as the default str-ordering (lexicographical).
    type_casts = {'Reg Valid Date': np.datetime64}
    gen_table = common.CsvTable('DMV_gen', csv_file, cols, type_casts).data
    gt_table = gt_table.data
elif dataset == 'census':
    gt_table = datasets.LoadCensus()
    table_card = gt_table.cardinality
    csv_file = './single_relation_results/census_num_12.csv'
    # # cols = ['c0', 'c1', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10', 'c11', 'c12', 'c13', 'c14']
    # cols = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    # type_casts = {}
    # gen_table = common.CsvTable('Census_gen', csv_file, cols, type_casts).data
    sample_frame = pd.read_csv("/home_nfs/jingyi/db_generation/db_gen/baseline/ms_graphical/single_relation_results/train_num_13.csv", header=0, names=[0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
    sample_frame = sample_frame.astype({0:'float64', 4:'float64', 10:'float64', 11:'float64', 12:'float64'})
    print(sample_frame.head())
    print(sample_frame[0])
    gen_table = datasets.LoadCensus(sample_frame).data
    gt_table = gt_table.data

gen_table = gen_table.fillna(-1)
gen_table_dics = {}
i=0
for tuple in gen_table.itertuples():
    value = list(tuple)
    value = value[1:len(value)]
    if i==0:
        print (tuple)
    value_str = [str(i) for i in value]
    value_str = ','.join(value_str)
    if value_str in gen_table_dics:
        gen_table_dics[value_str] += 1
    else:
        gen_table_dics[value_str] = 1
    i += 1

print(gen_table_dics)
ce = AR_ComputeCE(gt_table, gen_table_dics, table_card, eps=1e-9)

print(ce)