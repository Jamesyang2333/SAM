import argparse
import collections
import glob
import os
import pickle
import re
import time

import numpy as np
import pandas as pd
import torch
import json
import common
import datasets
import made


dataset = 'census'

def AR_ComputeCE(gt_table, gen_total_num, eps=1e-9):
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

        ce -= gt_prob * np.log(gt_prob)

    return ce


if dataset == 'dmv':
    gt_table = datasets.LoadDmv()
    table_card = gt_table.cardinality
    csv_file = './generated_tables/dmv.csv'
    cols = [
        'Record_Type', 'Registration_Class', 'State', 'County', 'Body_Type',
        'Fuel_Type', 'Reg_Valid_Date', 'Color', 'Scofflaw_Indicator',
        'Suspension_Indicator', 'Revocation_Indicator'
    ]
    # Note: other columns are converted to objects/strings automatically.  We
    # don't need to specify a type-cast for those because the desired order
    # there is the same as the default str-ordering (lexicographical).
    type_casts = {'Reg_Valid_Date': np.datetime64}
    gt_table = gt_table.data
elif dataset == 'census':
    gt_table = datasets.LoadCensus()
    table_card = gt_table.cardinality
    csv_file = './generated_tables/census.csv'
    cols = ['c0', 'c1', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10', 'c11', 'c12', 'c13', 'c14']
    type_casts = {}
    gt_table = gt_table.data

#print(gen_table_dics)
ce = AR_ComputeCE(gt_table, table_card, eps=1e-9)

print(ce)




