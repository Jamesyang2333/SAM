"""Dataset registrations."""
import os

import numpy as np
import pandas as pd

import common


def LoadDmv(filename='Vehicle__Snowmobile__and_Boat_Registrations.csv'):
    csv_file = '../datasets/{}'.format(filename)
    cols = [
        'Record Type', 'Registration Class', 'State', 'County', 'Body Type',
        'Fuel Type', 'Reg Valid Date', 'Color', 'Scofflaw Indicator',
        'Suspension Indicator', 'Revocation Indicator'
    ]
    # Note: other columns are converted to objects/strings automatically.  We
    # don't need to specify a type-cast for those because the desired order
    # there is the same as the default str-ordering (lexicographical).
    type_casts = {'Reg Valid Date': np.datetime64}
    return common.CsvTable('DMV', csv_file, cols, type_casts)

def LoadCovtype(filename='covtype.csv'):
    csv_file = '../datasets/{}'.format(filename)
    cols = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,54]
    type_casts = {}
    return common.CsvTable('Covtype', csv_file, cols, type_casts, header=None)

def LoadKddcup(filename='kddcup.csv'):
    csv_file = '../datasets/{}'.format(filename)
    #cols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41]
    cols = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]
    type_casts = {}
    return common.CsvTable('Kddcup', csv_file, cols, type_casts, header=None)

def LoadCensus(filename_or_df='census.csv'):
    if isinstance(filename_or_df, str):
        filename_or_df = '../../../datasets/{}'.format(filename_or_df)
    else:
        assert (isinstance(filename_or_df, pd.DataFrame))
    cols =[0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    type_casts = {}
    return common.CsvTable('Census', filename_or_df, cols, type_casts, header=None)

def LoadPoker(filename='poker.csv'):
    csv_file = '../datasets/{}'.format(filename)
    cols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    type_casts = {}
    return common.CsvTable('Poker', csv_file, cols, type_casts, header=None)
