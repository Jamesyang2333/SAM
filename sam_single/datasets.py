"""Dataset registrations."""
import os

import numpy as np
import pandas as pd

import common

def LoadDmv(filename='Vehicle__Snowmobile__and_Boat_Registrations.csv'):
    csv_file = './datasets/{}'.format(filename)
    cols = [
        'Record Type','Registration Class', 'State', 'County', 'Body Type',
        'Fuel Type', 'Reg Valid Date', 'Color', 'Scofflaw Indicator',
        'Suspension Indicator', 'Revocation Indicator'
    ]
    # Note: other columns are converted to objects/strings automatically.  We
    # don't need to specify a type-cast for those because the desired order
    # there is the same as the default str-ordering (lexicographical).
    type_casts = {'Reg Valid Date': np.datetime64}
    return common.CsvTable('DMV', csv_file, cols, type_casts)

# def LoadCensus(filename='census.csv'):
#     csv_file = '../datasets/{}'.format(filename)
#     cols =[0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
#     type_casts = {}
#     return common.CsvTable('Adult', csv_file, cols, type_casts, header=None)
def LoadCensus(filename_or_df='census.csv'):
    if isinstance(filename_or_df, str):
        filename_or_df = './datasets/{}'.format(filename_or_df)
    else:
        assert (isinstance(filename_or_df, pd.DataFrame))
    cols =[0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    type_casts = {}
    return common.CsvTable('Census', filename_or_df, cols, type_casts, header=None)

