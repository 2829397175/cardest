"""Dataset registrations."""
from ctypes import Union
import os

import numpy as np

import common
import pandas as pd

def LoadDmv(filename='dmv_ofnan.csv',name='DMV_ofnan'):
    csv_file = './datasets/{}'.format(filename)
    cols = [
        'Record Type', 'Registration Class', 'State', 'County', 'Body Type',
        'Fuel Type', 'Reg Valid Date', 'Color', 'Scofflaw Indicator',
        'Suspension Indicator', 'Revocation Indicator'
    ]
    # cols = [
    #     'Record Type', 'Registration Class', 'State', 'County', 'Body Type',
    #     'Fuel Type', 'Color', 'Scofflaw Indicator',
    #     'Suspension Indicator', 'Revocation Indicator'
    # ]
    # Note: other columns are converted to objects/strings automatically.  We
    # don't need to specify a type-cast for those because the desired order
    # there is the same as the default str-ordering (lexicographical).
    type_casts = {'Reg Valid Date': np.datetime64}
    
    return common.CsvTable(name, csv_file, cols, type_casts)

def UpdateDmv(insert_table:common.CsvTable,
              old_table,
              name='DMV'):
    
    cols = [
        'Record Type', 'Registration Class', 'State', 'County', 'Body Type',
        'Fuel Type', 'Reg Valid Date', 'Color', 'Scofflaw Indicator',
        'Suspension Indicator', 'Revocation Indicator'
    ]
    # Note: other columns are converted to objects/strings automatically.  We
    # don't need to specify a type-cast for those because the desired order
    # there is the same as the default str-ordering (lexicographical).
    type_casts = {'Reg Valid Date': np.datetime64}
    if isinstance(old_table,common.CsvTable):
        df_concat = pd.concat([old_table.data,insert_table.data])
        name=old_table.name+"_"+insert_table.name
    else:
        df_concat = pd.concat([old_table,insert_table.data])
        name=name+"_"+insert_table.name
    new_table = common.CsvTable(name, df_concat, cols, type_casts)
    return new_table


def DF_to_CSVtable_Dmv(df,name='DMV'):
    cols = [
        'Record Type', 'Registration Class', 'State', 'County', 'Body Type',
        'Fuel Type', 'Reg Valid Date', 'Color', 'Scofflaw Indicator',
        'Suspension Indicator', 'Revocation Indicator'
    ]
    # Note: other columns are converted to objects/strings automatically.  We
    # don't need to specify a type-cast for those because the desired order
    # there is the same as the default str-ordering (lexicographical).
    type_casts = {'Reg Valid Date': np.datetime64}
    new_table = common.CsvTable(name, df, cols, type_casts)
    return new_table



def LoadAdult(filename='adult.csv'):
    csv_file = './datasets/{}'.format(filename)
    cols =[0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    type_casts = {}
    return common.CsvTable('Adult', csv_file, cols, type_casts, header=None)


def LoadCup98(filename='cup.csv'):
    csv_file = './datasets/{}'.format(filename)
    cols = [473, 5, 9, 10, 11, 12, 52, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 74, 6, 17, 18, 361, 19, 20, 21, 22, 42, 73, 470, 472, 475, 477, 50, 476, 478, 318, 25, 51, 54, 408, 474, 358, 384, 14, 317, 326, 350, 161, 234, 148, 324, 13, 147, 88, 299, 316, 397, 410, 288, 233, 258, 183, 325, 387, 182, 307, 357, 314, 322, 92, 304, 323, 210, 93, 464, 94, 255, 271, 15, 219, 259, 3, 294, 359, 96, 209, 336, 265, 319, 360, 272, 277]
    type_casts = {}
    return common.CsvTable('Cup98', csv_file, cols, type_casts, header=None)

def LoadDataset(dataset):
    func_dict={
        "adult":LoadAdult,
        "cup":LoadCup98
    }
    return func_dict[dataset]()