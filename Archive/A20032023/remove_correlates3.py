import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from Controllers.ConfigManager import ConfigManager as cm
from Controllers.MySQLManager import MySQLManager as msm
from Controllers.DataManager import DataManager as dtm
from Controllers.LearningManager import LearningManager as lrn

def get_correlation_dict(corr_df:pd.DataFrame, threshold:float):
    corr_dict = {}
    for row_index, row in corr_df.iterrows():
        for col_index, col in row.items():
            corr_val = row[col_index]
            if row_index != col_index and corr_val >= threshold:
                if not row_index in corr_dict:
                    corr_dict[row_index] = []
                corr_dict[row_index].append(col_index)
    return corr_dict

def overlap(A:list, B:list):
    return list(set(A) & set(B))

def remove_correlates(corr_dict:dict):
    keys_to_check = list(corr_dict.keys())
    uniques = []

    while len(keys_to_check) > 0:
        current_key = keys_to_check[-1]
        current_correlates = corr_dict[current_key]
        uniques.append(current_key)
        
        items_to_remove = [current_key] + current_correlates
        for item in items_to_remove:
            if item in keys_to_check:
                keys_to_check.remove(item)

    return uniques

print('Starting...')

#Instantiate Controllers
use_full_dataset = True
config = cm.ConfigManaager().config
sql = msm.MySQLManager(config=config)
data_mgr = dtm.DataManager(config=config, use_full_dataset=use_full_dataset)
lrn_mgr = lrn.LearningManager(config=config)

phenos_df = data_mgr.outcomes(fill_na = False, fill_na_value = 0.0, partition = 'everything')
y_labels = phenos_df.columns[1:-2]
phenos_df = phenos_df[y_labels].copy()

#clean-up 
del config, sql, data_mgr, lrn_mgr

# Prep Data
phenos_df = phenos_df.reindex(sorted(phenos_df.columns), axis=1)

print('Computing Correlation Matrix...')
corr_df = phenos_df.corr()

#clean-up 2
del phenos_df

print('Computing Correlation Dict')
threshold = 0.9
corr_dict = get_correlation_dict(corr_df, threshold)

print('Removing Correlates')
uniques = remove_correlates(corr_dict)
uniques = pd.Series(uniques, name='unlike_phenos')

date = "26022023"
filename = "Data/unlike_phenos_{}.csv".format(date)
uniques.to_csv(filename)