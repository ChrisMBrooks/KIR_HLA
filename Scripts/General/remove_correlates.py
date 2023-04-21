import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from Controllers.DataScienceManager import DataScienceManager as dsm

def get_correlation_dict(corr_df:pd.DataFrame, threshold:float):
    corr_dict = {}
    for row_index, row in corr_df.iterrows():
        if not row_index in corr_dict:
            corr_dict[row_index] = []
        for col_index, col in row.items():
            corr_val = row[col_index]
            if row_index != col_index and corr_val >= threshold:
                corr_dict[row_index].append(col_index)
    return corr_dict

def overlap(A:list, B:list):
    return list(set(A) & set(B))

def remove_correlates(corr_dict:dict, corr_df:pd.DataFrame):
    keys_to_check = list(corr_dict.keys())
    uniques = []

    while len(keys_to_check) > 0:
        current_key = keys_to_check[-1]
        current_correlates = corr_dict[current_key]
        
        rep_pheno = get_best_representative([current_key]+current_correlates, corr_df)
        
        uniques.append(rep_pheno)
        items_to_remove = [current_key] + current_correlates
        for item in items_to_remove:
            if item in keys_to_check:
                keys_to_check.remove(item)

    return uniques

def get_best_representative(similar_phenos:list, corr_df:pd.DataFrame):
    first_pheno = similar_phenos[0]

    elements_to_check = list(similar_phenos)
    threshold = 10
    while len(elements_to_check) > 0:
        subset = corr_df[elements_to_check]
        subset = subset.loc[elements_to_check]
        idx = subset.values.sum().argmax()
        candidate = subset.columns[idx]
        if get_nans_count(candidate) <= threshold:
            return candidate
        else: 
            elements_to_check.remove(candidate)
    
    return first_pheno

def get_nans_count(pheno:str):
    match = phenos_stats[phenos_stats['measurement_id']==pheno]
    return match['nans_count'].values[0] 

#Instantiate Controllers
use_full_dataset = True
use_database = False
data_sci_mgr = dsm.DataScienceManager(
    use_full_dataset=use_full_dataset, 
    use_database=use_database
)

filename = 'Data/immunophenotype_summary_stats.csv'
phenos_stats = pd.read_csv(filename, index_col=0)

# Prep Data
phenos_df = data_sci_mgr.data_mgr.outcomes(fill_na=False, partition='everything')
y_labels = phenos_df.columns[1:-2]
phenos_df = phenos_df[y_labels].copy()
phenos_df = phenos_df.reindex(sorted(phenos_df.columns), axis=1)

print('Computing Correlation Matrix...')
corr_df = phenos_df.corr()

# Clean-up 1
del phenos_df

print('Computing Correlation Dict')
threshold = 0.95
corr_dict = get_correlation_dict(corr_df, threshold)
corr_look_up_tbl = [[key, corr_dict[key]] for key in corr_dict]
corr_look_up_tbl_df = pd.DataFrame(corr_look_up_tbl, columns=['label', 'correlates'])

print('Exporting Correlation Dict')
date_str = data_sci_mgr.data_mgr.get_date_str()
filename = "Data/phenos_corr_dict_{}_{}.parquet".format(threshold, date_str)
corr_look_up_tbl_df.to_parquet(filename)

corr_look_up_tbl_df = pd.read_parquet(filename)
print(corr_look_up_tbl_df)

print('Removing Correlates')
uniques = remove_correlates(corr_dict, corr_df)
uniques = pd.Series(uniques)

print('Exporting unlikes')
filename = "Data/unlike_phenos_{}.csv".format(date_str)
uniques.to_csv(filename)
print('Complete')