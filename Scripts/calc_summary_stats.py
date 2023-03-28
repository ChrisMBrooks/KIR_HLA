#Compute Summary Stats

import pandas as pd
from Controllers.DataScienceManager import DataScienceManager as dsm
use_full_dataset = True
dsci_mgr = dsm.DataScienceManager(use_full_dataset=use_full_dataset)

phenos_df_t = dsci_mgr.data_mgr.outcomes(fill_na=False, fill_na_value=None, partition='training')
phenos_df_v = dsci_mgr.data_mgr.outcomes(fill_na=False, fill_na_value=None, partition='validation')
phenos_df = pd.concat([phenos_df_t, phenos_df_v])

phenos_df = phenos_df[phenos_df.columns[1:-2]]
columns = phenos_df.columns

records = []
for label in columns:
    column = phenos_df[label]
    original_column_size = column.shape[0]

    column = column[~column.isna()].copy()
    num_nans = original_column_size - column.shape[0]
    num_valids = original_column_size - num_nans

    min = column.min()
    max = column.max()
    mean = column.mean()
    std = column.std()
    q1=column.quantile(0.25)
    q3=column.quantile(0.75)
    IQR=q3-q1
    outliers = column[((column<(q1-1.5*IQR)) | (column>(q3+1.5*IQR)))]
    num_outliers = outliers.shape[0]
    record = [label, 
              round(float(min), 3), 
              round(float(max), 3), 
              round(float(mean), 3), 
              round(float(std), 3), 
              round(float(q1), 3), 
              round(float(q3), 3), 
              round(float(IQR),3), 
              num_outliers, 
              num_valids, 
              num_nans
            ]
 
    records.append(record)
    
columns = [
    'measurement_id', 
    'minimum', 
    'maximum', 
    'mean', 
    'std_dev', 
    'q1', 
    'q3', 
    'iqr', 
    'outliers_count', 
    'valids_count', 
    'nans_count'
    ] 

dsci_mgr.sql.insert_records(schema_name='KIR_HLA_STUDY', 
                            table_name='immunophenotype_summary_stats', 
                            column_names=columns, 
                            values= records
                            )

print('Complete')