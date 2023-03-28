#Find correlates for a given immunophenotype label

import pandas as pd
from Controllers.DataScienceManager import DataScienceManager as dsm
use_full_dataset = True
dsci_mgr = dsm.DataScienceManager(use_full_dataset=use_full_dataset)

phenos_df_t = dsci_mgr.data_mgr.outcomes(fill_na=False, fill_na_value=None, partition='training')
phenos_df_v = dsci_mgr.data_mgr.outcomes(fill_na=False, fill_na_value=None, partition='validation')
phenos_df = pd.concat([phenos_df_t, phenos_df_v])

target_lbl = 'MFI:469'
columns = phenos_df.columns[1:-2]
cut_off = 0.7

correlates = []
for index, label_i in enumerate(columns):

    C = phenos_df[[target_lbl, label_i]].copy()
    p_corr = C.corr(method='pearson').values[1,0]
    if label_i != target_lbl and p_corr > cut_off:
        correlates.append(label_i)

pheno_definitions = dsci_mgr.sql.read_table_into_data_frame(schema_name='KIR_HLA_STUDY', 
    table_name='immunophenotype_definitions'
)

correlates_df = pd.Series(correlates, name='phenotype_id')
correlates_df = pheno_definitions.merge(correlates_df, how='right', on='phenotype_id')

print(correlates_df)