#Find correlates for a given immunophenotype label

import pandas as pd
from Controllers.DataScienceManager import DataScienceManager as dsm

def find_correlates(target_lbl:str, cut_off:float, phenos_df:pd.DataFrame,  pheno_definitions:pd.DataFrame):
    correlates = []
    columns = phenos_df.columns[1:-2]
    for index, label_i in enumerate(columns):
        if label_i != target_lbl:
            C = phenos_df[[target_lbl, label_i]].copy()
            p_corr = C.corr(method='pearson').values[1,0]
            if label_i != target_lbl and p_corr > cut_off:
                correlates.append(label_i)

    data = {'phenotype_id':[target_lbl],'correlates':[correlates]}
    correlates_df = pd.DataFrame(data, columns=['phenotype_id', 'correlates'])  
    correlates_df = pheno_definitions.merge(correlates_df, how='right', on='phenotype_id')
    correlates_df['threshold'] = cut_off
    return correlates_df

#Instantiate Controllers
use_full_dataset = True
data_sci_mgr = dsm.DataScienceManager(use_full_dataset=use_full_dataset)

#Import Data
phenos_df_t = data_sci_mgr.data_mgr.outcomes(fill_na=False, fill_na_value=None, partition='training')
phenos_df_v = data_sci_mgr.data_mgr.outcomes(fill_na=False, fill_na_value=None, partition='validation')
phenos_df = pd.concat([phenos_df_t, phenos_df_v])

pheno_definitions = data_sci_mgr.sql.read_table_into_data_frame(schema_name='KIR_HLA_STUDY', 
    table_name='immunophenotype_definitions'
)

univar_ols_results = data_sci_mgr.sql.read_table_into_data_frame(schema_name='KIR_HLA_STUDY', 
    table_name='model_result_ols'
)
univar_ols_results = univar_ols_results.rename(columns={'feature_name':'phenotype_id'})
univar_ols_results = univar_ols_results[['phenotype_id', 'p_1']].copy()

#Find Correlates
target_lbls = ['MFI:469']
cut_off = 0.95
correlate_df = [find_correlates(target_lbl, cut_off, phenos_df, pheno_definitions) for target_lbl in target_lbls]
correlate_df = pd.concat(correlate_df)
print(correlate_df)
correlate_df = correlate_df.merge(univar_ols_results, how='left', on='phenotype_id')

#Export Data
date_str = data_sci_mgr.data_mgr.get_date_str()
filename = "Analysis/General/candidate_feature_correlates_{}.parquet".format(date_str)
correlate_df.to_parquet(filename)

#Check Export
correlate_df = pd.read_parquet(filename)
print(correlate_df[['phenotype_id', 'correlates']])