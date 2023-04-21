
import math, random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme()

import pandas as pd
from Controllers.DataScienceManager import DataScienceManager as dsm

#Instantiate Controllers
use_full_dataset = True
use_database = False

cluster_cut_off = 0.9

data_sci_mgr = dsm.DataScienceManager(
    use_full_dataset=use_full_dataset, 
    use_database=use_database
)

filename = 'Analysis/Multivariate/11042023_c_rc2/multivar_qc_fs_bs_candidate_features_11042023.csv' 
phenos_subset = pd.read_csv(filename, index_col=0)
indeces = phenos_subset.values[:,1:3].sum(axis=1)
indeces = np.where(indeces >= 1)
candidates = pd.Series(phenos_subset.iloc[indeces]['label'].values, name='phenotype_id')

if cluster_cut_off == 0.95:
    filename = 'Data/phenos_corr_dict_0.95_05042023.parquet'
elif cluster_cut_off == 0.9:
    filename = 'Data/phenos_corr_dict_0.9_06042023.parquet'
elif cluster_cut_off == 0.7:
    filename = 'Data/phenos_corr_dict_0.7_06042023.parquet'
correlates = pd.read_parquet(filename)

pheno_ref_data = data_sci_mgr.sql.read_table_into_data_frame(schema_name='KIR_HLA_STUDY', table_name='immunophenotype_definitions')
ols_stats = data_sci_mgr.sql.read_table_into_data_frame(schema_name='KIR_HLA_STUDY', table_name='model_result_ols')
pheno_data_stats = data_sci_mgr.sql.read_table_into_data_frame(schema_name='KIR_HLA_STUDY', table_name='immunophenotype_summary_stats')

candidates = pheno_ref_data.merge(candidates, how='right', left_on='phenotype_id', right_on='phenotype_id')
candidates = candidates.merge(ols_stats, how='left', left_on='phenotype_id', right_on='feature_name')
candidates = candidates.merge(pheno_data_stats, how='left', left_on='phenotype_id', right_on='measurement_id')
candidates = candidates.merge(correlates, how='left', left_on='phenotype_id', right_on='label')

date_str = data_sci_mgr.data_mgr.get_date_str()
filename = 'Analysis/Multivariate/candidate_summary_stats_{}.csv'.format(date_str)
candidates.to_csv(filename)

