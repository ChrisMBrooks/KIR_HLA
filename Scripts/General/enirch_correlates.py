
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
data_sci_mgr = dsm.DataScienceManager(
    use_full_dataset=use_full_dataset, 
    use_database=use_database
)


#Import Data
filename = 'Analysis/Multivariate/10042023/candidate_summary_stats_10042023.csv' 
phenos_candidates = pd.read_csv(filename, index_col=0)

filename = '/Users/chrismbrooks/Documents/Imperial/Asquith Group/KIR_HLA/Data/phenos_corr_dict_0.95_05042023.parquet'
phenos_correlates = pd.read_parquet(filename)

pheno_ref_data = data_sci_mgr.sql.read_table_into_data_frame(
    schema_name='KIR_HLA_STUDY', 
    table_name='immunophenotype_definitions'
)

#Transform Data
phenos_candidates = phenos_candidates[['phenotype_id', 'marker_definition', 'parent_population']].copy()
phenos_candidates = phenos_candidates.merge(
    phenos_correlates, how='left', left_on='phenotype_id', right_on='label'
)
phenos_candidates = phenos_candidates[['phenotype_id', 'correlates']].copy()

mstr_corr_df = pd.DataFrame()
for index, row in phenos_candidates.iterrows():
    cluster_rep = row['phenotype_id']
    correlates = pd.DataFrame([cluster_rep] + list(row['correlates']), columns=['phenotype_id'])
    correlates['cluster_rep'] = cluster_rep

    if mstr_corr_df.empty:
        mstr_corr_df = correlates
    else: 
        mstr_corr_df = pd.concat([mstr_corr_df, correlates])

mstr_corr_df = mstr_corr_df.merge(pheno_ref_data, how='left', left_on='phenotype_id', right_on='phenotype_id')

#Export Data
date_str = data_sci_mgr.data_mgr.get_date_str()
filename = 'Analysis/Multivariate/{}/candidate_correlates_summary_{}.csv'.format('10042023','10042023')
mstr_corr_df.to_csv(filename)

