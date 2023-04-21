
import pandas as pd
from Controllers.DataScienceManager import DataScienceManager as dsm

use_full_dataset=True
data_sci_mgr = dsm.DataScienceManager(use_full_dataset=use_full_dataset)

scores = data_sci_mgr.data_mgr.features(fill_na=False, fill_na_value=None, partition='everything')

filename = 'Data/becca_twinsuk_analysis.csv'
beccas_scores = pd.read_csv(filename, index_col=0) 
columns = ['PublicID', 'Func_iKIR_count',	'Func_iKIR_score', 'iKIR_count']
beccas_scores = beccas_scores[columns].copy()

scores = scores.merge(beccas_scores, how='right', left_on='public_id', right_on='PublicID')

filename = 'QC/chris_becca_ikir_score_cross_check.csv'
scores.to_csv(filename)