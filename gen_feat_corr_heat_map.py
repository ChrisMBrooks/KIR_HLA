
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
date_str = '06042023'
filename = 'Analysis/Multivariate/{}/multivar_qc_fs_bs_candidate_features_{}.csv'.format(date_str, date_str) 
phenos_subset = pd.read_csv(filename, index_col=0)
indeces = phenos_subset.values[:,1:3].sum(axis=1)
indeces = np.where(indeces >= 1)
phenos_to_plot = list(phenos_subset.iloc[indeces]['label'].values)

phenos_to_plot = ['P1:20365', 'MFI:469', 'MFI:8',
'P7 Mono:1287', 'P4:5290', 'P1:10321']

phenos_df_t = data_sci_mgr.data_mgr.outcomes(fill_na=False, fill_na_value=None, partition='training')
phenos_df_v = data_sci_mgr.data_mgr.outcomes(fill_na=False, fill_na_value=None, partition='validation')

scores_df_t = data_sci_mgr.data_mgr.features(fill_na=False, fill_na_value=None, partition='training')
scores_df_v = data_sci_mgr.data_mgr.features(fill_na=False, fill_na_value=None, partition='validation')

phenos_df = pd.concat([phenos_df_t, phenos_df_v])
scores_df = pd.concat([scores_df_t, scores_df_v])

data = scores_df.merge(phenos_df, how='inner', on='public_id')
data = data[phenos_to_plot].copy().corr(method='pearson')
sns.heatmap(data, annot = True)
plt.title('Correlation Heat Map')
date_str = data_sci_mgr.data_mgr.get_date_str()

filename = 'Analysis/Multivariate/{}/heatmap_{}_{}.png'.format(date_str, phenos_to_plot[0], date_str)
plt.savefig(filename)
plt.show()