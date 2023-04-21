# Using Forward Selection & Backward Selection to stress test the "significant" immunophenotypes

import time, uuid
import numpy as np
import pandas as pd

from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SequentialFeatureSelector

from Controllers.DataScienceManager import DataScienceManager as dsm

print('Starting...')
start_time = time.time()
run_id = str(uuid.uuid4().hex)

#Instantiate Controllers
use_full_dataset=True
use_database = False
data_sci_mgr = dsm.DataScienceManager(
    use_full_dataset=use_full_dataset, 
    use_database=use_database
)

#Declare Subset of Immunophenotypes
filename = 'Analysis/ElasticNet/11042023_c_rc3/en_results_candidate_phenos_11042023.csv'
phenos_subset = list(pd.read_csv(filename, index_col=0).values[:, 0])

scores = data_sci_mgr.data_mgr.features(fill_na=False, fill_na_value=None, partition='training')
phenos = data_sci_mgr.data_mgr.outcomes(fill_na=False, fill_na_value=None, partition='training')
phenos = phenos[phenos_subset]

# Standardise Data
scores = scores['f_kir_score'].values.reshape(-1,1)
if len(phenos.values.shape) == 1:
    phenos = phenos.values.reshape(-1,1)
else:
    phenos = phenos.values[:, 0:]

phenos, scores = data_sci_mgr.data_mgr.preprocess_data(
    X=phenos, Y=scores, impute = True, standardise = True, 
    normalise = True, strategy='median'
)

scores = scores.ravel()

# Fit the Model 
n_splits = 4
n_repeats = 20
cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1)

model = LinearRegression()

sfs_for = SequentialFeatureSelector(model, direction='forward', scoring='neg_mean_absolute_error', 
    cv=cv, n_jobs=-1,
)
sfs_for.fit(phenos, scores)
print('Forward Selection Complete.')

sfs_bac = SequentialFeatureSelector(model, direction='backward', scoring='neg_mean_absolute_error', 
    cv=cv, n_jobs=-1)
sfs_bac.fit(phenos, scores)
print('Backward Selection Complete.')

for_selected_features = sfs_for.get_support()
bac_selected_features = sfs_bac.get_support()

summary = [[phenos_subset[i], for_selected_features[i], bac_selected_features[i]] for i in range(0, len(phenos_subset), 1)]
summary_df = pd.DataFrame(summary, columns=['label', 'forward_selected', 'backward_selected'])

date_str = data_sci_mgr.data_mgr.get_date_str()
filename = "Analysis/Multivariate/multivar_qc_fs_bs_candidate_features_{}.csv".format(date_str)
summary_df.to_csv(filename)
print('Complete.')

