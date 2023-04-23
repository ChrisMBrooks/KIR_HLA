# Using Forward Selection & Backward Selection to stress test the "significant" immunophenotypes

import time, uuid, random, math
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme()

from sklearn.linear_model import LinearRegression
from sklearn.inspection import permutation_importance
from Controllers.DataScienceManager import DataScienceManager as dsm

def preprocess_for_validation(
        phenos_t:pd.DataFrame, scores_t:pd.DataFrame, 
        phenos_v:pd.DataFrame, scores_v:pd.DataFrame,
        impute, strategy, standardise, normalise 
    ):
    phenos_t, scores_t = data_sci_mgr.data_mgr.reshape(phenos_t, scores_t)
    phenos_v, scores_v = data_sci_mgr.data_mgr.reshape(phenos_v, scores_v)

    phenos_t, scores_t, phenos_v, scores_v = data_sci_mgr.data_mgr.preprocess_data_v(
        X_t=phenos_t, Y_t=scores_t, X_v=phenos_v, Y_v=scores_v,
        impute = impute, strategy=strategy, standardise = standardise, 
        normalise = normalise
    )

    scores_t = scores_t.ravel()
    scores_v = scores_v.ravel()
    return phenos_t, scores_t, phenos_v, scores_v

# Initiate Script
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

#Set Configuration Params
partition_training_dataset = True
num_repeats = 10
n_jobs = 16 - 1
random_state = 42
n_splits = 4
impute = True
strategy='median'
standardise = True
normalise = True 
source_filename = 'Analysis/Multivariate/11042023_c_rc3/multivar_qc_fs_bs_candidate_features_11042023.csv'
date_str = data_sci_mgr.data_mgr.get_date_str()
results_filename = 'Analysis/Multivariate/feature_importance_perm_values_{}.csv'.format(date_str)
plot_filename = "Analysis/Multivariate/feature_import_box_plot_{}.png".format(date_str)

#Retrieve Data
phenos_subset = pd.read_csv(source_filename, index_col=0)
indeces = phenos_subset.values[:,1:3].sum(axis=1)
indeces = np.where(indeces >= 1)
phenos_subset = list(phenos_subset.iloc[indeces]['label'].values)

scores_t = data_sci_mgr.data_mgr.features(fill_na=False, fill_na_value=None, partition='training')
phenos_t = data_sci_mgr.data_mgr.outcomes(fill_na=False, fill_na_value=None, partition='training')
phenos_t = phenos_t[phenos_subset]

scores_v = data_sci_mgr.data_mgr.features(fill_na=False, fill_na_value=None, partition='validation')
phenos_v = data_sci_mgr.data_mgr.outcomes(fill_na=False, fill_na_value=None, partition='validation')
phenos_v = phenos_v[phenos_subset]

# Partition Data
if partition_training_dataset:
    phenos_t, scores_t, phenos_v, scores_v = data_sci_mgr.data_mgr.partition_training_data(
        phenos_t, scores_t, n_splits, random_state
    )

# Massage Data
phenos_t, scores_t, phenos_v, scores_v = preprocess_for_validation(
    phenos_t, scores_t, phenos_v, scores_v, impute, strategy, 
    standardise, normalise
)

# Fit the Model
model = LinearRegression()
fitted_model = model.fit(phenos_t, scores_t)

# Run Permutation Tests
results = permutation_importance(fitted_model, phenos_v, scores_v, n_repeats=num_repeats,
    random_state=random_state, scoring ='neg_mean_absolute_error', n_jobs=n_jobs
)

# Format Results
results_values = []
for i in range(0, results.importances.shape[1]):
    importance_values = results.importances[:, i]
    results_values.append(importance_values)

importances_df = pd.DataFrame(results_values, columns=phenos_subset)

sorted_indeces = np.argsort(importances_df.values.mean(axis=0))
columns = [phenos_subset[x] for x in sorted_indeces]
importances_df = importances_df[columns].copy()

# Export Results
importances_df.to_csv(results_filename)

# Plot Results
if partition_training_dataset:
    perm_type = 'train-test'
else:
    perm_type = 'train-validate'

ax = importances_df.plot.box(vert=False, whis=1.5)
ax.set_title("Permutation Importances ({})".format(perm_type))
ax.axvline(x=0, color="k", linestyle="--")
ax.set_xlabel("Decrease in accuracy score")
ax.figure.tight_layout()

plt.savefig(plot_filename)
print('Complete.')
