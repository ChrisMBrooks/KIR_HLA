# Compute Results w/ Standardisation. 
# More robust to outliers. 

import numpy as np
import pandas as pd
import time, uuid

from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error

from Controllers.DataScienceManager import DataScienceManager as dsm

start_time = time.time()
run_id = str(uuid.uuid4().hex)

#Instantiate Controllers
#Instantiate Controllers
use_full_dataset=True
use_database = False
data_sci_mgr = dsm.DataScienceManager(
    use_full_dataset=use_full_dataset, 
    use_database=use_database
)

# Declare Config Parameters
# Instantiate Model & Hyper Params
dependent_var = 'kir_count'

impute = True
standardise = True
normalise = True
strategy = 'mean'

n_jobs = 16 - 1
n_splits = 5
n_repeats = 4
random_state = 42

alpha =  0.009
l1_ratio = 0.8

source_filename = "Data/Subsets/clustered_0.95_and_restricted_to_phenos_with_less_thn_10_zeros_05042023.csv"
date_str = data_sci_mgr.data_mgr.get_date_str()
feature_coefs_filename = 'Analysis/ElasticNet/en_feature_coefs_{}.csv'.format(date_str)
output_filename = 'Analysis/ElasticNet/en_results_{}.csv'.format(date_str)

# Pull Data from DB
#Read in Subset of Immunophenotypes
phenos_subset = list(pd.read_csv(source_filename, index_col=0).values[:, 0])
scores = data_sci_mgr.data_mgr.features(fill_na=False, fill_na_value=None, partition='training')
phenos = data_sci_mgr.data_mgr.outcomes(fill_na=False, fill_na_value=None, partition='training')
phenos = phenos[phenos_subset]

# Standardise Data
scores = scores[dependent_var].values.reshape(-1,1)
if len(phenos.values.shape) == 1:
    phenos = phenos.values.reshape(-1,1)
else:
    phenos = phenos.values[:, 0:]

phenos, scores = data_sci_mgr.data_mgr.preprocess_data(
    X=phenos, Y=scores, 
    impute = impute, 
    strategy=strategy,
    standardise = standardise, 
    normalise = normalise
)

scores = scores.ravel()

# Fit the Optimal Hyper Params to the Full Dataset
model = ElasticNet(alpha=alpha, 
    l1_ratio=l1_ratio
)

cv = RepeatedKFold(
    n_splits=n_splits, 
    n_repeats=n_repeats, 
    random_state=random_state
)

splits_gen = cv.split(phenos)

results = []
feature_coefs = pd.DataFrame()
for i in range(0, n_repeats+1):
    split = next(splits_gen)
    train_indeces = split[0]
    test_indeces = split[1]

    model.fit(phenos[train_indeces, :], scores[train_indeces])

    name = 'coef_{}'.format(i)
    if feature_coefs.empty:
        feature_coefs = pd.DataFrame(model.coef_, index=phenos_subset, columns=[name])
    else:
        feature_coefs[name] = model.coef_

    # Computer Predictions and Summary Stats
    y_hat = model.predict(phenos[test_indeces, :])
    neg_mae = -1*mean_absolute_error(scores[test_indeces], y_hat)
    results.append(neg_mae)

feature_coefs['mean'] = feature_coefs.values.mean(axis=1)
feature_coefs = feature_coefs[feature_coefs['mean'] > 0].copy()
feature_coefs['abs(mean)'] = np.abs(feature_coefs['mean'])
candidates = list(feature_coefs.index)
feature_coefs = feature_coefs.sort_values(by='abs(mean)', ascending=False)
feature_coefs.to_csv(feature_coefs_filename)

# Compute Predictions and Summary Stats
results = np.array(results)
avg_neg_mae = results.mean()
non_zero_coeff_count  = np.where(np.absolute(feature_coefs['abs(mean)']) > 0)[0].size
run_time = time.time() - start_time

output = {}
output['data_source'] = source_filename
output['dependent_var'] = dependent_var
output['impute'] = impute
output['strategy'] = strategy
output['standardise'] = standardise
output['normalise'] = normalise
output['n_splits'] = n_splits
output['n_repeats'] = n_repeats
output['random_state'] = random_state
output['avg_neg_mae'] = avg_neg_mae
output['avg_non_zero_coeff_count'] = non_zero_coeff_count
output['l1_ratio'] = l1_ratio
output['alpha'] = alpha
output['candidates'] = candidates
output['run_time'] = run_time
output['run_id'] = run_id
output = pd.Series(output)
output.to_csv(output_filename)

print(output)
print('Complete.')