import os, random, math, time, uuid
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme()

from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

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

# Declare Config Params
n_jobs = 16 - 1
n_splits = 5
n_repeats = 4
random_state_1 = 42
random_state_2 = 21

impute = True
strategy = 'mean'
normalise = True
standardise = True

hyper_params_grid = dict()
hyper_params_grid['solver'] = ['saga']
hyper_params_grid['penalty'] = ['elasticnet'] 
hyper_params_grid['l1_ratio']= [0.61] 
hyper_params_grid['C'] = [0.25]
hyper_params_grid['fit_intercept'] = [True]
hyper_params_grid['max_iter'] = [100] 
hyper_params_grid['n_jobs'] = [n_jobs]

c_step = 0.05
c_min = 0.05
c_max = 0.9 + c_min 
hyper_params_grid['C'] = np.arange(
    c_min, c_max, c_step
)

l1_ratio_step = 0.05
l1_ratio_min = 0.05
l1_ratio_max = 0.9 + l1_ratio_step
hyper_params_grid['l1_ratio'] = np.arange(
    l1_ratio_min, l1_ratio_max, l1_ratio_step
)

ikir_labels = ['f_kir2dl1', 'f_kir2dl2_s', 'f_kir2dl2_w', 'f_kir2dl3', 'f_kir3dl1']
index = 4
ikir_label = ikir_labels[index]

source_filename = 'Data/Subsets/clustered_0.95_and_restricted_to_phenos_with_less_thn_10_zeros_05042023.csv'
date_str = data_sci_mgr.data_mgr.get_date_str()
gs_scores_filename = 'Analysis/LogisticRegression/lr_gs_scores_{}_{}.csv'.format(ikir_label, date_str)
plot_filename = 'Analysis/LogisticRegression/lr_gs_heat_map_{}_{}.png'.format(ikir_label, date_str)
feature_weights_filename = 'Analysis/LogisticRegression/lr_gs_feature_weights_{}_{}.csv'.format(ikir_label, date_str)
output_filename = 'Analysis/LogisticRegression/lr_gs_results_{}_{}.csv'.format(ikir_label, date_str)

#Read in Subset of Immunophenotypes
phenos_subset = list(pd.read_csv(source_filename, index_col=0).values[:, 0])

scores_df = data_sci_mgr.data_mgr.features(fill_na=False, fill_na_value=None, partition='training')
phenos_df = data_sci_mgr.data_mgr.outcomes(fill_na=False, fill_na_value=None, partition='training')

# Standardise Data
phenos = phenos_df[phenos_subset]
scores = scores_df[ikir_label].values.reshape(-1,1)
if len(phenos.values.shape) == 1:
    phenos = phenos.values.reshape(-1,1)
else:
    phenos = phenos.values[:, 0:]

phenos, placeholder = data_sci_mgr.data_mgr.preprocess_data(
    X=phenos, Y=scores, impute = impute, standardise = standardise, 
    normalise = normalise, strategy=strategy
)

scores = scores.ravel()

cv = RepeatedKFold(
    n_splits=n_splits, 
    n_repeats=n_repeats, 
    random_state=random_state_1
)

gsc = GridSearchCV(
    estimator=LogisticRegression(),
    param_grid=hyper_params_grid,
    cv=cv, 
    scoring='neg_mean_absolute_error', 
    verbose=0, 
    n_jobs=n_jobs # parallelism, -1 means using all processors
)

# Perform Grid Search
grid_search_results = gsc.fit(phenos, scores) # Regress Immunos on iKIR Score
h_params = grid_search_results.best_params_
optimal_h_parms = h_params
best_gs_mae = grid_search_results.best_score_
gs_h_params_sets = grid_search_results.cv_results_['params']
gs_scores = grid_search_results.cv_results_['mean_test_score']

grid_results = [[gs_h_params_sets[i][key] for key in gs_h_params_sets[i]]  + [gs_scores[i]]\
            for i in range(0, len(gs_h_params_sets),1)]
columns = [key for key in gs_h_params_sets[0]] + ['avg_neg_mae']

# Export Results
grid_results_df = pd.DataFrame(grid_results, columns=columns)
grid_results_df.to_csv(gs_scores_filename)

sns.relplot(
    data=grid_results_df, x="C", y="l1_ratio", hue="avg_neg_mae", palette='rocket_r',
)
plt.savefig(plot_filename)

# Score the winning H_Parms w/ CV
# Instantiate Model    
model = LogisticRegression(
    solver = h_params['solver'],
    penalty = h_params['penalty'], 
    l1_ratio = h_params['l1_ratio'], 
    C = h_params['C'],
    fit_intercept = h_params['fit_intercept'],
    max_iter = h_params['max_iter'], 
    n_jobs = h_params['n_jobs']
)

splits_gen = cv.split(phenos)

predictions = []
actuals_indeces = []
feature_weights_cv = {}
coef_counts_cv = []
scores_cv = []

for i in range(0, n_repeats+1):
    split = next(splits_gen)
    train_indeces = split[0]
    test_indeces = split[1]

    model.fit(phenos[train_indeces, :], scores[train_indeces])
    feature_weights = model.coef_[0]
    feature_weights_cv['coef_weight_{}'.format(i)] = feature_weights
    non_zero_coeff_count  = np.where(np.absolute(feature_weights) > 0)[0].size
    coef_counts_cv.append(non_zero_coeff_count)

    # Computer Predictions and Summary Stats
    y = scores[test_indeces]
    y_hat = model.predict(phenos[test_indeces, :])
    predictions.append(y_hat)
    actuals_indeces.append(test_indeces)

    neg_mae = -1*mean_absolute_error(y_true=y, y_pred=y_hat)
    scores_cv.append(neg_mae)
    
# Format Phenotype Coefficients 
columns = ['coef_weight_{}'.format(i) for i in range(0, n_repeats+1)]
pheno_weights_df = pd.DataFrame(feature_weights_cv)
pheno_weights_df['coef_weight_mean'] = pheno_weights_df.values.mean(axis=1)
pheno_weights_df['label'] = phenos_subset
pheno_weights_df['abs(cwm)'] = np.abs(pheno_weights_df['coef_weight_mean'])
pheno_weights_df = pheno_weights_df.sort_values(by='abs(cwm)', ascending=False)
pheno_weights_df = pheno_weights_df[pheno_weights_df['abs(cwm)'] > 0].copy()
labels = list(pheno_weights_df['label'])
pheno_weights_df.to_csv(feature_weights_filename)

run_time = time.time() - start_time 

#Export Run Summary 
output = {}
output['ikir'] = ikir_label
output['data_source'] = source_filename
output['best_gs_mae'] = best_gs_mae
output['avg_neg_mae'] = np.array(scores_cv).mean()
output['solver'] = hyper_params_grid['solver']
output['penalty'] = hyper_params_grid['penalty']
output['l1_ratio'] = hyper_params_grid['l1_ratio']
output['C'] = hyper_params_grid['C']
output['fit_intercept'] = hyper_params_grid['fit_intercept']
output['max_iter'] = hyper_params_grid['max_iter']
output['n_jobs'] = hyper_params_grid['n_jobs']
output['optimal_h_parms'] = optimal_h_parms
output['coef_counts'] = coef_counts_cv
output['run_time'] = run_time
output['run_id'] = run_id

output = pd.Series(output)
output.to_csv(output_filename)

print(output)
print('Complete.')