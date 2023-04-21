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
use_full_dataset=True
data_sci_mgr = dsm.DataScienceManager(use_full_dataset=use_full_dataset)

# Pull Data from DB

#Read in Subset of Immunophenotypes
cut_off = 10
filename = "Data/Subsets/na_filtered_phenos_less_thn_{}_zeros.csv".format(str(cut_off))
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
    normalise = True
)

# Instantiate Model & Hyper Params
alpha =  0.009
l1_ratio = 0.8

# Fit the Optimal Hyper Params to the Full Dataset
model = ElasticNet(alpha=alpha, 
    l1_ratio=l1_ratio
)
model.fit(phenos, scores)

# Computer Predictions and Summary Stats
y_hat = model.predict(phenos)
neg_mae = -1*mean_absolute_error(scores, y_hat)

feature_weights = model.coef_
non_zero_coeff_count  = np.where(np.absolute(feature_weights) > 0)[0].size

pheno_weights = [[phenos_subset[i], feature_weights[i]] for i in range(0, len(feature_weights), 1)]
pheno_weights_df = pd.DataFrame(pheno_weights, columns=['labels', 'en_betas'])
pheno_weights_df['abs(en_betas)'] = np.abs(pheno_weights_df['en_betas'])
pheno_weights_df = pheno_weights_df.sort_values(by='abs(en_betas)', ascending=False)
pheno_weights_df = pheno_weights_df[pheno_weights_df['abs(en_betas)'] > 0].copy()
labels = list(pheno_weights_df['labels'])

filename = 'Analysis/ElasticNet/en_feature_importances_{}.csv'.format(data_sci_mgr.data_mgr.get_date_str())
pheno_weights_df.to_csv(filename)

run_time = time.time() - start_time

output = {}
output['neg mae'] = neg_mae
output['l1_ratio'] = l1_ratio
output['alpha'] = alpha
output = pd.Series(output)
filename = 'Analysis/ElasticNet/en_results_{}.csv'.format(data_sci_mgr.data_mgr.get_date_str())
output.to_csv(filename)

print(output)
print('Complete.')