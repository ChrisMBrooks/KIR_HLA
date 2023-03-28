import numpy as np
import pandas as pd
import time, uuid

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme()

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

from Controllers.DataScienceManager import DataScienceManager as dsm

start_time = time.time()
run_id = str(uuid.uuid4().hex)

#Instantiate Controllers
use_full_dataset=True
data_sci_mgr = dsm.DataScienceManager(use_full_dataset=use_full_dataset)

# Pull Data from DB

#Read in Subset of Immunophenotypes
phenos_subset = ['MFI:469', 'P1:22210']

scores = data_sci_mgr.data_mgr.features(fill_na=False, fill_na_value=None, partition='validation')
phenos = data_sci_mgr.data_mgr.outcomes(fill_na=False, fill_na_value=None, partition='validation')
phenos = phenos[phenos_subset]

# Standardise Data
scores = scores['f_kir_score'].values.reshape(-1,1)
if len(phenos.values.shape) == 1:
    phenos = phenos.values.reshape(-1,1)
else:
    phenos = phenos.values[:, 0:]

phenos, scores = data_sci_mgr.data_mgr.preprocess_data(
    phenos, scores
)

# Fit the Multivar Linear Regression Model
model = LinearRegression()
model.fit(phenos, scores)

# Computer Predictions and Summary Stats
y_hat = model.predict(phenos)
neg_mae = -1*mean_absolute_error(scores, y_hat)

p_vals = data_sci_mgr.lrn_mgr.regression_p_score2(phenos, scores)[1:]

feature_weights = model.coef_[0]

pheno_weights = [[phenos_subset[i], feature_weights[i]] for i in range(0, len(feature_weights), 1)]
pheno_weights_df = pd.DataFrame(pheno_weights, columns=['labels', 'lr_betas'])
pheno_weights_df['multi_var_p_vals'] = p_vals
pheno_weights_df['abs(lr_betas)'] = np.abs(pheno_weights_df['lr_betas'])
pheno_weights_df.sort_values(by='abs(lr_betas)', ascending=False)

print(pheno_weights_df)

run_time = time.time() - start_time

print('Complete.')