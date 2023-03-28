import numpy as np
import pandas as pd
import scipy as sp

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import RepeatedKFold, cross_validate
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error

from Controllers.DataScienceManager import DataScienceManager as dsm

#Instantiate Controllers
use_full_dataset=True
data_sci_mgr = dsm.DataScienceManager(use_full_dataset=use_full_dataset)

# Pull Data from DB
#Read in Subset of Immunophenotypes
cut_off = 10
filename = "Data/na_filtered_phenos_less_thn_{}_zeros.csv".format(str(cut_off))
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
    phenos, scores
)

# Instantiate Model 
l1_ratio = 0.1
alpha = 0.2
model = ElasticNet(alpha=alpha, 
    l1_ratio=l1_ratio
)

# Evaluate Model
model.fit(phenos, scores)
y_hat = model.predict(phenos)
betas = list(model.coef_)
intercept = model.intercept_

# Baselining
y_hat_qc = np.ones(scores.shape[0])*intercept

# Analysis
residuals = scores - y_hat
residuals_std = (residuals - residuals.mean()) / residuals.std()

data = [[phenos_subset[i], betas[i]] for i in range(0, len(betas), 1)]
betas_df = pd.DataFrame(data, columns=['labels', 'en_betas'])

#Enrich with univariate p-values
pheno_defs_df = data_sci_mgr.sql.read_table_into_data_frame(schema_name='KIR_HLA_STUDY', 
    table_name='immunophenotype_definitions')

ols_results = data_sci_mgr.sql.read_table_into_data_frame(schema_name='KIR_HLA_STUDY', 
    table_name='model_result_ols')
ols_results = ols_results[['feature_name', 'p_1']]

pheno_weights = pheno_defs_df.merge(betas_df, left_on='phenotype_id', 
    right_on='labels', how='right'
)

pheno_weights = pheno_weights.merge(ols_results, left_on='phenotype_id', 
    right_on='feature_name', how='left'
)

pheno_weights['abs(en_betas)'] = np.abs(pheno_weights['en_betas'])

pheno_weights.sort_values(by='abs(en_betas)', ascending=False)

non_zero_coeff_count  = np.where(pheno_weights['abs(en_betas)'] > 0)[0].size

#Export Results
columns = ['feature_name', 'marker_definition', 'parent_population', 'en_betas', 'p_1']
pheno_weights = pheno_weights[columns]

filename = "Analysis/ElasticNet/en_betas_{}.csv".format(data_sci_mgr.data_mgr.get_date_str())
pheno_weights.to_csv(filename)

#Calc MAE
baseline_neg_mae = -1*mean_absolute_error(scores, y_hat_qc)
neg_mae = -1*mean_absolute_error(scores, y_hat)

print('baseline: {}'.format(baseline_neg_mae))
print('model results:')
print(neg_mae)
print(model.coef_)
print(non_zero_coeff_count)
