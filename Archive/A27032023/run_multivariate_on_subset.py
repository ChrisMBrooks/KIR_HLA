import numpy as np
import pandas as pd
import scipy as sp

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import RepeatedKFold, cross_validate
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error

from Controllers.ConfigManager import ConfigManager as cm
from Controllers.MySQLManager import MySQLManager as msm
from Controllers.DataManager import DataManager as dtm
from Controllers.LearningManager import LearningManager as lrn

def preprocess_data(X, Y):
    X, Y = impute_missing_values(X, Y, strategy='mean')
    X, Y = standardise(X, Y)
    X, Y = normalise(X, Y, min=0, max=1) #Centeres about the origin
    return X, Y

def impute_missing_values(X, Y, strategy):
    imputer = SimpleImputer(missing_values=np.nan, strategy=strategy)
    X = imputer.fit_transform(X)
    Y = imputer.fit_transform(Y)
    return X, Y

def standardise(X,Y):
    x_std_sclr = StandardScaler()
    y_std_sclr = StandardScaler()

    X = x_std_sclr.fit_transform(X)
    Y = y_std_sclr.fit_transform(Y)
    return X, Y

def normalise(X, Y, min, max):
    x_mms = MinMaxScaler(feature_range=(min, max))
    y_mms = MinMaxScaler(feature_range=(min, max))

    X = x_mms.fit_transform(X)

    # MMS scales and translates each feature individually
    Y = y_mms.fit_transform(Y) 

    return X, Y

def retrieve_subset_of_data(filename:str, partition = 'training', use_full_dataset = True):
    # Get iKIR Scores
    ikir_scores = data_mgr.feature_values(normalise = False, fill_na = False, 
        fill_na_value = 0.0, partition = partition
    )

    # Read in List of Desired Phenos
    phenos_subset = pd.read_csv(filename, index_col=0)
    phenos_subset['filter_criteria'] = phenos_subset['forward_selected'] & phenos_subset['backward_selected']

    phenos_subset = phenos_subset[phenos_subset['filter_criteria'] == True] 

    phenos_subset = list(phenos_subset['label'].values)

    # Filter Master Dataset on Desired Subset
    if use_full_dataset: 
        immunos_maxtrix_subset = data_mgr.outcomes_subset(desired_columns=phenos_subset, 
            partition = partition
        )
    else:
        # Logic to Handle When the Small Traits file is used. 
        pheno_labels_small = data_mgr.outcomes(fill_na=False, partition='everything').columns[1:-2]
        phenos_subset_overlap = np.intersect1d(pheno_labels_small, phenos_subset)
        immunos_maxtrix_subset = data_mgr.outcomes_subset(desired_columns=phenos_subset_overlap, 
            partition = partition
        )

    immunos_labels = immunos_maxtrix_subset.columns[1:-2]
    immunos_maxtrix_subset = immunos_maxtrix_subset.values[:,1:-2].astype(float)
    return ikir_scores, immunos_labels, immunos_maxtrix_subset

#Instantiate Controllers
use_full_dataset=True
config = cm.ConfigManaager().config
sql = msm.MySQLManager(config=config)
data_mgr = dtm.DataManager(config=config, use_full_dataset=use_full_dataset)
lrn_mgr = lrn.LearningManager(config=config)

# Pull Data from DB
filename = "Data/candidate_phenos_qc_20022023.csv"
ikir_scores_t_raw, immunos_labels_t, immunos_maxtrix_subset_t = \
    retrieve_subset_of_data(filename, partition = 'training', use_full_dataset=use_full_dataset)

# Normalise Data 
ikir_scores_t, immunos_maxtrix_subset_t = preprocess_data(ikir_scores_t_raw, immunos_maxtrix_subset_t)

# Instantiate Model 
model = linear_model.LinearRegression()

# Evaluate Model
model.fit(immunos_maxtrix_subset_t, ikir_scores_t)
y_hat = model.predict(immunos_maxtrix_subset_t)
betas = model.coef_[0]
intercept = model.intercept_

beta_hat = np.array([intercept[0]] + list(betas))

p_vals = lrn_mgr.regression_p_score2(immunos_maxtrix_subset_t, ikir_scores_t)[1:]

# Baselining
# intercept = -0.53569521
# y_hat_qc = np.ones(ikir_scores_t.shape[0])*intercept

# Analysis
residuals = ikir_scores_t - y_hat
residuals_std = (residuals - residuals.mean()) / residuals.std()

data = [[immunos_labels_t[i], betas[i]] for i in range(0, len(betas), 1)]
betas_df = pd.DataFrame(data, columns=['labels', 'betas'])

pheno_defs_df = sql.read_table_into_data_frame(schema_name='KIR_HLA_STUDY', 
    table_name='immunophenotype_definitions')

ols_results = sql.read_table_into_data_frame(schema_name='KIR_HLA_STUDY', 
    table_name='model_result_ols')
ols_results = ols_results[['feature_name', 'p_1']]

pheno_weights = pheno_defs_df.merge(betas_df, left_on='phenotype_id', 
    right_on='labels', how='right'
)

pheno_weights = pheno_weights.merge(ols_results, left_on='phenotype_id', 
    right_on='feature_name', how='left'
)

pheno_weights['multi_var_p_val'] = p_vals

columns = ['feature_name', 'marker_definition', 'parent_population', 'betas', 'multi_var_p_val', 'p_1']
pheno_weights = pheno_weights[columns]

date = "20022023"
filename = "Analysis/Multivariate/multivariate_betas_{}.csv".format(date)
pheno_weights.to_csv(filename)
    