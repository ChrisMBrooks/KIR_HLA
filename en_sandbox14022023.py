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

from Controllers.ConfigManager import ConfigManager as cm
from Controllers.MySQLManager import MySQLManager as msm
from Controllers.DataManager import DataManager as dtm
from Controllers.LearningManager import LearningManager as lrn

def preprocess_data(X, Y):
    X, Y = impute_missing_values(X, Y, strategy='mean')
    X, Y = standardise(X, Y)
    X, Y = normalise(X, Y) #Centeres about the origin
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

def normalise(X, Y, min=-1, max=1):
    x_mms = MinMaxScaler(feature_range=(min, max))
    y_mms = MinMaxScaler(feature_range=(min, max))

    X = x_mms.fit_transform(X)

    # MMS scales and translates each feature individually
    Y = y_mms.fit_transform(Y) 

    return X, Y

def retrieve_subset_of_data(partition = 'training', use_full_dataset = True):
    # Get iKIR Scores
    ikir_scores = data_mgr.feature_values(normalise = False, fill_na = False, 
        fill_na_value = 0.0, partition = partition
    )

    # Read in List of Desired Phenos
    filter = 20
    filename = "Data/14022023/na_filtered_phenos_less_thn_{}_zeros_14022023.csv".format(str(filter))
    phenos_subset = list(pd.read_csv(filename, index_col=0).values[:, 0])

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
ikir_scores_t_raw, immunos_labels_t, immunos_maxtrix_subset_t = \
    retrieve_subset_of_data(partition = 'training', use_full_dataset=use_full_dataset)

# Normalise Data 
ikir_scores_t, immunos_maxtrix_subset_t = preprocess_data(ikir_scores_t_raw, immunos_maxtrix_subset_t)

# Instantiate Model 
l1_ratio = 0.1
alpha = 0.3
model = ElasticNet(alpha=alpha, 
    l1_ratio=l1_ratio
)

# Evaluate Model
model.fit(immunos_maxtrix_subset_t, ikir_scores_t)
y_hat = model.predict(immunos_maxtrix_subset_t)
betas = list(model.coef_)
intercept = model.intercept_

# Baselining
y_hat_qc = np.ones(ikir_scores_t.shape[0])*intercept

# Analysis
residuals = ikir_scores_t - y_hat
residuals_std = (residuals - residuals.mean()) / residuals.std()

data = [[immunos_labels_t[i], betas[i]] for i in range(0, len(betas), 1)]
betas_df = pd.DataFrame(data, columns=['labels', 'betas'])

#Enrich with univariate p-values
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

#Export Results
columns = ['feature_name', 'marker_definition', 'parent_population', 'betas', 'p_1']
pheno_weights = pheno_weights[columns]

filename = "Analysis/ElasticNet/en_betas_14022023.csv"
pheno_weights.to_csv(filename)

#Calc MAE
baseline_neg_mae = -1*mean_absolute_error(ikir_scores_t, y_hat_qc)
neg_mae = -1*mean_absolute_error(ikir_scores_t, y_hat)

print('baseline: {}',format(baseline_neg_mae))
print('model results:')
print(neg_mae)
print(model.coef_)
