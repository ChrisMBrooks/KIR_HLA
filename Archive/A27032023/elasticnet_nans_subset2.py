# Experimenting with dataset filtering by removing immunophenotypes with greater than a defined number of NaNs

import time, uuid
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer

from Controllers.ConfigManager import ConfigManager as cm
from Controllers.MySQLManager import MySQLManager as msm
from Controllers.DataManager import DataManager as dtm
from Controllers.LearningManager import LearningManager as lrn

#Instantiate Controllers
config = cm.ConfigManaager().config
sql = msm.MySQLManager(config=config)
data_mgr = dtm.DataManager(config=config, use_full_dataset=True)
lrn_mgr = lrn.LearningManager(config=config)

#Read in Subset of Immunophenotypes
filter = 10
filename = "Data/na_filtered_phenos_less_thn_{}_zeros.csv".format(str(filter))
phenos_subset = list(pd.read_csv(filename, index_col=0).values[:, 0])

#Filter Dataset on Desired Subset 
immunos_maxtrix_subset = data_mgr.outcomes_subset(desired_columns=phenos_subset)
immunos_labels = immunos_maxtrix_subset.columns[1:-2]
immunos_maxtrix_subset = immunos_maxtrix_subset.values[:,1:-2]

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

Y = imputer.fit_transform(immunos_maxtrix_subset)

X = data_mgr.feature_values(normalise = False, fill_na = True, fill_na_value = 0.0, 
    partition = 'training')

# Standardise Data
x_std_sclr = StandardScaler()
y_std_sclr = StandardScaler()

X = x_std_sclr.fit_transform(X)
Y = y_std_sclr.fit_transform(Y)

# Normalise Data
x_mms = MinMaxScaler(feature_range=(0, 1))
y_mms = MinMaxScaler(feature_range=(0, 1))

X = x_mms.fit_transform(X)
Y = y_mms.fit_transform(Y) # MMS scales and translates each feature individually

# Fit the Model 
l1_ratio = 0.3
alpha = 0.02

model = ElasticNet(alpha=alpha, 
    l1_ratio=l1_ratio
)

model.fit(Y, X)

feature_weights = model.coef_

feature_weights = [[immunos_labels[i], model.coef_[i]] for i in range(0, immunos_labels.shape[0],1)]
feature_weights = pd.DataFrame(feature_weights, columns=['labels', 'weights'])

feature_weights['abs_weights'] = np.absolute(feature_weights['weights'])
non_zero_coeffs = feature_weights[feature_weights['abs_weights'] > 0].copy()

filename = "Data/candidate_phenos_20022023.csv"
non_zero_coeffs['labels'].to_csv(filename)

