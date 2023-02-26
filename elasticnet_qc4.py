# Using Forward Selection & Backward Selection to stress test the "significant" immunophenotypes

import time, uuid
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error
from sklearn.feature_selection import SequentialFeatureSelector

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
filename = "Data/candidate_phenos_20022023.csv"
phenos_subset = list(pd.read_csv(filename, index_col=0).values[:, 0])

#Filter Dataset on Desired Subset 
immunos_maxtrix_subset = data_mgr.outcomes_subset(desired_columns=phenos_subset)
#immunos_maxtrix_subset.fillna(0.0, inplace=True)
labels = immunos_maxtrix_subset.columns[1:-2]
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

#Sigmoid Transformation
#Y = 1/(1+np.exp(-1*Y))

# Fit the Model 
run_id = str(uuid.uuid4().hex)

n_splits = 2
n_repeats = 3
cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1)

l1_ratio = 0.3
alpha = .02

model = ElasticNet(alpha=alpha, 
    l1_ratio=l1_ratio
)

sfs_for = SequentialFeatureSelector(model, direction='forward', scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
sfs_for.fit(Y, X)
sfs_bac = SequentialFeatureSelector(model, direction='backward', scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
sfs_bac.fit(Y, X)

for_selected_features = sfs_for.get_support()
bac_selected_features = sfs_bac.get_support()

summary = [[labels[i], for_selected_features[i], bac_selected_features[i]] for i in range(0, labels.shape[0], 1)]

summary_df = pd.DataFrame(summary, columns=['label', 'forward_selected', 'backward_selected'])

filename = "Data/candidate_phenos_qc_20022023.csv"
summary_df.to_csv(filename)
print('Complete.')

