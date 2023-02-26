# Experimenting with dataset filtering by removing immunophenotypes with greater than a defined number of NaNs

import time, uuid
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error

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
filename = "Data/candidate_phenos_09022023.csv"
phenos_subset = list(pd.read_csv(filename, index_col=0).values[:, 0])

#Filter Dataset on Desired Subset 
immunos_maxtrix_subset = data_mgr.outcomes_subset(desired_columns=phenos_subset)
#immunos_maxtrix_subset.fillna(0.0, inplace=True)
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
x_mms = MinMaxScaler(feature_range=(-1, 1))
y_mms = MinMaxScaler(feature_range=(-1, 1))

X = x_mms.fit_transform(X)
Y = y_mms.fit_transform(Y) # MMS scales and translates each feature individually

#Sigmoid Transformation
#Y = 1/(1+np.exp(-1*Y))

# Fit the Model 
run_id = str(uuid.uuid4().hex)

l1_ratio = 0.3
alpha = .07

model = ElasticNet(alpha=alpha, 
    l1_ratio=l1_ratio
)

model.fit(Y, X)

X_hat = model.predict(Y)

mae = mean_absolute_error(X, X_hat)

run_record = [run_id, float(mae), float(l1_ratio), float(alpha)]
        
print(run_record)
print('Complete.')

