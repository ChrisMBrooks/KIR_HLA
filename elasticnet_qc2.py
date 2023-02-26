import numpy as np
import pandas as pd
import time

from Controllers.ConfigManager import ConfigManager as cm
from Controllers.MySQLManager import MySQLManager as msm
from Controllers.DataManager import DataManager as dtm
from Controllers.LearningManager import LearningManager as lrn

#Instantiate Controllers
config = cm.ConfigManaager().config
sql = msm.MySQLManager(config=config)
data_mgr = dtm.DataManager(config=config, use_full_dataset=False)
lrn_mgr = lrn.LearningManager(config=config)

# Pull Data from DB

iKIR_scores_vector = data_mgr.feature_values(normalise = False, fill_na = True, fill_na_value = 0.0, 
        partition = 'training')

immunos_maxtrix = data_mgr.outcome_values(normalise = False, fill_na = True, fill_na_value = 0.0, 
        partition = 'training')

# Perform Grid Search for Hyper Params
# Instantiate evaluation method model

from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

# Normalise Data
immuno_mms = MinMaxScaler(feature_range=(-1, 1))
iKIR_mms = MinMaxScaler(feature_range=(-1, 1))

immunos_maxtrix = immuno_mms.fit_transform(immunos_maxtrix)
iKIR_scores_vector = iKIR_mms.fit_transform(iKIR_scores_vector) # MMS scales and translates each feature individually
iKIR_scores_vector = iKIR_scores_vector.ravel()

alpha = 0.1
l1_ratio = 0.45
model = ElasticNet(l1_ratio=l1_ratio, alpha=alpha)

#Fit the Model
model.fit(immunos_maxtrix,iKIR_scores_vector)

#Evaluate Performance
y_hat = model.predict(immunos_maxtrix)
mae = mean_absolute_error(iKIR_scores_vector, y_hat)

#Baselining
y_hat_rand = lrn_mgr.generate_iKIR_scores(immunos_maxtrix.shape[0], normalise=True)
mae_rand = mean_absolute_error(iKIR_scores_vector, y_hat_rand)

y_hat_const = np.ones((immunos_maxtrix.shape[0], 1))*-0.2528113286130779
mae_const = mean_absolute_error(iKIR_scores_vector, y_hat_const)

#Manage Results
feature_weights = model.coef_
candidate_coefs = np.where(np.absolute(feature_weights) > 0)[0]

cand_coef_count = candidate_coefs.shape[0]
result = [-1*mae, -1*mae_rand, -1*mae_const, cand_coef_count, model.intercept_]
    
print(result)