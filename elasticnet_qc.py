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

iKIR_scores_vector = iKIR_scores_vector.ravel()

immunos_maxtrix = data_mgr.outcome_values(normalise = False, fill_na = True, fill_na_value = 0.0, 
        partition = 'training')

# Perform Grid Search for Hyper Params
# Instantiate evaluation method model

from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error

# Instantiate SciKit Learn Components
n_splits = 2 # Number of folds. Must be at least 2.
n_repeats = 2 # Number of folds. Must be at least 2.
cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1)

results = []
for train_indeces, test_indeces in cv.split(X=immunos_maxtrix):

    alpha = 4000
    l1_ratio = 0.85 
    model = ElasticNet(l1_ratio=l1_ratio, alpha=alpha)

    #Fit the Model
    model.fit(immunos_maxtrix[train_indeces],iKIR_scores_vector[train_indeces])

    #Evaluate Performance
    y_hat = model.predict(immunos_maxtrix[test_indeces])

    shape = (immunos_maxtrix[test_indeces].shape[0], immunos_maxtrix[test_indeces].shape[1])
    blanks = np.zeros(shape=shape)
    y_hat = model.predict(blanks)
    
    # print(blanks)
    # print(y_hat)

    mae = mean_absolute_error(iKIR_scores_vector[test_indeces], y_hat)

    #Baselining
    y_hat_rand = lrn_mgr.generate_iKIR_scores(test_indeces.shape[0])
    y_hat_rand = np.ones((test_indeces.shape[0], 1))*2.0829694323144103
    mae_rand = mean_absolute_error(iKIR_scores_vector[test_indeces], y_hat_rand)

    #Manage Results
    feature_weights = model.coef_
    candidate_coefs = np.where(np.absolute(feature_weights) > 0)[0]

    cand_coef_count = candidate_coefs.shape[0]
    result = [-1*mae, -1*mae_rand, cand_coef_count, model.intercept_]
    
    results.append(result)

results = np.array(results)
print(results[:, 0].mean())
print(results[:, 1].mean())
