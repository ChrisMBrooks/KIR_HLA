import numpy as np
import pandas as pd
import time

from sklearn.model_selection import RepeatedKFold
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

# Retrieve and Format Independent Variable
iKIR_scores_vector = data_mgr.feature_values(normalise = False, fill_na = True, fill_na_value = 0.0, 
        partition = 'training')

iKIR_scores_vector = iKIR_scores_vector.ravel()

# Retrieve and format Outcomes
immunos_maxtrix_mfi_subset = data_mgr.outcomes_by_class('MFI')
immunos_maxtrix_mfi_subset.fillna(0.0, inplace=True)
immunos_maxtrix_mfi_subset = immunos_maxtrix_mfi_subset.values[:,1:-2]

# Instantiate the model
alpha = 500
l1_ratio = 0.85 
model = ElasticNet(l1_ratio=l1_ratio, alpha=alpha)

#Fit the Model
model.fit(immunos_maxtrix_mfi_subset,iKIR_scores_vector)

#Evaluate Performance
y_hat = model.predict(immunos_maxtrix_mfi_subset)
mae = mean_absolute_error(iKIR_scores_vector, y_hat)

#Baselining
y_hat_rand = lrn_mgr.generate_iKIR_scores(immunos_maxtrix_mfi_subset.shape[0])
mae_rand = mean_absolute_error(iKIR_scores_vector, y_hat_rand)

y_hat_const = np.ones((immunos_maxtrix_mfi_subset.shape[0], 1))*2.057580174927114
mae_const = mean_absolute_error(iKIR_scores_vector, y_hat_const)

#Manage Results
feature_weights = model.coef_
candidate_coefs = np.where(np.absolute(feature_weights) > 0)[0]

cand_coef_count = candidate_coefs.shape[0]
result = [-1*mae, -1*mae_rand, -1*mae_const, cand_coef_count, model.intercept_]
    
print(result)