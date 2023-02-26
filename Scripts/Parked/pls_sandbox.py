import numpy as np
import pandas as pd
import time

from Controllers.ConfigManager import ConfigManager as cm
from Controllers.MySQLManager import MySQLManager as msm
from Controllers.DataManager import DataManager as dtm
from Controllers.LearningManager import LearningManager as lrn

from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold
import matplotlib.pyplot as plt

def optimise_pls(X, Y, num_components):

    n_splits = 2
    n_repeats = 1
    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1)

    r2_scores = []
    mean_sqr_errors = []
    rpds = []
    for i in range(1, 1 + num_components, 1):
        model = PLSRegression(n_components=i)
        Y_cv = cross_val_predict(model, X, Y, cv=cv)

        r_sqaured = r2_score(Y, Y_cv)
        mean_sqr_error = mean_squared_error(Y, Y_cv)
        rpd = Y.std()/np.sqrt(mean_sqr_error)

        r2_scores.append(r_sqaured)
        mean_sqr_errors.append(mean_sqr_error)
        rpds.append(rpd)
    
    return r2_scores, mean_sqr_errors, rpds

def plot_mean_sq_error(errors):
    with plt.style.context('ggplot'):
        X = np.arange(1, 1+ len(errors))
        Y = np.array(errors)
        plt.plot(X, Y, '-v', color='blue', mfc='blue')

        plt.xlabel('Number of PLS components')
        plt.xticks = np.arange(1, 1 + len(errors), 1)
        plt.ylabel('MSE')
        plt.title('PLS')

        plt.show()

#Instantiate Controllers
config = cm.ConfigManaager().config
sql = msm.MySQLManager(config=config)
data_mgr = dtm.DataManager(config=config, use_full_dataset=True)
lrn_mgr = lrn.LearningManager(config=config)

# Pull Data from DB

X = data_mgr.feature_values(normalise = False, fill_na = True, fill_na_value = 0.0, 
        partition = 'training')

Y = data_mgr.outcome_values(normalise = False, fill_na = True, fill_na_value = 0.0, 
        partition = 'training')

r2s, errors, rpds = optimise_pls(Y, X, 10)
plot_mean_sq_error(errors)
