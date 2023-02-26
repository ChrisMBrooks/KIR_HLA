import time 
import numpy as np
import pandas as pd
from random import randint
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme()

#Instantiate Controllers
from Controllers.ConfigManager import ConfigManager as cm
from Controllers.MySQLManager import MySQLManager as msm
from Controllers.DataManager import DataManager as dtm
from Controllers.LearningManager import LearningManager as lrn

def construct_distribution(population:np.ndarray):
    counts = {}
    for i in range(population.shape[0]):
        item = population[i, 0]
        if item not in counts:
            counts[item] = 1
        else:
            counts[item] += 1

    counts = {key:value/population.shape[0] for (key,value) in counts.items()}
    keys = tuple(sorted(list(counts.keys())))

    distribution = {}
    distribution[keys[0]] = (0, counts[keys[0]])
    for i in range(1, len(keys)):
        min = distribution[keys[i-1]][1]
        max = min + counts[keys[i]]
        distribution[keys[i]] = (min, max)

    return distribution

def get_random_ikir_score(distribution):
    i = randint(1, 1000000)/1000000
    for key in distribution:
        tup = distribution[key]
        if i > tup[0] and i <= tup[1]:
            return key
    return 0

start_time = time.time()

use_full_dataset = True
config = cm.ConfigManaager().config
sql = msm.MySQLManager(config=config)
data_mgr = dtm.DataManager(config=config, use_full_dataset=use_full_dataset)
lrn_mgr = lrn.LearningManager(config=config)

population_t = data_mgr.feature_values(normalise = False, fill_na = True, fill_na_value = 0.0, 
    partition = 'training'
)

population_v = data_mgr.feature_values(normalise = False, fill_na = True, fill_na_value = 0.0, 
    partition = 'validation'
)    

population = np.concatenate((population_t, population_v))

distribution = construct_distribution(population)

X = data_mgr.features(fill_na = False, partition = 'training')
X0 = X[['public_id', 'f_kir_score']]

Ya = data_mgr.outcomes(fill_na = False, partition = 'training')
Yb = data_mgr.outcomes(fill_na = False, partition = 'validation')
Y = pd.concat([Ya, Yb])

p_vals = []
n = 10000
for feature_name in Y.columns[1:11]:
#for feature_name in Y.columns[1:-2]:
    Y0 = Y[['public_id', feature_name]]
    
    #Filter NAs
    Z0 = Y0[~Y0.isna().any(axis=1)].copy()
    for i in range(n):
        Z0['psuedo_ikir_score'] = np.array([get_random_ikir_score(distribution) for x in range(Z0.shape[0])])

        Z1 = Z0[['psuedo_ikir_score', feature_name]].values
        X1 = Z1[:, 0]
        Y1 = Z1[:, 1]

        p_val = lrn_mgr.regression_p_score2(Y1, X1)[-1]
        p_vals.append(p_val)

p_vals = np.array(p_vals)
quantiles = np.quantile(p_vals, q=[0.05, 0.95])
percentiles = np.percentile(p_vals, q=[5, 95])

print(quantiles)
print(percentiles)

sns.histplot(data=p_vals, binwidth=0.05)

#plt.show()
plt.title('Permutation Significance Test')
date = "22022023"
plt.savefig('Analysis/Univariate/perm_test_hist{}.png'.format(date))

elapsed_time = time.time() - start_time
print(elapsed_time)