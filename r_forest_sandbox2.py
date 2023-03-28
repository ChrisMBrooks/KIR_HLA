import os, random, math, time, uuid
import pandas as pd
import numpy as np

from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

from Controllers.DataScienceManager import DataScienceManager as dsm

def preprocess(phenos:pd.DataFrame, scores:pd.DataFrame):
    scores = scores['f_kir_score'].values.reshape(-1,1)
    if len(phenos.values.shape) == 1:
        phenos = phenos.values.reshape(-1,1)
    else:
        phenos = phenos.values[:, 0:]

    phenos, scores = data_sci_mgr.data_mgr.preprocess_data(
        X=phenos, Y=scores, impute = True, standardise = False, 
        normalise = True
    )

    scores = scores.ravel()
    return phenos, scores

def get_neg_mae(model, phenos:pd.DataFrame, scores:pd.DataFrame):
    model.fit(phenos, scores)

    # Computer Predictions and Summary Stats
    y_hat = model.predict(phenos)
    neg_mae = -1*mean_absolute_error(scores, y_hat)
    return neg_mae

print('Starting...')
start_time = time.time()
run_id = str(uuid.uuid4().hex)

#Instantiate Controllers
use_full_dataset=True
data_sci_mgr = dsm.DataScienceManager(use_full_dataset=use_full_dataset)

#Read in Subset of Immunophenotypes
phenos_subset_1 = ['MFI:469']
phenos_subset_2 = ['MFI:469', 'P2:1968']

scores = data_sci_mgr.data_mgr.features(fill_na=False, fill_na_value=None, partition='training')
phenos = data_sci_mgr.data_mgr.outcomes(fill_na=False, fill_na_value=None, partition='training')
phenos_1 = phenos[phenos_subset_1]
phenos_2 = phenos[phenos_subset_2]

# Standardise Data
phenos_1, scores_1 = preprocess(phenos_1, scores)
phenos_2, scores_2 = preprocess(phenos_2, scores)

max_depth = 30
n_estimators = 110

# Instantiate Model    
model = RandomForestRegressor(
    max_depth=max_depth, 
    n_estimators=n_estimators,
    bootstrap=True,
    max_samples=0.8,
    random_state=False, 
    verbose=1,
    n_jobs=-1
)


neg_mae_deltas = []
for i in range(1000):
    neg_mae = get_neg_mae(model=model, phenos=phenos_1, scores=scores_1)

    values = phenos_2[:, 1]
    np.random.shuffle(values)
    phenos_2[:, 1] = values
    psuedo_neg_mae = get_neg_mae(model=model, phenos=phenos_2, scores=scores_2)
    delta = psuedo_neg_mae - neg_mae
    neg_mae_deltas.append(delta)
    
neg_mae_deltas = np.array(neg_mae_deltas)

neg_mae_deltas.mean()

print('delta mean:', neg_mae_deltas.mean())
print('delta std:', neg_mae_deltas.std())

run_time = time.time() - start_time 
print('run id:', run_id)
print('run time:', run_time)
print('Complete.')