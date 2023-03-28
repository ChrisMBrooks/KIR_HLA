import numpy as np
import pandas as pd

from Controllers.ConfigManager import ConfigManager as cm
from Controllers.MySQLManager import MySQLManager as msm
from Controllers.DataManager import DataManager as dtm
from Controllers.LearningManager import LearningManager as lrn

def generate_en_coeffs_from_h_params(X, Y, l1_ratio:float, alpha:float):
    run_id = str(uuid.uuid4().hex)

    model = ElasticNet(alpha=alpha, 
        l1_ratio=l1_ratio
    )

    model.fit(Y, X)

    feature_weights = model.coef_
    non_zero_coeffs  = []

    for index, weight in enumerate(feature_weights):
        if np.abs(weight) > 0:
            non_zero_coeffs.append(
                [run_id, pheno_labels[index], float(weight), l1_ratio, alpha])
            
    table_columns = ['run_id', 'phenotype_label', 'beta', 'l1_ratio', 'alpha']

    sql.insert_records(schema_name='KIR_HLA_STUDY', 
        table_name='model_result_el_net_coeffs',
        column_names=table_columns, values=non_zero_coeffs
    )

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

pheno_labels = data_mgr.outcomes(fill_na = False, partition = 'training').columns[1:-2]

# Perform Grid Search for Hyper Params
# Instantiate evaluation method model

from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet
import uuid

hyper_params = [(0.85, 18), (0.85, 48), (0.85, 90), (0.85, 190), (0.85, 380), (0.85, 580), (0.85, 980), (0.85, 1460), (0.85, 1860)]

for l1_ratio, alpha in hyper_params:
    generate_en_coeffs_from_h_params(X, Y, l1_ratio=l1_ratio, alpha=alpha)

print('Complete.')

