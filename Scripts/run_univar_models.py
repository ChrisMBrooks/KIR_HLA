import numpy as np
import pandas as pd
import scipy as sp
import uuid

from Controllers.ConfigManager import ConfigManager as cm
from Controllers.MySQLManager import MySQLManager as msm
from Controllers.DataManager import DataManager as dtm
from Controllers.LearningManager import LearningManager as lrn

#Instantiate Controllers
config = cm.ConfigManaager().config
sql = msm.MySQLManager(config=config)
data_mgr = dtm.DataManager(config=config, use_full_dataset=True)
lrn_mgr = lrn.LearningManager(config=config)

run_id = str(uuid.uuid4().hex)

# Pull Data from DB
X = data_mgr.features(fill_na = False, partition = 'training')
X0 = X[['public_id', 'f_kir_score']]

Y = data_mgr.outcomes(fill_na = False, partition = 'training')

results = []
for feature_name in Y.columns[1:-2]:
    Y0 = Y[['public_id', feature_name]]
    Z = X0.merge(Y0, on='public_id', how='inner')
    
    #Filter NAs
    Z0 = Z[~Z.isna().any(axis=1)]

    Z1 = Z0[['f_kir_score', feature_name]].values
    X1 = Z1[:, 0]
    Y1 = Z1[:, 1]

    beta_hat = lrn_mgr.UnivariateRegression(X1, Y1)[1:]
    p_scores = lrn_mgr.regression_p_score(X1, Y1, beta_hat=beta_hat)

    #Out of the Box Check of P-value calculation
    #sp_stats_r = sp.stats.linregress(x=X1, y=Y1, alternative='two-sided')
    #print(sp_stats_r)

    results.append([run_id, feature_name] + beta_hat + [p_scores[1]])
    
columns = ["run_id" , "feature_name", "beta_0", "beta_1", "p_1"]
results_df = pd.DataFrame(results, columns=columns)

data_mgr.insert_df_to_sql_table(df=results_df, columns=columns, schema='KIR_HLA_STUDY', 
    table='model_result_ols', use_batches=True, batch_size=5000)

print('Complete.')
