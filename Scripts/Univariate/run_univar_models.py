import numpy as np
import pandas as pd
import scipy as sp
import uuid, time

from Controllers.DataScienceManager import DataScienceManager as dsm

start_time = time.time()
run_id = str(uuid.uuid4().hex)

#Instantiate Controllers
use_full_dataset=True
data_sci_mgr = dsm.DataScienceManager(use_full_dataset=use_full_dataset)

phenos_subset = ['MFI:469', 'P1:22210', 'P1:20054', 'P1:22213']
metric = 'f_ikir_count'
partition = 'training'

# Pull Data from DB
X = data_sci_mgr.data_mgr.features(fill_na = False, partition = partition)
X['strong_or_weak'] = X['f_kir2dl2_s'] | X['f_kir2dl2_w']
X['f_ikir_count'] = X['f_kir2dl1'] + X['strong_or_weak'] + X['f_kir2dl3'] + X['f_kir3dl1']

X0 = X[['public_id', metric]]

Y = data_sci_mgr.data_mgr.outcomes(fill_na = False, partition = partition)

results = []
for feature_name in phenos_subset:
    Y0 = Y[['public_id', feature_name]]
    Z = X0.merge(Y0, on='public_id', how='inner')
    
    #Filter NAs
    Z0 = Z[~Z.isna().any(axis=1)]

    Z1 = Z0[[metric, feature_name]].values
    X1 = Z1[:, 0] #i.e score/ count
    Y1 = Z1[:, 1] #i.e. immunophenotype

    beta_hat = data_sci_mgr.lrn_mgr.UnivariateRegression(X=Y1, Y=X1)[1:]
    p_scores = data_sci_mgr.lrn_mgr.regression_p_score2(X=Y1, Y=X1)

    results.append([run_id, feature_name] + beta_hat + [p_scores[1]])
    
columns = ["run_id" , "feature_name", "beta_0", "beta_1", "p_1"]
results_df = pd.DataFrame(results, columns=columns)

results_df['metric'] = metric
results_df['partition'] = partition

print(results_df)

#data_mgr.insert_df_to_sql_table(df=results_df, columns=columns, schema='KIR_HLA_STUDY', 
#    table='model_result_ols', use_batches=True, batch_size=5000)

print('Complete.')
