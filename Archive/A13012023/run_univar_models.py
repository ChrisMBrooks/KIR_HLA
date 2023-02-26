import numpy as np
import pandas as pd

from Controllers.ConfigManager import ConfigManager as cm
from Controllers.MySQLManager import MySQLManager as msm
from Controllers.DataManager import DataManager as dtm
from Controllers.LearningManager import LearningManager as lrn

#Instantiate Controllers
config = cm.ConfigManaager().config
sql = msm.MySQLManager(config=config)
data_mgr = dtm.DataManager(config=config, use_full_dataset=True)
lrn_mgr = lrn.LearningManager(config=config)

# Pull Data from DB

X = data_mgr.feature_values(normalise = True, fill_na = True, fill_na_value = 0.0, 
        partition = 'training')

Y = data_mgr.outcome_values(normalise = True, fill_na = True, fill_na_value = 0.0, 
        partition = 'training')

columns = ["run_id" , "feature_name", "beta_0", "beta_1", "beta_2", "beta_3", "beta_4", "beta_5"]

results = lrn_mgr.UnivariateRegression(X, Y)

results_df = pd.DataFrame(results, columns=[columns[0]]+columns[2:])
results_df.insert(1, columns[1], data_mgr.data['outcomes'].columns[1:-2])

data_mgr.insert_df_to_sql_table(df=results_df, columns=columns, schema='KIR_HLA_STUDY', 
    table='model_result_ols', use_batches=True, batch_size=5000)

