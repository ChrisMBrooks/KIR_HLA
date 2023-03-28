# Experimenting with Pearson Correlation Matrices
import math
import numpy as np
import pandas as pd 

from Controllers.ConfigManager import ConfigManager as cm
from Controllers.MySQLManager import MySQLManager as msm
from Controllers.DataManager import DataManager as dtm
from Controllers.LearningManager import LearningManager as lrn

#Instantiate Controllers
config = cm.ConfigManaager().config
sql = msm.MySQLManager(config=config)
data_mgr = dtm.DataManager(config=config, use_full_dataset=False)
lrn_mgr = lrn.LearningManager(config=config)

Y = data_mgr.outcomes(fill_na = False, partition = 'training')

Y0 = Y.values[:, 1:-2].astype(float)

pearson_coeffs = np.corrcoef(x=Y0, rowvar=False)
pearson_coeffs_df = pd.DataFrame(pearson_coeffs, columns=Y.columns[1:-2], index=Y.columns[1:-2])

tuples = [] 
for row_key, row in pearson_coeffs_df.iterrows():
   for item_key, item_value in row.iteritems():
      item_value2 = round(math.fabs(item_value), 2)
      if item_value2 >= 0.90 and (item_key != row_key):
         covariates = [row_key, item_key]
         covariates.sort()
         tuples.append((covariates[0], covariates[1], item_value2))

print(tuples)