import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme()

from Controllers.ConfigManager import ConfigManager as cm
from Controllers.MySQLManager import MySQLManager as msm
from Controllers.DataManager import DataManager as dtm
from Controllers.LearningManager import LearningManager as lrn

#Instantiate Controllers
use_full_dataset = True
config = cm.ConfigManaager().config
sql = msm.MySQLManager(config=config)
data_mgr = dtm.DataManager(config=config, use_full_dataset=use_full_dataset)
lrn_mgr = lrn.LearningManager(config=config)

kir_geno_df = data_mgr.data['func_kir_genotype']
training_df = data_mgr.data['training_partition']
validation_df = data_mgr.data['validation_partition']

primary_df_t = training_df.merge(kir_geno_df, on='public_id', how='inner')
primary_df_v = validation_df.merge(kir_geno_df, on='public_id', how='inner')

primary_df = pd.concat([primary_df_t, primary_df_v])

critical_columns = ['public_id', 'kir2dl1', 'kir2dl2', 'kir2dl3', 'kir3dl1']
primary_df = primary_df[critical_columns]

primary_df['ikir_count'] = primary_df['kir2dl1'] + primary_df['kir2dl2'] + primary_df['kir2dl3'] + primary_df['kir3dl1']

sns.histplot(primary_df, x="ikir_count", discrete=True)
plt.title(r'iKIR Count Histogram: $\mu={:.2f}, $\sigma={:.2f}'.format(primary_df['ikir_count'].mean(), primary_df['ikir_count'].std()))
#plt.show()
plt.savefig('Analysis/Data Characterisation/iKIR_count_hist20022023.png')

