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

mstr_df = sql.read_table_into_data_frame(schema_name='KIR_HLA_STUDY', table_name='functional_kir_genotype')

mstr_df['A'] = mstr_df['f_kir2dl2_w'] & ~mstr_df['f_kir2dl2_s']
mstr_df['B'] = np.where(mstr_df['A'], 1, 0)

mstr_df['f_kir_score'] = mstr_df['f_kir2dl1'].astype('int32')*1.0 + \
    mstr_df['f_kir2dl2_s'].astype('int32')*1.0 + \
    mstr_df['B'].astype('int32')*0.5+ \
    mstr_df['f_kir2dl3'].astype('int32')*0.75 + \
    mstr_df['f_kir3dl1'].astype('int32')*1.0

columns = ['public_id', 'f_kir2dl1',
       'f_kir2dl2_s', 'f_kir2dl2_w', 'f_kir2dl3', 'f_kir3dl1', 'A', 'B',
       'f_kir_score']

mstr_df = mstr_df[columns]
public_d = 'TUK10734802'
print(mstr_df[mstr_df['public_id'] == public_d])