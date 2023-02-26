import pandas as pd
import numpy as np

from Controllers.ConfigManager import ConfigManager as cm
from Controllers.MySQLManager import MySQLManager as msm
from Controllers.DataManager import DataManager as dtm
from Controllers.LearningManager import LearningManager as lrn

#Instantiate Controllers
config = cm.ConfigManaager().config
sql = msm.MySQLManager(config=config)

coeffs_df = sql.read_table_into_data_frame(schema_name='KIR_HLA_STUDY', 
    table_name='model_result_el_net_coeffs')

pheno_defs_df = sql.read_table_into_data_frame(schema_name='KIR_HLA_STUDY', 
    table_name='immunophenotype_definitions')

counts = {}

for index, row in coeffs_df.iterrows():
    if row[1] not in counts:
        counts[row[1]] = 1
    else:
        counts[row[1]] += 1

counts = np.array([[key, counts[key]] for key in counts])
counts = counts[counts[:,1].argsort()][::-1]
counts_df = pd.DataFrame(counts, columns=['phenotype_id', 'counts'])

counts_df = counts_df.merge(pheno_defs_df, how="inner", on="phenotype_id")

filename = 'Analysis/en_candidate_immunophenotypes{}.csv'.format('310123')
counts_df.to_csv(filename)
print(counts_df)