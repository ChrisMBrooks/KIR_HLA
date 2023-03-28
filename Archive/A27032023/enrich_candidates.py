import pandas as pd
import numpy as np

from Controllers.ConfigManager import ConfigManager as cm
from Controllers.MySQLManager import MySQLManager as msm

#Instantiate Controllers
config = cm.ConfigManaager().config
sql = msm.MySQLManager(config=config)

pheno_defs_df = sql.read_table_into_data_frame(schema_name='KIR_HLA_STUDY', 
    table_name='immunophenotype_definitions')

filename = "Data/candidate_phenos_09022023w.csv"
candidate_phenos_df = pd.read_csv(filename, index_col=0)

candidate_phenos_df = candidate_phenos_df.merge(pheno_defs_df, left_on='labels', 
    right_on='phenotype_id', how='left'
)

columns = ['phenotype_id', 'marker_definition',
    'parent_population', 'weights', 'abs_weights'
]
candidate_phenos_df = candidate_phenos_df[columns]
candidate_phenos_df.sort_values('abs_weights', inplace=True, ascending=False)

filename="Data/enriched_candidates_09022023.csv"
candidate_phenos_df.to_csv(filename)