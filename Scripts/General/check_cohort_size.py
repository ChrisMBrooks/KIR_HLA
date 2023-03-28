import os, random, math
import pandas as pd
import numpy as np

from Controllers.ConfigManager import ConfigManager as cm
from Controllers.MySQLManager import MySQLManager as msm

#Instantiate Controllers
config = cm.ConfigManaager().config
sql = msm.MySQLManager(config=config)

datasources = config['setup']['datasources']
table_names = [datasources[0]["table_name"], datasources[1]["table_name"], 
    datasources[2]["table_name"]
]
schema_name = datasources[0]["schema_name"]

# Pull Data from DB
df_hla_geno_tbl = sql.read_table_into_data_frame(schema_name=schema_name, 
    table_name=table_names[0]
)

df_kir_geno_tbl = sql.read_table_into_data_frame(schema_name=schema_name,
    table_name=table_names[1]
)

df_mapping_tbl = sql.read_table_into_data_frame(schema_name=schema_name, 
    table_name=table_names[2]
)

# Concatenate into a Single Dataframe
mster_df = df_mapping_tbl.merge(df_hla_geno_tbl, 
    on='public_id', how='inner').merge(df_kir_geno_tbl, 
    on='public_id', how='inner')

outer_df = pd.concat([mster_df['public_id'],df_mapping_tbl['public_id']]).drop_duplicates(keep=False)

required_columns = ['public_id', 'flow_jo_subject_id', 'kir2dl1_t50', 'kir2dl2_t50', 'kir2dl3_t50', 'kir3dl1ex4_t50', 'kir3dl1ex9_t50', 
    'a_1', 'a_2', 'b_1', 'b_2', 'c_1', 'c_2']

mster_df = mster_df[required_columns].copy()

complete_record_status = [1 if row.isna().sum() <= 0 else 0 for index, row in mster_df.iterrows()] 

mster_df['complete_record'] = complete_record_status
mster_df = mster_df[mster_df['complete_record'] > 0].copy() 
mster_df['subject_id'] = mster_df['flow_jo_subject_id'] 

# Train-Validation Data Split
dataset_indeces = set(np.arange(0, mster_df['public_id'].shape[0], 1))

training_size = math.floor(.8*mster_df['public_id'].shape[0])
training_set = set(random.sample(dataset_indeces, k=training_size))

validation_set = dataset_indeces.difference(training_set)

training_df = pd.DataFrame(mster_df.iloc[list(training_set)][['public_id', 'subject_id']])
training_df.insert(loc=1, column='validation_partition', value="TRAINING")

validation_df = pd.DataFrame(mster_df.iloc[list(validation_set)][['public_id','subject_id']])
validation_df.insert(loc=1, column='validation_partition', value="VALIDATION")

train_valid_df = training_df.append(validation_df)

values = [list(x) for x in train_valid_df.values]

print('df_hla_geno_tbl: ')
print(df_hla_geno_tbl.shape)

print('df_kir_geno_tbl: ')
print(df_kir_geno_tbl.shape)

print('df_mapping_tbl: ')
print(df_mapping_tbl.shape)

print('outer_df: ')
print(outer_df.shape)

print('mster_df: ')
print(mster_df.shape)

print('part1 + part2 size:', len(values))