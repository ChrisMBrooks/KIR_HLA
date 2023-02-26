import os
import pandas as pd

from Controllers.ConfigManager import ConfigManager as cm
from Controllers.DefinitionManager import DefinitionManager as dm
from Controllers.MySQLManager import MySQLManager as msm

def insert_records(data_frame:pd.DataFrame, schema_name:str, 
    table_name:str, column_names:list):

    valid_values = []
    for index, row in data_frame.iterrows():
        valid_values.append([value for key, value in row.items()])
    
    if len(valid_values) > 0:
        sql.insert_records(schema_name=schema_name, 
            table_name=table_name, 
            column_names=column_names, values=valid_values
    )

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

# Construct List of Required Indicators 
indicator_var_tuples = []
indicator_var_strings = []
category_vals = {}
loci = [locus for locus, values in df_hla_geno_tbl.items()]
for locus in loci[1:]:
    category_vals[locus] = []
    for category in df_hla_geno_tbl[locus].unique():
        if not pd.isnull(category):
            indicator_var_tuples.append((locus, category))
            category_vals[locus].append(category)
            indicator_var_strings.append("HLA_{}_CAT_{}".format(locus,int(category)))

# Construct 2D List of Inidicator Valuea
# The below approach currently assumes that NaN is equivalent to false
indicator_matrix = []
for index, public_id in df_hla_geno_tbl['public_id'].items():
    hla_data_row = df_hla_geno_tbl.iloc[[index]]
    indicator_row = [public_id]
    for locus, category in indicator_var_tuples:
        indicator_bool = hla_data_row[locus].values[0] == category
        indicator_row.append(indicator_bool)
    indicator_matrix.append(indicator_row)

# Convert 2D List to Dataframe
column_names = ['public_id']
column_names.extend(indicator_var_strings)
indicator_df = pd.DataFrame (indicator_matrix, columns=column_names)

# Insert Records to Table
insert_records(data_frame=indicator_df, schema_name=schema_name, 
    table_name='train_hla_indicator', column_names=column_names
)