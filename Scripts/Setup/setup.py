import os
import pandas as pd

from Controllers.ConfigManager import ConfigManager as cm
from Controllers.DefinitionManager import DefinitionManager as dm
from Controllers.MySQLManager import MySQLManager as msm

def get_data_src_reader(src_filename:str):
    extension = src_filename.split('.')[-1]
    if extension == "csv":
        return pd.read_csv
    else:
        return pd.read_excel

def setup_data_source(root_path:str, definition_detais:dict):
    dd = definition_detais
    full_path_filename = os.path.join(root_path, dd["datasource"])
    data_frame = get_data_src_reader(full_path_filename)(full_path_filename)

    #Create Table
    table_name = dd["table_name"]
    schema_name = dd["schema_name"]
    table_def = dm.DefinitionManaager(dd["definition_file"]).definition
    sql.create_table(schema_name=schema_name, table_name=table_name, definition=table_def)

    #Push Data
    valid_values = []
    for index, row in data_frame.iterrows():
        if not pd.isnull(row[dd["src_primary_key"]]):
            src_col_names = dd["source_col_names"]
            record = [row[col_name] for col_name in src_col_names]
            valid_values.append(record)
    
    column_names = dd["table_col_names"]

    sql.insert_records(schema_name=schema_name, table_name=table_name, 
        column_names=column_names, values=valid_values
    )

def insert_measurements():
    full_path_filename = os.path.join(root_path, "5_Trait_Values/2. Trait Values.xlsx")
    schema_name = "KIR_HLA_STUDY"
    table_name = "imunophenotype_measurements"
    column_names = ["subject_id","measurement_id","measurement_value"]
    
    data_frame = get_data_src_reader(full_path_filename)(full_path_filename)

    valid_values = []
    for index, row in data_frame.iterrows():
        if not pd.isnull(row["FlowJo Subject ID"]):
            for key, value in row.iloc[1:].items():
                valid_values.append((key, row["FlowJo Subject ID"], value))
        
                if len(valid_values) > 100000:
                    sql.insert_records(schema_name=schema_name, 
                        table_name=table_name, 
                        column_names=column_names, values=valid_values
                    )

                    valid_values = []
    
    if len(valid_values) > 0:
        sql.insert_records(schema_name=schema_name, 
            table_name=table_name, 
            column_names=column_names, values=valid_values
        )

def transpose_and_output_trait_values(use_short_file:bool = False):
    root_path = "/Users/chrismbrooks/Documents/Imperial/Asquith Group/Raw Data/Twins UK/"
    if use_short_file:
        file_filename = "5_Trait_Values/2. Trait Values_short.xlsx"
        output_filename = "Data/trait_values_transpose_short.csv"
    else:
        file_filename = "5_Trait_Values/2. Trait Values.xlsx"
        output_filename = "Data/trait_values_transpose.csv"

    full_path_filename = os.path.join(root_path, file_filename)   
    data_frame = pd.read_excel(full_path_filename, index_col=0).T
    data_frame.to_csv(output_filename, index_label ='subject_id')

    return output_filename
    
    # data_frame.to_parquet("Data/trait_values_transpose.parquet")
    # data_frame = pd.read_parquet("Data/trait_values_transpose.parquet")

#Instantiate Controllers
config = cm.ConfigManaager().config
sql = msm.MySQLManager(config=config)

#Create and Populate Tables
root_path = "/Users/chrismbrooks/Documents/Imperial/Asquith Group/Raw Data/Twins UK/"
for definition_details in config['setup']['datasources']:
    #setup_data_source(root_path,definition_details)
    pass

# Decompose Measurements Table into Individuals Records
sql.create_mesurements_table()
insert_measurements()

# Validation Partition Table
sql.create_partition_table()

# Transpose Measurements Table
filename = transpose_and_output_trait_values(use_short_file=False)
#data_frame = pd.read_csv(filename, index_col=0)
