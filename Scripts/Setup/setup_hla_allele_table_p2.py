# Code to insert allele data, from hla_allele, supplement with motif 
# stats and copy into table, hla_allele2

import os, random, math
import pandas as pd
import numpy as np

from Controllers.ConfigManager import ConfigManager as cm
from Controllers.MySQLManager import MySQLManager as msm

def compute_motif_posession(allele_df:pd.DataFrame, 
    motif_definitions:dict, offset = 24
):
    motif_statuses = []

    for index, item in allele_df['protein_sequence'].items():
        motif_status = {}
        for key in motif_definitions:
            criterea = motif_definitions[key]
            motif_status[key] = True
            for critereon in criterea:
                amino_acid = item[critereon[0]-1+offset]
                condition = amino_acid in critereon[1]
                if not condition:
                    motif_status[key] = False
                    break
                else: 
                    pass
        motif_statuses.append(motif_status)
    
    return motif_statuses

#Instantiate Controllers
config = cm.ConfigManaager().config
sql = msm.MySQLManager(config=config)

datasources = config['setup']['datasources']
table_names = [datasources[0]["table_name"], datasources[1]["table_name"], 
    datasources[2]["table_name"]
]
schema_name = datasources[0]["schema_name"]

# Pull Data from DB
df_hla_allele_tbl = sql.read_table_into_data_frame(schema_name=schema_name, 
    table_name='hla_allele'
)

#Define Constants
motif_definitions = {"c1":[(77,["S"]), (78,["L"]), (79,["R"]), (80,["N"])],
    "c2":[(77,["N"]), (78,["L"]), (79,["R"]), (80,["K"])], 
    "bw4":[(80,["T","I"])], 
    "bw6":[(80,["N"])]
}

offset = 24

# Compute Motif Posession
motif_statuses = compute_motif_posession(
    df_hla_allele_tbl, motif_definitions, offset
)

motif_statuses = [[item['c1'], item['c2'], item['bw4'], item['bw6']] 
    for item in motif_statuses]

# Format and Export Results
df_hla_allele_tbl[['c1', 'c2', 'bw4', 'bw6']] = motif_statuses

records = [[value for key, value in row.items()] for index, row in df_hla_allele_tbl.iterrows()] 

sql.insert_records(schema_name="KIR_HLA_STUDY", 
    table_name="hla_allele2", 
    column_names=["ebi_id", "ebi_name", "hla_gene", "short_code", 
        "protein_sequence", "c1", "c2", "bw4", "bw6"
    ], 
    values=records
)