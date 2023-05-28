import requests, time  

import numpy as np
import pandas as pd
from Controllers.DataScienceManager import DataScienceManager as dsm

#Instantiate Controllers
use_full_dataset = True
use_database = False
data_sci_mgr = dsm.DataScienceManager(
    use_full_dataset=use_full_dataset, 
    use_database=use_database
)

def get_signal_peptide_length(allele_name:str):

    url_template = 'https://www.ebi.ac.uk/cgi-bin/ipd/api/allele/{}'

    url = url_template.format(allele_name)
    request = requests.get(url)

    if request.status_code == 200:
        peptides = request.json()['feature']['protein']
        length = [peptide['length'] for peptide in peptides if peptide['type'] == 'signal'][0]
        return length
    else:
        return None


hla_allele_tbl = data_sci_mgr.sql.read_table_into_data_frame(schema_name='KIR_HLA_STUDY', table_name='hla_allele')

alleles = list(hla_allele_tbl['ebi_id'])

lengths = []
for allele_name in alleles: 
    length = get_signal_peptide_length(allele_name)
    lengths.append(length)
    time.sleep(.1)

hla_allele_tbl['signal_peptide_len'] = np.array(lengths)
columns = list(hla_allele_tbl.columns)

print(hla_allele_tbl)

data_sci_mgr.data_mgr.insert_df_to_sql_table(hla_allele_tbl, columns=columns, schema='KIR_HLA_STUDY', table='hla_allele_2')