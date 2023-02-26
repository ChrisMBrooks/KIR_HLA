import re, requests, time
import numpy as np
import pandas as pd

from Controllers.ConfigManager import ConfigManager as cm
from Controllers.MySQLManager import MySQLManager as msm

def format_hla_allele_name(allele_tuple:list):
    short_code_str = str(int(allele_tuple[1]))

    if len(short_code_str) <= 3:
        allele_tuple[1] = "0{}".format(short_code_str)
    else:
        allele_tuple[1] = short_code_str

    allele_name = "{}*{}:{}".format(
        allele_tuple[0], allele_tuple[1][:2], allele_tuple[1][2:]
    )

    return allele_name

def get_hla_allele_details(allele_name:str):
    url_template = "https://www.ebi.ac.uk/cgi-bin/ipd/api/allele?limit=10&project=HLA&query=startsWith(name,\"{}\")"
    url = url_template.format(allele_name)

    request = requests.get(url)
    
    output = {}
    
    if request.status_code == 200 and len(request.json()['data']) > 0 and \
        request.json()['data'][0]['accession'] and request.json()['data'][0]['accession'] and \
        request.json()['data'][0]['name']:

        output["ebi_id"] = request.json()['data'][0]['accession']
        output["ebi_name"] = request.json()['data'][0]['name']
    
    return output

def get_hla_prot_sequence(ebi_id:str):

    url_template = "https://www.ebi.ac.uk/Tools/dbfetch/dbfetch?db=imgthlapro;id={}&format=fasta&style=raw"
    url = url_template.format(ebi_id)

    ebi_hla_prot_request = requests.get(url)

    raw_strand = ""
    if ebi_hla_prot_request.status_code == 200:
        raw_fasta_prot_txt = ebi_hla_prot_request.text

        regex_defintion = r"(>)(\w+:)(.+\n)([ACDEFGHIKLMNPQRSTVWY\n]+)"
        uniprot_parser = re.compile(regex_defintion)
        matches = uniprot_parser.findall(raw_fasta_prot_txt)

        if len(matches) > 0:
            raw_strand=matches[0][-1].replace('\n','').strip()
    
    return raw_strand

def get_unique_hla_alleles(df_hla_geno_tbl, hla_gene_types):
    alleles = []
    for hla_gene_type in hla_gene_types:

        unique_values = np.unique(np.concatenate((df_hla_geno_tbl[hla_gene_type+"_1"].values, 
        df_hla_geno_tbl[hla_gene_type+"_2"].values)))

        alleles_subset = [[hla_gene_type.upper(), value] for value in unique_values]
        alleles.extend(alleles_subset[:-1])
    return alleles

def retrieve_hla_meta_data_from_db(alleles):
    records = []
    failed_records = []
    for allele_tuple in alleles:
        allele_name = format_hla_allele_name(allele_tuple=allele_tuple)
        ebi_hla_details = get_hla_allele_details(allele_name=allele_name)
        if "ebi_id" in ebi_hla_details:
            ebi_hla_details["prot_seq"] = get_hla_prot_sequence(ebi_hla_details["ebi_id"])

        if "ebi_id" in ebi_hla_details and ebi_hla_details["prot_seq"] != "":
                
            ebi_hla_details["hla_gene"] = allele_tuple[0]
            ebi_hla_details["short_code"] = allele_tuple[1]
            record = [ebi_hla_details["ebi_id"], ebi_hla_details["ebi_name"], 
            ebi_hla_details["hla_gene"], ebi_hla_details["short_code"], 
            ebi_hla_details["prot_seq"]
                ]
            records.append(record)
        else:
            print("Failed to find record for {}".format(allele_tuple))
            record = [allele_tuple[0], allele_tuple[1], allele_name]
            failed_records.append(record)
        time.sleep(.1)
    return records, failed_records

def manually_get_deleted_records():
    record1 = ['HLA01437', 'B*47:01:01:02', 
            'B', '4701', 
            'MRVTAPRTLLLLLWGAVALTETWAGSHSMRYFYTAMSRPGRGEPRFITVGYVDDTLFVRFDSDATSPRKEPRAPWIEQEG'+ 
            'PEYWDRETQISKTNTQTYREDLRTLLRYYNQSEAGSHTLQRMFGCDVGPDGRLLRGYHQDAYDGKDYIALNEDLSSWTAA'+
            'DTAAQITQRKWEAARVAEQLRAYLEGECVEWLRRYLENGKETLQRADPPKTHVTHHPISDHEATLRCWALGFYPAEITLT'+
            'WQRDGEDQTQDTELVETRPAGDRTFQKWAAVVVPSGEEQRYTCHVQHEGLPKPLTLRWEPSSQSTVPIVGIVAGLAVLAV'+
            'VVIGAVVAAVVCRRKSSGGKGGSYSQAACSDSAQGSDVSLTA'
    ]
    record2 = ['HLA04311', 'C*17:01:01:02', 
                'C', '1701', 
                'MRVMAPQALLLLLSGALALIETWAGSHSMRYFYTAVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASPRGEPRAPWVEQEG'+
                'PEYWDRETQKYKRQAQADRVNLRKLRGYYNQSEAGSHTIQRMYGCDLGPDGRLLRGYNQFAYDGKDYIALNEDLRSWTAA'+
                'DTAAQISQRKLEAAREAEQLRAYLEGECVEWLRGYLENGKETLQRAERPKTHVTHHPVSDHEATLRCWALGFYPAEITLT'+
                'WQRDGEDQTQDTELVETRPAGDGTFQKWAAVVVPSGQEQRYTCHVQHEGLQEPCTLRWKPSSQPTIPNLGIVSGPAVLAV'+
                'LAVLAVLAVLGAVVAAVIHRRKSSGGKGGSCSQAASSNSAQGSDESLIACKA'
    ]
    records = [record1, record2]
    return records

config = cm.ConfigManaager().config
sql = msm.MySQLManager(config=config)

datasources = config['setup']['datasources']
table_names = [datasources[0]["table_name"], datasources[1]["table_name"], 
    datasources[2]["table_name"]
]

schema_name = datasources[0]["schema_name"]

df_hla_geno_tbl = sql.read_table_into_data_frame(schema_name=schema_name, 
    table_name=table_names[0]
)

hla_gene_types = ['a', 'b', 'c']

alleles = get_unique_hla_alleles(df_hla_geno_tbl, hla_gene_types)

records, failed_records = retrieve_hla_meta_data_from_db(alleles=alleles)

failed_records_df = pd.DataFrame(failed_records, 
columns=["hla_gene", "short_code", "allele_name"]
)

failed_records_df.to_csv("failed_records.csv")

print('Manually Insert EBI Deleted Records')
records.extend(manually_get_deleted_records())

sql.insert_records(schema_name="KIR_HLA_STUDY", 
    table_name="hla_allele", 
    column_names=["ebi_id", "ebi_name", "hla_gene", "short_code", "protein_sequence"], 
    values=records
)
