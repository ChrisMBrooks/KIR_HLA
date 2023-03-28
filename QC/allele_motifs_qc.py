import pandas as pd
import requests,re,time

from Controllers.DataScienceManager import DataScienceManager as dsm

# Part 1

def get_unique_motif_details(grid_hla_genotype, filename):
    unique_hla_motifs = grid_hla_genotype[['A_1', 'A_2', 'B_1', 'B_2', 'C_1', 'C_2']].values.reshape(-1, 1)[:, 0]
    unique_hla_motifs = list(set(unique_hla_motifs))

    records, failed_records = retrieve_hla_meta_data_from_ipd_db(unique_hla_motifs)

    records_df = pd.DataFrame(records, columns=["ebi_id", "ebi_name", "hla_gene", "short_code", "protein_sequence"])

    records_df.to_csv(filename)

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

def retrieve_hla_meta_data_from_ipd_db(alleles):
    records = []
    failed_records = []
    for allele_name in alleles:
        ebi_hla_details = get_hla_allele_details(allele_name=allele_name)
        if "ebi_id" in ebi_hla_details:
            ebi_hla_details["prot_seq"] = get_hla_prot_sequence(ebi_hla_details["ebi_id"])

        if "ebi_id" in ebi_hla_details and ebi_hla_details["prot_seq"] != "":
                
            ebi_hla_details["hla_gene"] = allele_name[0:1]
            ebi_hla_details["short_code"] = allele_name[2:7].replace(':', "")
            record = [ebi_hla_details["ebi_id"], ebi_hla_details["ebi_name"], 
            ebi_hla_details["hla_gene"], ebi_hla_details["short_code"], 
            ebi_hla_details["prot_seq"]
                ]
            records.append(record)
        else:
            print("Failed to find record for {}".format(allele_name))
            record = [allele_name[0:1], allele_name[1:6].replace(':', ""), allele_name]
            failed_records.append(record)
        #Sleep so as to not overwhelm the IP DB
        time.sleep(.1)
    return records, failed_records

def format_hla_genotypes(grid_hla_genotype:pd.DataFrame, columns:list):
    A = grid_hla_genotype.copy()
    for column_id in columns:
        new_id = column_id.replace('.', '_')
        A[new_id] = A[column_id].str.slice(start=0, stop=1) + '*' + \
            A[column_id].str.slice(start=1, stop=3) + ":" + \
            A[column_id].str.slice(start=3, stop=5)
    return A

# Part 2 

def compute_motif_posession0(allele_df:pd.DataFrame, 
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

def enrich_hla_allele_table(df_hla_allele_tbl):
    motif_definitions = {"c1":[(76,["V"]),(77,["S"]), (78,["L"]), (79,["R"]), (80,["N"])],
                        "c2":[(76,["V"]),(77,["N"]), (78,["L"]), (79,["R"]), (80,["K"])], 
                        "bw4":[(80,["T","I"])], 
                        "bw6":[(80,["N"])]
    }

    offset = 24

    # Compute Motif Posession
    motif_statuses = compute_motif_posession0(
        df_hla_allele_tbl, motif_definitions, offset
    )

    motif_statuses = [[item['c1'], item['c2'], item['bw4'], item['bw6']] 
        for item in motif_statuses]

    # Format and Export Results
    df_hla_allele_tbl[['c1', 'c2', 'bw4', 'bw6']] = motif_statuses

    return df_hla_allele_tbl.copy()

def trim_hla_prefixes(df:pd.DataFrame, columns:list):
    for column_id in columns:
        df[column_id] = df[column_id].str.slice(start=1)
        df[column_id] = df[column_id].astype('int32')
    return df.copy()

print('Running...')

#Import List of Alleles
filename = 'Data/All_IPD_HLA_LigandGroups.csv'
ipd_alleles = pd.read_csv(filename)

ligands = list(set(list(ipd_alleles['LigandGroup'])))

alleles = list(ipd_alleles['Allele'])
alleles = ['{}*{}'.format(x[0], x[1:6]) for x in alleles]
alleles = sorted(list(set(alleles)))

#Get Motif Posession from Alleles
#Pull Data from IPD
#records, failed_records = retrieve_hla_meta_data_from_ipd_db(alleles)
#records = pd.DataFrame(records, columns=["ebi_id", "ebi_name", "hla_gene", "short_code", "protein_sequence"])

#Write Raw Data to File
filename = 'QC/alleles_cross_check_ipd_raw_data.csv'
#records.to_csv(filename)
records = pd.read_csv(filename, index_col=0)

#failed_records = pd.DataFrame(failed_records, columns=['prefix', 'suffix', 'allele'])
#filename = 'QC/alleles_cross_check_ipd_defintions_failures.csv'
#failed_records.to_csv(filename)

#Enriche Records
#records = enrich_hla_allele_table(records)

#Export Final Results
filename = 'QC/alleles_cross_check_ipd_defintions.csv'
#records.to_csv(filename)

records = pd.read_csv(filename)

ipd_alleles['bw4_ba'] = (ipd_alleles['LigandGroup'] == 'Bw4-80I') | (ipd_alleles['LigandGroup'] == 'Bw4-80T')
ipd_alleles['c1_ba'] = ipd_alleles['LigandGroup'] == 'C1' 
ipd_alleles['c2_ba'] = ipd_alleles['LigandGroup'] == 'C2' 
ipd_alleles['bw6_ba'] = ipd_alleles['LigandGroup'] == 'Bw6' 
ipd_alleles['ebi_name_short'] = ipd_alleles['Allele'].str[0] + '*' + ipd_alleles['Allele'].str[1:len('01:01')+1]

records['ebi_name_short'] = records['ebi_name'].str[:len('A*01:01')]

merged_datasets = ipd_alleles.merge(records, on='ebi_name_short', how='right')

desired_columns = ['ebi_name_short', 'bw4_ba', 'bw4', 'c1_ba', 'c1', 'c2_ba', 'c2', 'bw6_ba', 'bw6']
merged_datasets = merged_datasets[desired_columns].copy()
merged_datasets = merged_datasets.drop_duplicates(ignore_index=True)

#Instantiate Controller
use_full_dataset = False
data_sci_mgr = dsm.DataScienceManager(use_full_dataset=use_full_dataset)
hla_geno_tbl_df = data_sci_mgr.sql.read_table_into_data_frame(schema_name='KIR_HLA_STUDY', table_name='raw_hla_genotype')
columns = ['public_id', 'a_1', 'a_2', 'b_1', 'b_2', 'c_1', 'c_2']
hla_geno_tbl_df = hla_geno_tbl_df[columns].copy()
for id in columns[1:]:
    hla_geno_tbl_df[id] = hla_geno_tbl_df[id].values.astype('int')
    hla_geno_tbl_df[id] = hla_geno_tbl_df[id].values.astype('str')
    hla_geno_tbl_df.loc[hla_geno_tbl_df[id].str.len() < 4, id] = '0' + hla_geno_tbl_df.loc[hla_geno_tbl_df[id].str.len() < 4, id]
    hla_geno_tbl_df[id] = id[0].upper() + hla_geno_tbl_df[id]

hla_geno_tbl_df = format_hla_genotypes(hla_geno_tbl_df, columns=columns[1:])

partition_df = data_sci_mgr.sql.read_table_into_data_frame(schema_name='KIR_HLA_STUDY', table_name='validation_partition')
hla_geno_tbl_df = partition_df.merge(hla_geno_tbl_df, how='left', on='public_id')

alleles = set()
for id in columns[1:]:
    new_set = set(list(hla_geno_tbl_df[id].values))
    alleles = alleles.union(new_set)

alleles = list(alleles)
alleles = sorted(alleles)

alleles = pd.DataFrame(alleles, columns=['ebi_name_short'])

merged_datasets = alleles.merge(merged_datasets, how='left', on='ebi_name_short')

columns = ['ebi_name_short', 'bw4_ba', 'bw4', 'c1_ba', 'c1', 'c2_ba', 'c2', 'bw6_ba', 'bw6']
merged_datasets = merged_datasets[columns].copy()

#Instantiate Controller
filename = 'QC/24032023/motif_posession_qc_merged_datasets.csv'
merged_datasets.to_csv(filename)
print('Complete.')