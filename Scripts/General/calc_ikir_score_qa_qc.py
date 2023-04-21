
# Compare and Contrast Laura's Methods w/ Chris's Methods 
# to calculate iKIR score on a subset of the GRID dataset

import time, requests, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Controllers.ConfigManager import ConfigManager as cm
from Controllers.MySQLManager import MySQLManager as msm
from Controllers.DataManager import DataManager as dtm
from Controllers.LearningManager import LearningManager as lrn

# Part 1

def get_unique_motif_details(grid_hla_genotype, filename):
    unique_hla_motifs = list(set(grid_hla_genotype[['A_1', 'A_2', 'B_1', 'B_2', 'C_1', 'C_2']].values.reshape(-1, 1)[:, 0]))

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

# Part 3

def add_col_full_hla_nomenclature1(df:pd.DataFrame):
    df['full_code'] = df['short_code'].astype('str') 

    df.loc[df['short_code'] < 1000, 'full_code'] = \
        "0" + df['short_code'].astype('string')

    df['full_code'] = df['hla_gene'] + df['full_code']

    return df

def add_col_full_hla_nomenclature2(df:pd.DataFrame, key:str):
    prefix = key[0].upper()
    new_key = 'fc_{}'.format(key)
    
    df[key].fillna(0.0, inplace=True)

    df[new_key] = df[key].astype('int').astype('str') 

    df.loc[df[key] < 1000, new_key] = \
        "0" + df[key].astype('int').astype('string')

    df[new_key] = prefix + df[new_key]

    pseudo_null = prefix + "00"
    df.loc[df[new_key] == pseudo_null, new_key] = np.nan

    return df

def compute_kir_gene_posession(df:pd.DataFrame):
    df['kir2dl1'] = df['kir2dl1_t50'] > 0

    df['kir2dl2'] = df['kir2dl2_t50'] > 0

    df['kir2dl3'] = df['kir2dl3_t50'] > 0

    df['kir3dl1'] = df['kir3dl1ex4_t50'] + df['kir3dl1ex9_t50'] > 0
    
    columns = ['public_id', 'kir2dl1', 'kir2dl2', 'kir2dl3', 'kir3dl1']
    return df[columns].copy()

def compute_motif_posession(df_hla_geno_tbl:pd.DataFrame, 
    df_hla_allele_tbl:pd.DataFrame
):
    # Refer to Paper for Definitions: 
    # https://doi.org/10.1016/j.jaip.2022.04.036

    #String smithing into new columns...
    # Compute Compound HLA Nomenclature, vis-a-vis Z:NN:NN
    df_hla_allele_tbl = add_col_full_hla_nomenclature1(df_hla_allele_tbl)

    snp_imputation_keys = ['a_1', 'a_2', 'b_1', 'b_2', 'c_1', 'c_2']
    for key in snp_imputation_keys:
        df_hla_geno_tbl = add_col_full_hla_nomenclature2(df_hla_geno_tbl, key)
    
    summplemental_columns = ['public_id', 'fc_a_1', 'fc_a_2', 
        'fc_b_1', 'fc_b_2', 'fc_c_1', 'fc_c_2']
    df_hla_geno_tbl = df_hla_geno_tbl[summplemental_columns].copy()

    # Use subject specific alleles to look up allele reference data, then
    # Compute if subject has motif
    ligand_matching_criteria = ['hla_c_c1', 'hla_c_c2', 'hla_b_46_c1', 'hla_b_73_c1', 'hla_b_bw4', 'hla_a_bw4']
    motifs = ['c1', 'c2', 'bw4', 'bw6']

    mstr_data = df_hla_geno_tbl
    ref_data = df_hla_allele_tbl
    motifs_records = []
    for index, row in mstr_data.iterrows():
        critereon_results = {key:False for key in ligand_matching_criteria}
        for i in range(1, len(summplemental_columns), 2):

            hla_code1 = row[summplemental_columns[i]]
            hla_code2 = row[summplemental_columns[i+1]]
            hla_loci = summplemental_columns[i][3:4].lower()

            #Pull Details for each copy of gene (e.g. c_1, c_2)
            ref_r1 = ref_data.loc[ref_data['full_code'] == hla_code1, motifs]
            ref_r2 = ref_data.loc[ref_data['full_code'] == hla_code2, motifs]

            #Handle Null Values
            if len(ref_r1.values) > 0:
                x1 = ref_r1.values[0]
            else:
                x1 = np.zeros(4)

            #Handle Null Values
            if len(ref_r2.values) > 0:
                x2 = ref_r2.values[0]
            else:
                x2 = np.zeros(4)

            motif_stats = [ x >= 1 for x in (x1 + x2)] #i.e. [c1_bool, c2_bool, bw4_bool, bw6_bool]

            if hla_loci == 'c':
                critereon_results['hla_c_c1'] = motif_stats[0] # i.e. c1
                critereon_results['hla_c_c2'] = motif_stats[1] # i.e. c2

            elif hla_loci == 'b':
                if not pd.isna(row['fc_b_1']) and row['fc_b_1'][1:3] == '46' and motif_stats[0]: # i.e. b46 & c1
                    critereon_results['hla_b_46_c1'] = True

                if not pd.isna(row['fc_b_2']) and row['fc_b_2'][1:3] == '46' and motif_stats[0]: # i.e. b46 & c1
                    critereon_results['hla_b_46_c1'] = True

                if not pd.isna(row['fc_b_1']) and row['fc_b_1'][1:3] == '73' and motif_stats[0]: # i.e. b73 & c1
                    critereon_results['hla_b_73_c1']
                if not pd.isna(row['fc_b_2']) and row['fc_b_2'][1:3] == '73' and motif_stats[0]: # i.e. b73 & c1
                    critereon_results['hla_b_73_c1']

                critereon_results['hla_b_bw4'] = motif_stats[2] # i.e. bw4

            elif hla_loci == 'a':
                if not pd.isna(row['fc_a_1']) and row['fc_a_1'][1:3] in ['23', '24', '32'] and motif_stats[2]: # e.g. A*2301 & bw4
                    critereon_results['hla_a_bw4'] = True

                if not pd.isna(row['fc_a_2']) and row['fc_a_2'][1:3] in ['23', '24', '32'] and motif_stats[2]: # e.g. A*2301 & bw4
                    critereon_results['hla_a_bw4'] = True
            
        results_record = [row['public_id']] + [critereon_results[key] for key in ligand_matching_criteria]
        motifs_records.append(results_record)

    columns = ['public_id']
    columns.extend(ligand_matching_criteria)
    temp = pd.DataFrame(motifs_records, columns=columns)

    return temp

def compute_functional_kir_genotype(df_kir_geno_tbl:pd.DataFrame, 
    df_hla_geno_tbl:pd.DataFrame, df_hla_allele_tbl:pd.DataFrame
):

    df_subject_kir_status = compute_kir_gene_posession(df=df_kir_geno_tbl)
    del df_kir_geno_tbl

    df_subject_motif_status = compute_motif_posession(df_hla_geno_tbl, df_hla_allele_tbl)
    del df_hla_geno_tbl, df_hla_allele_tbl

    mstr_df = df_subject_kir_status.merge(df_subject_motif_status, 
        on='public_id', how='inner')

    #Compute Aggregate Criteria
    # Refer to Supplemental Material in https://www.science.org/doi/10.1126/sciimmunol.aao2892
    # For critereon definitions. 

    mstr_df['f_kir2dl1'] = mstr_df['kir2dl1'] & mstr_df['hla_c_c2']
    mstr_df['f_kir2dl2_s'] = mstr_df['kir2dl2'] & (mstr_df['hla_c_c1'] | mstr_df['hla_b_46_c1'] | mstr_df['hla_b_73_c1'])
    mstr_df['f_kir2dl2_w'] = mstr_df['kir2dl2'] & mstr_df['hla_c_c2']
    mstr_df['f_kir2dl3'] = mstr_df['kir2dl3'] & (mstr_df['hla_c_c1'] | mstr_df['hla_b_46_c1'] | mstr_df['hla_b_73_c1'])
    mstr_df['f_kir3dl1'] = mstr_df['kir3dl1'] & (mstr_df['hla_b_bw4'] | mstr_df['hla_a_bw4'])

    mstr_df['f_kir_count'] =  mstr_df['f_kir2dl1'].astype('int32') + \
        (mstr_df['f_kir2dl2_s'].astype('int32') | mstr_df['f_kir2dl2_w'].astype('int32')) + \
        mstr_df['f_kir2dl3'].astype('int32') + \
        mstr_df['f_kir3dl1'].astype('int32')
    
    # Definition from Supplemental Materials - Boelen 2018
    # Inhibitory score= (1 if Func 2DL1) + (1 if Strong Func 2DL2 or 0.5 if weak Func 2DL2) +
    # (0.75 if Func 2DL3) + (1 if Func 3DL1).

    mstr_df['A'] = mstr_df['f_kir2dl2_w'] & ~mstr_df['f_kir2dl2_s']
    mstr_df['B'] = np.where(mstr_df['A'], 1, 0)

    mstr_df['f_kir_score'] = mstr_df['f_kir2dl1'].astype('int32')*1.0 + \
        mstr_df['f_kir2dl2_s'].astype('int32')*1.0 + \
        mstr_df['B'].astype('int32')*0.5 + \
        mstr_df['f_kir2dl3'].astype('int32')*0.75 + \
        mstr_df['f_kir3dl1'].astype('int32')*1.0

    mstr_df = mstr_df.drop(['A', 'B'], axis=1)

    return mstr_df

#Instantiate Controllers
use_full_dataset=False
config = cm.ConfigManaager().config
sql = msm.MySQLManager(config=config)
data_mgr = dtm.DataManager(config=config, use_full_dataset=use_full_dataset)
lrn_mgr = lrn.LearningManager(config=config)

grid_filename = "Data/grid_cohort_sample_14022023.csv"
grid_sample_data = pd.read_csv(grid_filename)
grid_results = grid_sample_data[['ID', 'iKIR_Count', 'iKIR_score']].copy()

grid_hla_genotype = grid_sample_data[['ID', 'A.1', 'A.2', 'B.1', 'B.2', 'C.1', 'C.2']].copy()
grid_hla_genotype = format_hla_genotypes(grid_hla_genotype, ['A.1', 'A.2', 'B.1', 'B.2', 'C.1', 'C.2'])

filename = "QC/15022023/motif_details.csv"
#get_unique_motif_details(grid_hla_genotype, filename)
df_hla_allele_tbl = pd.read_csv(filename, index_col=0)
df_hla_allele_tbl = enrich_hla_allele_table(df_hla_allele_tbl)

df_kir_geno_tbl = grid_sample_data[['ID', 'KIR2DL1_t50', 'KIR2DL2_t50', 'KIR2DL3_t50', 'KIR3DL1_t50']].copy()
df_kir_geno_tbl = df_kir_geno_tbl.rename(columns={"ID": "public_id", "KIR2DL1_t50": "kir2dl1_t50", "KIR2DL2_t50": "kir2dl2_t50", "KIR2DL3_t50": "kir2dl3_t50", "KIR3DL1_t50": "kir3dl1ex4_t50"})
df_kir_geno_tbl['kir3dl1ex9_t50'] = df_kir_geno_tbl['kir3dl1ex4_t50'].values 

df_hla_geno_tbl = grid_sample_data[['ID', 'A.1', 'A.2', 'B.1', 'B.2', 'C.1', 'C.2']].copy()
df_hla_geno_tbl = df_hla_geno_tbl.rename(columns={"ID": "public_id", "A.1": "a_1", "A.2": "a_2", "B.1": "b_1", "B.2": "b_2", "C.1": "c_1", "C.2": "c_2"})
df_hla_geno_tbl = trim_hla_prefixes(df_hla_geno_tbl, ['a_1', 'a_2', 'b_1', 'b_2', 'c_1', 'c_2'])

# Compute Functional Kir Gene Posession
f_kir_df = compute_functional_kir_genotype(df_kir_geno_tbl,df_hla_geno_tbl, df_hla_allele_tbl)

f_kir_df2 = f_kir_df.merge(grid_sample_data, left_on='public_id', right_on='ID', how='right')
filename = "QC/15022023/comparison_details_15022023_2.csv"
f_kir_df2.to_csv(filename)

results = f_kir_df[['public_id', 'f_kir_count', 'f_kir_score']].copy()
results = results.merge(grid_results, left_on='public_id', right_on='ID', how='right')

results['count_diff'] = results['f_kir_count'] - results['iKIR_Count']
results['score_diff'] = results['f_kir_score'] - results['iKIR_score']

results 

print(results)

#grid_kir_genotype = grid_sample_data[['ID', 'KIR2DL1_t50', 'KIR2DL2_t50', 'KIR2DL3_t50', 'KIR3DL1_t50']]
#grid_results = grid_sample_data[['ID', 'iKIR_Count', 'iKIR_score']].copy()

#print(unique_hla_motifs)