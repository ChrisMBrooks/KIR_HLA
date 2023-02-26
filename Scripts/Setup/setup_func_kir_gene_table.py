import os, random, math
import pandas as pd
import numpy as np

from Controllers.ConfigManager import ConfigManager as cm
from Controllers.MySQLManager import MySQLManager as msm

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
    # Computer if subject has motif
    ligand_matching_criteria = ['hla_c_c1', 'hla_c_c2', 'hla_b_46_c1', 'hla_b_73_c1', 'hla_b_bw4']
    motifs = ['c1', 'c2', 'bw4', 'bw6']

    mstr_data = df_hla_geno_tbl
    ref_data = df_hla_allele_tbl
    motifs_records = []
    for index, row in mstr_data.iterrows():
        critereon_results = {key:False for key in ligand_matching_criteria}
        for i in range(3, len(summplemental_columns), 2):

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

            motif_stats = [ x >= 1 for x in (x1 + x2)]

            if hla_loci == 'c':
                critereon_results['hla_c_c1'] = motif_stats[0]
                critereon_results['hla_c_c2'] = motif_stats[1]
            elif hla_loci == 'b':
                has_46_c1 = False
                if not pd.isna(row['fc_b_1']) and row['fc_b_1'][1:3] == '46' and motif_stats[0]: 
                    has_46_c1 = True

                if not pd.isna(row['fc_b_2']) and row['fc_b_2'][1:3] == '46' and motif_stats[0]: 
                    has_46_c1 = True

                has_73_c1 = False
                if not pd.isna(row['fc_b_1']) and row['fc_b_1'][1:3] == '73' and motif_stats[0]: 
                    has_73_c1 = True
                if not pd.isna(row['fc_b_2']) and row['fc_b_2'][1:3] == '73' and motif_stats[0]: 
                    has_73_c1 = True

                critereon_results['hla_b_46_c1'] = has_46_c1
                critereon_results['hla_b_73_c1'] = has_73_c1
                critereon_results['hla_b_bw4'] = motif_stats[2] 
        results_record = [critereon_results[key] for key in ligand_matching_criteria]
        motifs_records.append(results_record)

    columns = ['public_id']
    columns.extend(ligand_matching_criteria)
    mstr_data[ligand_matching_criteria] = motifs_records
    return mstr_data[columns].copy()

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
    mstr_df['f_kir3dl1'] = mstr_df['kir3dl1'] & mstr_df['hla_b_bw4']

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
config = cm.ConfigManaager().config
sql = msm.MySQLManager(config=config)

datasources = config['setup']['datasources']
table_names = [datasources[0]["table_name"], datasources[1]["table_name"], 
    datasources[2]["table_name"]
]
schema_name = datasources[0]["schema_name"]

# Pull Data from DB
df_kir_geno_tbl = sql.read_table_into_data_frame(schema_name=schema_name,
    table_name=table_names[1]
)

df_hla_geno_tbl = sql.read_table_into_data_frame(schema_name=schema_name, 
    table_name=table_names[0]
)

df_hla_allele_tbl = sql.read_table_into_data_frame(schema_name=schema_name, 
    table_name='hla_allele'
)

# Compute Functional Kir Gene Posession
f_kir_df = compute_functional_kir_genotype(df_kir_geno_tbl,df_hla_geno_tbl, df_hla_allele_tbl)

columns = ['public_id', 'kir2dl1', 'kir2dl2', 'kir2dl3', 'kir3dl1', 'hla_c_c1',
       'hla_c_c2', 'hla_b_46_c1', 'hla_b_73_c1', 'hla_b_bw4', 'f_kir2dl1',
       'f_kir2dl2_s', 'f_kir2dl2_w', 'f_kir2dl3', 'f_kir3dl1', 'f_kir_count',
       'f_kir_score'
]

# Export Data
records = [[value for key, value in row.items()] for index, row in f_kir_df.iterrows()] 

sql.insert_records(schema_name="KIR_HLA_STUDY", 
    table_name="functional_kir_genotype", 
    column_names=columns, 
    values=records
)