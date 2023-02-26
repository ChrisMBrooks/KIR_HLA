import numpy as np
import pandas as pd 

from Controllers.ConfigManager import ConfigManager as cm
from Controllers.MySQLManager import MySQLManager as msm
from Controllers.DataManager import DataManager as dtm
from Controllers.LearningManager import LearningManager as lrn

#Instantiate Controllers
config = cm.ConfigManaager().config
sql = msm.MySQLManager(config=config)
data_mgr = dtm.DataManager(config=config, use_full_dataset=False)
lrn_mgr = lrn.LearningManager(config=config)

hla_allele_df = sql.read_table_into_data_frame(schema_name='KIR_HLA_STUDY', table_name='hla_allele')
hla_geno_df = sql.read_table_into_data_frame(schema_name='KIR_HLA_STUDY', table_name='raw_hla_genotype')

hla_allele_df['long_code'] = hla_allele_df['hla_gene'] + hla_allele_df['short_code'].astype('str')


filename = 'Data/laura_func_kir_counts.csv'
qc_data = pd.read_csv(filename)

columns = hla_geno_df.columns
hla_geno_df.merge(qc_data, how='inner', left_on='public_id', right_on='PublicID')
hla_geno_df = hla_geno_df[columns]

columns = ['public_id', 'a_1', 'a_2', 'b_1', 'b_2', 'c_1', 'c_2']
hla_geno_df = hla_geno_df[columns]

hla_geno_df

### KIR3DL1 binds HLA-B alleles with Bw4 motif (Asparagine 77)
HLABw4 = ["B1302", "B1516", "B1517", "B1524", "B2701", "B2702", "B2704", "B2705", "B3701", "B3801", "B3802", "B4402", "B4403",
    "B4404", "B4405", "B4414", "B4417", "B4429", "B4435", "B4701", "B4901", "B5101", "B5108", "B5201", "B5301", "B5302", "B5701", "B5702", "B5801"]

## KIR2DL2 and KIR2DL3
### C*01, C*03, C*07, C*08, C*12, C*14, C*16 (C1) Asparagine position 80 (N)
HLAC1 = ["C0102", "C0302", "C0303", "C0304", "C0310", "C0701", "C0702", "C0704", "C0802", "C0803", "C1202", "C1203","C1402", "C1403", "C1601","C1602", "C1604", "B4601", "B7301"]

## KIR2DL1
###c*02, C*04, C*05, C*06, C*15, C*17, C*18 (C2) Lysine position 80 (K)
HLAC2 = ["C0202", "C0210", "C0401", "C0501","C0602", "C1502", "C1505", "C1701"]

A = HLABw4 + HLAC1 + HLAC2
A = set(A)

print(A)

motifs = []
for index, item in enumerate(A):
    bw4 = item in HLABw4
    c1 = item in HLAC1
    c2 = item in HLAC2
    motifs.append([item, c1, c2, bw4])
  
motifs_df = pd.DataFrame(motifs, columns= ['allele', 'has_c1', 'has_c2', 'has_bw4'])



hla_allele_df = hla_allele_df.merge(motifs_df, how='inner', left_on='long_code', right_on='allele')

columns = ['ebi_id', 'ebi_name', 'long_code',
       'c1', 'c2', 'bw4', 'bw6', 'has_c1', 'has_c2',
       'has_bw4', 'protein_sequence']
hla_allele_df = hla_allele_df[columns].copy()

# filename = '/Users/chrismbrooks/Desktop/HLA_alleles_qc.csv'
# hla_allele_df.to_csv(filename)