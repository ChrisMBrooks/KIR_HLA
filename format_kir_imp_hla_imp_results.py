import pandas as pd
import numpy as np

def format_kir_imp_results(source_filename:str, output_filename:str):
    kir_imputations_df = pd.read_csv(source_filename)
    columns = ['id_1', 'id_2', 'haplotype_id', 'locus', 'imputed_type', 'posterior_probability']

    kir_imputations_df = pd.DataFrame(kir_imputations_df.values, columns=columns)

    kir_imputation_records = {}

    #loci = ['KIRhaplotype', 'AvsB', 'KIR2DS2', 'KIR2DL2', 'KIR2DL3', 'KIR2DP1', 'KIR2DL1',
    # 'KIR3DP1', 'KIR2DL4', 'KIR3DL1ex4', 'KIR3DL1ex9', 'KIR3DS1', 'KIR2DL5',
    # 'KIR2DS3', 'KIR2DS5', 'KIR2DS1', 'KIR2DS4TOTAL', 'KIR2DS4WT', 'KIR2DS4DEL']

    loci = ['KIR2DS2', 'KIR2DL2', 'KIR2DL3', 'KIR2DP1', 'KIR2DL1',
    'KIR3DP1', 'KIR2DL4', 'KIR3DL1ex4', 'KIR3DL1ex9', 'KIR3DS1', 'KIR2DL5',
    'KIR2DS3', 'KIR2DS5', 'KIR2DS1', 'KIR2DS4TOTAL', 'KIR2DS4WT', 'KIR2DS4DEL']

    thresholds = {'t50':0.50, 't70':0.70, 't80':0.80}
    for subject_id in np.unique(kir_imputations_df['id_2'].values):
        kir_imputation_records[subject_id] = {'ID':subject_id}
        relevant_rows = kir_imputations_df[kir_imputations_df['id_2']== subject_id] 

        for locus in loci:
            haplotypes = relevant_rows[relevant_rows['locus']==locus]
            hap1 = haplotypes.iloc[0]
            hap2 = haplotypes.iloc[1]

            partial_record = {}
            for th_key in thresholds:
                key = '{}_{}'.format(locus, th_key)
                kir_imputation_records[subject_id][key] = np.nan
                if (hap1['posterior_probability'] > thresholds[th_key] and int(hap1['imputed_type']) == 1) or \
                    (hap2['posterior_probability'] > thresholds[th_key] and int(hap2['imputed_type']) == 1):
                    kir_imputation_records[subject_id][key] = 1
                elif (hap1['posterior_probability'] > thresholds[th_key] and int(hap1['imputed_type']) == 0) and \
                    (hap2['posterior_probability'] > thresholds[th_key] and int(hap2['imputed_type']) == 0):
                    kir_imputation_records[subject_id][key] = 1

    kir_imputation_records = [kir_imputation_records[key] for key in kir_imputation_records]
    kir_imputations_df = pd.DataFrame(kir_imputation_records)

    columns = ["ID", "KIR2DL1_t50", "KIR2DL2_t50", "KIR2DL3_t50", 
    "KIR2DL4_t50", "KIR2DL5_t50", "KIR2DP1_t50", "KIR2DS1_t50", 
    "KIR2DS2_t50", "KIR2DS3_t50", "KIR2DS4DEL_t50", "KIR2DS4TOTAL_t50", 
    "KIR2DS4WT_t50", "KIR2DS5_t50", "KIR3DL1ex4_t50", "KIR3DL1ex9_t50",
    "KIR3DP1_t50", "KIR3DS1_t50", "KIR2DL1_t70", 
    "KIR2DL2_t70", "KIR2DL3_t70", "KIR2DL4_t70", "KIR2DL5_t70", 
    "KIR2DP1_t70", "KIR2DS1_t70", "KIR2DS2_t70", "KIR2DS3_t70", 
    "KIR2DS4DEL_t70", "KIR2DS4TOTAL_t70", "KIR2DS4WT_t70", 
    "KIR2DS5_t70", "KIR3DL1ex4_t70", "KIR3DL1ex9_t70", "KIR3DP1_t70", 
    "KIR3DS1_t70", "KIR2DL1_t80", "KIR2DL2_t80", "KIR2DL3_t80", 
    "KIR2DL4_t80", "KIR2DL5_t80", "KIR2DP1_t80", "KIR2DS1_t80", 
    "KIR2DS2_t80", "KIR2DS3_t80", "KIR2DS4DEL_t80", "KIR2DS4TOTAL_t80", 
    "KIR2DS4WT_t80", "KIR2DS5_t80", "KIR3DL1ex4_t80", "KIR3DL1ex9_t80", 
    "KIR3DP1_t80", "KIR3DS1_t80"]

    kir_imputations_df = kir_imputations_df[columns].copy()

    kir_imputations_df.to_csv(output_filename, index=None)

    return output_filename

def format_hla_imp_results(source_filename:str, output_filename:str, threshold:float):
    hla_imputation_df = pd.read_csv(source_filename)

    columns = ['id_1', 'id_2', 'haplotype_id', 'locus', 'imputed_type', 'posterior_probability']
    hla_imputation_df = pd.DataFrame(hla_imputation_df.values, columns=columns)

    loci = ["A", "B", "C", "DRB1", "DQA1", "DQB1"]

    hla_imputation_records = {}

    for subject_id in np.unique(hla_imputation_df['id_1'].values):
        hla_imputation_records[subject_id] = {'ID_1':subject_id}
        relevant_rows = hla_imputation_df[hla_imputation_df['id_1']== subject_id] 

        for locus in loci:
            full_loci = '{}{}'.format('HLA', locus)
            haplotypes = relevant_rows[relevant_rows['locus']==full_loci]

            key_format = '{}.{}'
            key1 = key_format.format(locus, 1)
            key2 = key_format.format(locus, 2)

            hap1 = haplotypes.iloc[0]
            hap2 = haplotypes.iloc[1]

            if float(hap1['posterior_probability']) > threshold:
                hla_imputation_records[subject_id][key1] = hap1['imputed_type']
            else:
                hla_imputation_records[subject_id][key1] = np.NaN
            
            if float(hap2['posterior_probability']) > threshold:
                hla_imputation_records[subject_id][key2] = hap2['imputed_type']
            else:
                hla_imputation_records[subject_id][key2] = np.NaN

    hla_imputation_records = [hla_imputation_records[key] for key in hla_imputation_records]

    columns = ["ID_1", "A.1", "A.2", "B.1", "B.2", "C.1", "C.2", "DRB1.1", "DRB1.2", "DQA1.1", "DQA1.2", "DQB1.1", "DQB1.2"]

    hla_imputation_df = pd.DataFrame(hla_imputation_records, columns=columns)

    hla_imputation_df.to_csv(output_filename, index=None)

    return output_filename


source_filename = '/Users/chrismbrooks/Downloads/imputation_results/imputations.csv'
output_filename = '/Users/chrismbrooks/Downloads/imputation_results/kir_imputations_df.csv'

#source_filename = format_kir_imp_results(source_filename=source_filename, output_filename=output_filename)
#kir_imputations_df = pd.read_csv(source_filename, index_col=None)
#print(kir_imputations_df)


source_filename = '/Users/chrismbrooks/Documents/Imperial/Asquith Group/Raw Data/Twins UK/E1139_14092021/HLA_Imputation/HLA_imputations_ALL_TUK_sample_1000G.CSV'
output_filename = '/Users/chrismbrooks/Downloads/hla_imputations_df.csv'

source_filename = format_hla_imp_results(source_filename=source_filename, output_filename=output_filename, threshold=0.5)

hla_imputations_df = pd.read_csv(source_filename, index_col=None)
print(hla_imputations_df)