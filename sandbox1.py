
import math, random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme()

import pandas as pd
from Controllers.DataScienceManager import DataScienceManager as dsm

#Instantiate Controllers
use_full_dataset = False
use_database = False
data_sci_mgr = dsm.DataScienceManager(
    use_full_dataset=use_full_dataset, 
    use_database=use_database
)

"""
date_str = data_sci_mgr.data_mgr.get_date_str()
filename = "Analysis/General/candidate_feature_correlates_{}.parquet".format(date_str)
correlate_df = pd.read_parquet(filename)
correlates = correlate_df[['phenotype_id', 'correlates']].values

for index, label in enumerate(correlates[:, 0]):
    print(label)
    print(correlates[index, 1])
"""
"""
#Correlates Summary
MFI_469_corr = []

P1_20790_corr = ['P1:20709']

P1_4229_corr = ['P1:3986', 'P1:4256']

P1_20054_corr = ['P1:20027', 'P1:20030', 'P1:20057' ,'P1:20102', 'P1:20104', 'P1:20127',
 'P1:20130', 'P1:22210', 'P1:22213']

P1_20054_corr = set(P1_20054_corr)

P1_22210_corr = ['P1:20054', 'P1:20127','P1:20130', 'P1:22183', 'P1:22186','P1:22213']

P1_22210_corr = set(P1_22210_corr)

P1_22213_corr = ['P1:20054', 'P1:20057', 'P1:20102', 'P1:20104', 'P1:20127', 'P1:20130',
 'P1:22183', 'P1:22186', 'P1:22210']

P1_22213_corr = set(P1_22213_corr)

A = P1_20054_corr.union(P1_22210_corr).union(P1_22213_corr)
print('A:', list(A))
"""

"""

Trait ID	Subset or Surface Protein	Full Subset Name
MFI:s469	CD158a/h FITC	Lymphs/nonT/NK/16+56+/CD158a
MFI:s486	CD158a/h FITC	Lymphs/nonT/NK/16-56+/CD158a
MFI:s451	CD158a/h FITC	Lymphs/nonT/NK/16+56-/CD158a

"""

"""
phenos_df_t = data_sci_mgr.data_mgr.outcomes(fill_na=False, fill_na_value=None, partition='training')
phenos_df_v = data_sci_mgr.data_mgr.outcomes(fill_na=False, fill_na_value=None, partition='validation')

scores_df_t = data_sci_mgr.data_mgr.features(fill_na=False, fill_na_value=None, partition='training')
scores_df_v = data_sci_mgr.data_mgr.features(fill_na=False, fill_na_value=None, partition='validation')

phenos_df = pd.concat([phenos_df_t, phenos_df_v])
scores_df = pd.concat([scores_df_t, scores_df_v])

scores_df['ikir_count'] = scores_df['kir2dl1'] + scores_df['kir2dl2'] + scores_df['kir2dl3'] + scores_df['kir3dl1']

data = scores_df.merge(phenos_df, how='inner', on='public_id')
phenos_to_plot = ['MFI:469', 'MFI:486', 'MFI:451']

x_label = 'ikir_count'
"""


"""for pheno_label in phenos_to_plot:
    sns.regplot(data=data, x=x_label, y=pheno_label)
    plt.title('{} - Scatter Plot & Trendline'.format(pheno_label))
    filename = 'Analysis/Univariate/scatter_plot_{}_{}_{}.png'.format(pheno_label, x_label, data_sci_mgr.data_mgr.get_date_str())
    plt.savefig(filename)
    plt.clf()
    plt.cla()
"""

"""
data = data[phenos_to_plot].copy().corr(method='pearson')
sns.heatmap(data, annot = True)
plt.title('Correlation Heat Map')
filename = 'Analysis/Univariate/heatmap_{}_{}.png'.format(phenos_to_plot[0], data_sci_mgr.data_mgr.get_date_str())
plt.savefig(filename)
plt.show()
"""

"""
data = data[A].copy().corr(method='pearson')
sns.heatmap(data, annot = True, center = 0.5)
plt.title('Correlation Heat Map')
filename = 'Analysis/Univariate/heatmap_{}_{}.png'.format(list(A)[0], data_sci_mgr.data_mgr.get_date_str())
plt.savefig(filename)
plt.show()
"""

"""
pheno_definitions = data_sci_mgr.sql.read_table_into_data_frame(schema_name='KIR_HLA_STUDY', 
    table_name='immunophenotype_definitions'
)

A = ['P1:20104', 'P1:22186', 'P1:20030', 'P1:20054', 'P1:20127', 'P1:20130', 'P1:20027', 'P1:20057', 'P1:22213', 'P1:20102', 'P1:22210', 'P1:22183']
B = pd.Series(A, name='phenotype_id')
B = pheno_definitions.merge(B, how='right', on='phenotype_id')
print(B)
data = data[A].copy().corr(method='pearson').sum(axis=1).argmax()
best_rep = A[data]

print(best_rep)

"""

"""A = ['MFI:469', 'P1:20104', 'P1:22186', 'P1:20030', 'P1:20054', 'P1:20127', 
     'P1:20130', 'P1:20027', 'P1:20057', 'P1:22213', 'P1:20102', 
     'P1:22210', 'P1:22183', 'P1:3986' , 'P1:4256' , 'P1:4229',
     'P1:20709', 'P1:20790'
]

data = data[A].copy().corr(method='pearson')
sns.heatmap(data, annot = True, center = 0.5)
plt.title('Correlation Heat Map')
plt.show()"""

"""
A = ['MFI:469', 'P1:20102', 'P1:4229', 'P1:20709']
"""

"""
partition = data_sci_mgr.sql.read_table_into_data_frame(schema_name='KIR_HLA_STUDY', table_name='validation_partition')

functional_kir_genotype = data_sci_mgr.sql.read_table_into_data_frame(schema_name='KIR_HLA_STUDY', table_name='functional_kir_genotype')

functional_kir_genotype = functional_kir_genotype[['public_id','f_kir_score']]

filename = '/Users/chrismbrooks/Downloads/export2.csv'
functional_kir_genotype2 = pd.read_csv(filename, index_col=0)

functional_kir_genotype2 = functional_kir_genotype2[['public_id','f_kir_score']]

functional_kir_genotype = functional_kir_genotype.merge(functional_kir_genotype2, how='left', on='public_id')
functional_kir_genotype['score_diff'] = functional_kir_genotype['f_kir_score_y']  - functional_kir_genotype['f_kir_score_x'] 
functional_kir_genotype = functional_kir_genotype.merge(partition, how='right', on='public_id')

print(functional_kir_genotype)
print(functional_kir_genotype['score_diff'].values.sum())
filename = '/Users/chrismbrooks/Downloads/export4.csv'
functional_kir_genotype.to_csv(filename)
print('Complete')

"""
"""
A = ['TUK48882521', 'TUK48882522', 'TUK78844781', 'TUK78844782', 'TUK57801342', 
 'TUK55940671', 'TUK26259072', 'TUK68658062', 'TUK80787172', 'TUK89005991', 
 'TUK89005992', 'TUK43411312', 'TUK43510362', 'TUK68658061', 'TUK92753351', 
 'TUK92753352'
]

discrepancies = pd.Series(A, name='public_id')

raw_kir_genotype = data_sci_mgr.sql.read_table_into_data_frame(schema_name='KIR_HLA_STUDY', table_name='raw_kir_genotype')
raw_hla_genotype = data_sci_mgr.sql.read_table_into_data_frame(schema_name='KIR_HLA_STUDY', table_name='raw_hla_genotype')
functional_kir_genotype = data_sci_mgr.sql.read_table_into_data_frame(schema_name='KIR_HLA_STUDY', table_name='functional_kir_genotype')
partition = data_sci_mgr.sql.read_table_into_data_frame(schema_name='KIR_HLA_STUDY', table_name='validation_partition')

discrepancies = partition.merge(discrepancies, how='right', on='public_id')
discrepancies = discrepancies.merge(raw_kir_genotype, how='left', on='public_id')
discrepancies = discrepancies.merge(raw_hla_genotype, how='left', on='public_id')
discrepancies = discrepancies.merge(functional_kir_genotype, how='left', on='public_id')

required_columns = ['public_id', 'subject_id', 'kir2dl1_t50', 'kir2dl2_t50', 'kir2dl3_t50', 'kir3dl1ex4_t50', 'kir3dl1ex9_t50', 
    'a_1', 'a_2', 'b_1', 'b_2', 'c_1', 'c_2']

discrepancies = discrepancies[required_columns].copy()

filename = '/Users/chrismbrooks/Downloads/nulls_required_columns_cross_check.csv'
discrepancies.to_csv(filename)
"""
"""
raw_kir_genotype = data_sci_mgr.sql.read_table_into_data_frame(schema_name='KIR_HLA_STUDY', table_name='raw_kir_genotype')
raw_hla_genotype = data_sci_mgr.sql.read_table_into_data_frame(schema_name='KIR_HLA_STUDY', table_name='raw_hla_genotype')
functional_kir_genotype = data_sci_mgr.sql.read_table_into_data_frame(schema_name='KIR_HLA_STUDY', table_name='functional_kir_genotype')
partition = data_sci_mgr.sql.read_table_into_data_frame(schema_name='KIR_HLA_STUDY', table_name='validation_partition')

partition = partition.merge(raw_kir_genotype, how='left', on='public_id')
partition = partition.merge(raw_hla_genotype, how='left', on='public_id')
partition = partition.merge(functional_kir_genotype, how='left', on='public_id')

required_columns = ['public_id', 'subject_id', 'kir2dl1_t50', 'kir2dl2_t50', 'kir2dl3_t50', 'kir3dl1ex4_t50', 'kir3dl1ex9_t50', 
    'a_1', 'a_2', 'b_1', 'b_2', 'c_1', 'c_2', 'f_kir_score']

filename = 'Data/becca_twinsuk_analysis.csv'
beccas_data = pd.read_csv(filename)

merged_dataset = partition.merge(beccas_data, how='inner', left_on='public_id', right_on='PublicID')

date_time_str = data_sci_mgr.data_mgr.get_date_str()
filename = 'QC/chris_becca_ikir_score_cross_check_{}.csv'.format(date_time_str)
merged_dataset.to_csv(filename)

"""

"""

filename = '/Users/chrismbrooks/Documents/Imperial/Asquith Group/Raw Data/Twins UK/E1139_14092021/E1139_140921.xlsx'
cofactor_codes_df = pd.read_excel(filename, sheet_name='Coding', header=0)
cofactor_codes_df = cofactor_codes_df[cofactor_codes_df['PhenotypeID'] == 'P003135'].copy()
cofactor_codes_df = cofactor_codes_df[cofactor_codes_df['ResponseDescription'] == 'YC'].copy()
code = float(cofactor_codes_df['ResponseCode'].values[0])

filename = '/Users/chrismbrooks/Documents/Imperial/Asquith Group/Raw Data/Twins UK/E1139_14092021/E1139_140921.xlsx'
cofactors_df = pd.read_excel(filename, sheet_name='Phenotype_Cont', header=1)
cofactors_df = cofactors_df[cofactors_df['P003135'] == code].copy()

partition = data_sci_mgr.sql.read_table_into_data_frame(schema_name='KIR_HLA_STUDY', table_name='validation_partition')

partition = partition.merge(cofactors_df, how='inner', left_on='public_id', right_on='PublicID')
columns = ['public_id', 'validation_partition', 'P003135']

partition = partition[columns].copy()

print(partition)

"""
"""
datasource = '/Users/chrismbrooks/Documents/Imperial/Asquith Group/Raw Data/Twins UK/4_Trait_Analysis/4. Trait Analysis.xlsx'
raw_data = pd.read_excel(datasource)
rename_dict = {'Trait ID':'phenotype_id', 'Subset or Surface Protein':'marker_posession_def', 'Full Subset Name':'full_subset_name'}
raw_data = raw_data.rename(columns=rename_dict)
reference_data = raw_data[['phenotype_id', 'full_subset_name', 'marker_posession_def']].copy()
reference_data['relevant_parent_markers'] = []

punctuation = ['-', '+']
for idx, row in reference_data.iterrows():
    if row['full_subset_name'][-1] not in punctuation:
        row['full_subset_name'] = row[  'full_subset_name'] + '+'
        print(row['full_subset_name'])
    
    pieces = row['full_subset_name'].split('/')
    markers = []
    for piece in pieces:
        if piece != '*':
            markers.append(piece) 

    row['relevant_markers'] = pieces

print(reference_data)"""

"""filename = 'Data/phenotype_marker_reference_data.parquet'
reference_data = pd.read_parquet(filename)
targets = ['16+', '56+', '158a+']
candidates = []
for idx, row in reference_data.iterrows():
    has_markers = {key:False for key in targets}
    markers = row['relevant_markers']
    for key in targets:
        has_markers[key] = key in markers
        if not has_markers[key]:
            break
    
    has_everything = True
    for key in has_markers:
        if not has_markers[key]:
            has_everything = False
    
    if has_everything:
        candidates.append(row['phenotype_id'])

print(candidates)
"""

"""
sources = {'immunophenotype_assay': 'Data/immunophenotype_assay.csv',
            'ublic_mapping_vw':'Data/public_mapping_vw.csv',
            'raw_kir_genotype':'Data/raw_kir_genotype.csv',
            'functional_kir_genotype':'Data/functional_kir_genotype.csv',
            'raw_hla_genotype':'Data/raw_hla_genotype.csv',
            'validation_partition':'Data/validation_partition.csv'
}

for key in sources:
    table_name = key
    destination = sources[key]
    table_df = data_sci_mgr.sql.read_table_into_data_frame(schema_name='KIR_HLA_STUDY', table_name=table_name)
    table_df.to_csv(destination)
"""

"""
filename = 'Analysis/ElasticNet/en_feature_importances_31032023.csv'
importances = pd.read_csv(filename, index_col=0)

ols_results = data_sci_mgr.sql.read_table_into_data_frame(
    schema_name='KIR_HLA_STUDY', 
    table_name='model_result_ols'
)

pheno_reference_data = data_sci_mgr.sql.read_table_into_data_frame(
    schema_name='KIR_HLA_STUDY', 
    table_name='immunophenotype_definitions'
)

merged_dataset = importances.merge(ols_results, how='left', left_on='labels', right_on='feature_name')
merged_dataset = merged_dataset.merge(pheno_reference_data, how='left', left_on='labels', right_on='phenotype_id')

filename='/Users/chrismbrooks/Downloads/export2.csv'
merged_dataset.to_csv(filename)

print(merged_dataset)

"""

"""
filename = 'Analysis/General/candidate_feature_correlates_04042023.parquet'
data = pd.read_parquet(filename)

filename = 'Analysis/General/candidate_feature_correlates_04042023.csv'
data.to_csv(filename)
"""

"""data = data_sci_mgr.sql.read_table_into_data_frame(
    schema_name='KIR_HLA_STUDY', 
    table_name='immunophenotype_summary_stats'
)
filename = 'Data/immunophenotype_summary_stats.csv'
data.to_csv(filename)"""


"""
pheno_stats = data_sci_mgr.sql.read_table_into_data_frame(schema_name='KIR_HLA_STUDY', 
    table_name='immunophenotype_summary_stats'
)

filename = 'Data/unlike_phenos_05042023.csv'
unlike_phenos = pd.read_csv(filename, index_col=0)

unlike_phenos = unlike_phenos.merge(pheno_stats, how='left', left_on='0', right_on='measurement_id')

low_nan_phenos = unlike_phenos[unlike_phenos['nans_count'] <= 10]
low_nan_phenos = low_nan_phenos['measurement_id']
date_str = data_sci_mgr.data_mgr.get_date_str()
filename = 'Data/Subsets/clustered_and_restricted_to_phenos_with_less_thn_10_zeros_{}.csv'.format(date_str)
low_nan_phenos.to_csv(filename)
"""

"""

phenos_to_plot = ['MFI:469', 'P2:16111', 'P1:20054', 'P7 Mono:1287', 'P5 gd:523', 
'P4:3052', 'P1:10240', 'P7:31', 'P4:5263', 'P5 gd:393', 'P5 gd:1927', 
'P4:5287'
]

phenos_df_t = data_sci_mgr.data_mgr.outcomes(fill_na=False, fill_na_value=None, partition='training')
phenos_df_v = data_sci_mgr.data_mgr.outcomes(fill_na=False, fill_na_value=None, partition='validation')

scores_df_t = data_sci_mgr.data_mgr.features(fill_na=False, fill_na_value=None, partition='training')
scores_df_v = data_sci_mgr.data_mgr.features(fill_na=False, fill_na_value=None, partition='validation')

phenos_df = pd.concat([phenos_df_t, phenos_df_v])
scores_df = pd.concat([scores_df_t, scores_df_v])

scores_df['ikir_count'] = scores_df['kir2dl1'] + scores_df['kir2dl2'] + scores_df['kir2dl3'] + scores_df['kir3dl1']

data = scores_df.merge(phenos_df, how='inner', on='public_id')
data = data[phenos_to_plot].copy().corr(method='pearson')
sns.heatmap(data, annot = True)
plt.title('Correlation Heat Map')
filename = 'Analysis/Multivariate/heatmap_{}_{}.png'.format(phenos_to_plot[0], data_sci_mgr.data_mgr.get_date_str())
plt.savefig(filename)
plt.show()

"""
"""
filename = 'Analysis/Multivariate/07042023/multivar_qc_fs_bs_candidate_features_07042023.csv' 
subset = pd.read_csv(filename, index_col=0)
indeces = subset.values[:,1:3].sum(axis=1)
indeces = np.where(indeces >= 1)
subset1 = list(subset.iloc[indeces]['label'].values)

filename = 'Analysis/Multivariate/08042023/multivar_qc_fs_bs_candidate_features_08042023.csv' 
subset = pd.read_csv(filename, index_col=0)
indeces = subset.values[:,1:3].sum(axis=1)
indeces = np.where(indeces >= 1)
subset2 = list(subset.iloc[indeces]['label'].values)

filename = 'Analysis/Multivariate/10042023/multivar_qc_fs_bs_candidate_features_10042023.csv' 
subset = pd.read_csv(filename, index_col=0)
indeces = subset.values[:,1:3].sum(axis=1)
indeces = np.where(indeces >= 1)
subset3 = list(subset.iloc[indeces]['label'].values)

master_set = list(set(subset1).union(subset2).union(subset3))
master_df = pd.DataFrame(master_set, columns=['phenotype_id'])
master_df['07042023'] = 0
master_df['08042023'] = 0
master_df['10042023'] = 0

master_df.loc[master_df['phenotype_id'].isin(subset1), '07042023'] = 1
master_df.loc[master_df['phenotype_id'].isin(subset2), '08042023'] = 1
master_df.loc[master_df['phenotype_id'].isin(subset3), '10042023'] = 1

master_df['summation'] = master_df['07042023'] + master_df['08042023'] + master_df['10042023']

candidates = master_df.sort_values(by='summation', ascending=False)

pheno_ref_data = data_sci_mgr.sql.read_table_into_data_frame(schema_name='KIR_HLA_STUDY', table_name='immunophenotype_definitions')
ols_stats = data_sci_mgr.sql.read_table_into_data_frame(schema_name='KIR_HLA_STUDY', table_name='model_result_ols')
pheno_data_stats = data_sci_mgr.sql.read_table_into_data_frame(schema_name='KIR_HLA_STUDY', table_name='immunophenotype_summary_stats')
filename = 'Data/phenos_corr_dict_0.95_05042023.parquet'
correlates = pd.read_parquet(filename)

candidates = pheno_ref_data.merge(candidates, how='right', left_on='phenotype_id', right_on='phenotype_id')
candidates = candidates.merge(ols_stats, how='left', left_on='phenotype_id', right_on='feature_name')
candidates = candidates.merge(pheno_data_stats, how='left', left_on='phenotype_id', right_on='measurement_id')
candidates = candidates.merge(correlates, how='left', left_on='phenotype_id', right_on='label')

date_str = data_sci_mgr.data_mgr.get_date_str()
filename = 'Analysis/Multivariate/Summary/candidates_summary_{}.csv'.format(date_str)
candidates.to_csv(filename)
"""

"""
filename = 'Analysis/Multivariate/07042023/feature_importance_perm_rankings_07042023.csv'
rankings_1 = pd.read_csv(filename, index_col=0)
rankings_1['importance_std'] = rankings_1['importance_std']/0.17547016829603876

filename = 'Analysis/Multivariate/08042023/feature_importance_perm_rankings_08042023.csv'
rankings_2 = pd.read_csv(filename, index_col=0)
rankings_2['importance_std'] = rankings_2['importance_std']/0.17177343634105535

filename = 'Analysis/Multivariate/10042023/feature_importance_perm_rankings_10042023.csv'
rankings_3 = pd.read_csv(filename, index_col=0)
rankings_3['importance_std'] = rankings_3['importance_std']/0.17618841317727751

master_set = ['P1:20365', 'MFI:469', 'MFI:8',
'P7 Mono:1287', 'P4:5290', 'P1:10321']

master_df = pd.DataFrame(master_set, columns=['phenotype_id'])
master_df = master_df.merge(rankings_1, how='left', left_on='phenotype_id', right_on='feature')
master_df = master_df.merge(rankings_2, how='left', left_on='phenotype_id', right_on='feature')
master_df = master_df.merge(rankings_3, how='left', left_on='phenotype_id', right_on='feature')

master_df['summation'] = master_df['importance_mean'] + master_df['importance_mean_x'] + master_df['importance_mean_y']

master_df = master_df.sort_values(by='summation', ascending=False)

master_df['rank'] = np.arange(1, master_df.shape[0]+1, 1)

master_df = master_df[['phenotype_id', 'rank']].copy()

print(master_df)
"""
"""
pheno_stats = data_sci_mgr.sql.read_table_into_data_frame(schema_name='KIR_HLA_STUDY', 
    table_name='immunophenotype_summary_stats'
)

filename = 'Data/unlike_phenos_0.9_06042023.csv'
unlike_phenos = pd.read_csv(filename, index_col=0)

unlike_phenos = unlike_phenos.merge(pheno_stats, how='left', left_on='0', right_on='measurement_id')

low_nan_phenos = unlike_phenos[unlike_phenos['nans_count'] <= 10]
low_nan_phenos = low_nan_phenos['measurement_id']
date_str = data_sci_mgr.data_mgr.get_date_str()
filename = 'Data/Subsets/clustered_0.9_and_restricted_to_phenos_with_less_thn_10_zeros_{}.csv'.format(date_str)
low_nan_phenos.to_csv(filename)"""


"""
scores = data_sci_mgr.data_mgr.features(fill_na=False, partition='everything')
print(scores.shape)

sns.histplot(data=scores, x='f_kir_score')
plt.title('Functional iKir Score Distribution')
date_str = data_sci_mgr.data_mgr.get_date_str()
filename = 'Analysis/General/ikir_score_hist_{}.png'.format(date_str)
plt.savefig(filename)"""

"""scores = data_sci_mgr.data_mgr.features(fill_na=False, partition='everything')
print(set(list(scores['f_kir_score'].values)))"""

"""filename = 'Analysis/Multivariate/11042023_c_rc2/feature_importance_perm_rankings_11042023.csv'
perm_ranks_rc2 = pd.read_csv(filename, index_col=0)

filename = 'Analysis/Multivariate/11042023_c_rc3/feature_importance_perm_rankings_11042023.csv' 
perm_ranks_rc3 = pd.read_csv(filename, index_col=0)

perm_ranks = perm_ranks_rc2.merge(perm_ranks_rc3, how='outer', on='feature')

perm_ranks = perm_ranks.fillna(0)
perm_ranks['importance_mean_total'] = perm_ranks['importance_mean_x'] + perm_ranks['importance_mean_y']

perm_ranks = perm_ranks.sort_values('importance_mean_total', ascending=False)

perm_ranks['relative_importance'] =  perm_ranks['importance_mean_total'] / perm_ranks['importance_mean_total'].max()

pheno_ref_data = data_sci_mgr.sql.read_table_into_data_frame(schema_name='KIR_HLA_STUDY', table_name='immunophenotype_definitions')

candidates = perm_ranks.merge(pheno_ref_data, how='left', left_on='feature', right_on='phenotype_id')

date_str = data_sci_mgr.data_mgr.get_date_str()
filename = 'Analysis/Multivariate/Summary/candidates_summary_{}.csv'.format(date_str)
candidates.to_csv(filename)"""

"""pheno_stats = data_sci_mgr.sql.read_table_into_data_frame(schema_name='KIR_HLA_STUDY', 
    table_name='immunophenotype_summary_stats'
)

filename = 'Data/unlike_phenos_0.8_20042023.csv'
unlike_phenos = pd.read_csv(filename, index_col=0)

unlike_phenos = unlike_phenos.merge(pheno_stats, how='left', left_on='0', right_on='measurement_id')

low_nan_phenos = unlike_phenos[unlike_phenos['nans_count'] <= 10]
low_nan_phenos = low_nan_phenos['measurement_id']
date_str = data_sci_mgr.data_mgr.get_date_str()
filename = 'Data/Subsets/clustered_and_restricted_to_phenos_with_less_thn_10_zeros_{}.csv'.format(date_str)
low_nan_phenos.to_csv(filename)"""

"""ikir_labels = ['kir2dl1', 'kir2dl2_s', 'kir2dl2_w', 'kir2dl3', 'kir3dl1']
h_param_sets = []
for index in range(0, len(ikir_labels)):
    ikir_label = ikir_labels[index].upper()
    ikir_tag = 'f_{}'.format(ikir_labels[index])

    filename = 'Analysis/LogisticRegression/26042023/{}/lr_gs_scores_{}_26042023.csv'.format(ikir_label, ikir_tag)
    gs_scores = pd.read_csv(filename, index_col=0)
    cut_off = np.trunc(gs_scores['avg_neg_mae'].max()*100)/100

    gs_scores['mae_trnc'] = np.trunc(gs_scores['avg_neg_mae']*100)/100

    gs_scores = gs_scores[gs_scores['mae_trnc'] == cut_off].copy()
    candidates = gs_scores[gs_scores['C'] == gs_scores['C'].min()]

    h_params = {}
    h_params['label'] = ikir_labels[index]
    h_params['C'] = candidates.values[0, 0]
    h_params['l1_ratio'] = candidates.values[0,2]
    h_params['score'] = candidates.values[0,7]
    h_param_sets.append(h_params)

h_param_sets  = pd.DataFrame(h_param_sets)
print(h_param_sets)

filename = 'Analysis/LogisticRegression/lr_gs_candidate_h_params_26042023.csv'
h_param_sets.to_csv(filename)"""

"""pheno_definitions = data_sci_mgr.sql.read_table_into_data_frame(schema_name='KIR_HLA_STUDY', 
    table_name='immunophenotype_definitions'
)

source_filename = 'Analysis/RandomForest/20042023_c0.95_100/r_forest_fs_bs_candidate_features_100_20042023_2.csv'
candidates = pd.read_csv(source_filename, index_col=0)


candidates = candidates.merge(pheno_definitions, how='left', left_on='label', right_on='phenotype_id')
filename = '/Users/chrismbrooks/Downloads/output.csv'
candidates.to_csv(filename)"""

"""ikir_data = data_sci_mgr.sql.read_table_into_data_frame(schema_name='KIR_HLA_STUDY', 
    table_name='functional_kir_genotype'
)

columns = ['public_id', 'kir2dl1', 'kir2dl2', 'kir2dl3', 'kir3dl1', 'hla_c_c1',
       'hla_c_c2', 'hla_b_46_c1', 'hla_b_73_c1', 'hla_b_bw4', 'hla_a_bw4',
       'f_kir2dl1', 'f_kir2dl2_s', 'f_kir2dl2_w', 'f_kir2dl3', 'f_kir3dl1',
       'f_kir_count', 'f_kir_score']

ikir_data['kir_count'] = ikir_data['kir2dl1'] + ikir_data['kir2dl2'] + ikir_data['kir2dl3'] + ikir_data['kir3dl1']

columns = ['public_id', 'kir2dl1', 'kir2dl2', 'kir2dl3', 'kir3dl1', 'kir_count', 'hla_c_c1',
       'hla_c_c2', 'hla_b_46_c1', 'hla_b_73_c1', 'hla_b_bw4', 'hla_a_bw4',
       'f_kir2dl1', 'f_kir2dl2_s', 'f_kir2dl2_w', 'f_kir2dl3', 'f_kir3dl1',
       'f_kir_count', 'f_kir_score']

ikir_data = ikir_data[columns].copy()

data_sci_mgr.data_mgr.insert_df_to_sql_table(ikir_data, columns=columns, table='functional_kir_genotype2', schema='KIR_HLA_STUDY')

print(ikir_data)
print(ikir_data.columns)"""


ikir_data = data_sci_mgr.sql.read_table_into_data_frame(schema_name='KIR_HLA_STUDY', 
    table_name='functional_kir_genotype')

filename = 'Data/functional_kir_genotype.csv'
ikir_data.to_csv(filename)