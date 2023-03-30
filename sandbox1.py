
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme()

import pandas as pd
from Controllers.DataScienceManager import DataScienceManager as dsm

#Instantiate Controllers
use_full_dataset = False
data_sci_mgr = dsm.DataScienceManager(use_full_dataset=use_full_dataset)

"""
date_str = data_sci_mgr.data_mgr.get_date_str()
filename = "Analysis/General/candidate_feature_correlates_{}.parquet".format(date_str)
correlate_df = pd.read_parquet(filename)
correlates = correlate_df[['phenotype_id', 'correlates']].values

for index, label in enumerate(correlates[:, 0]):
    print(label)
    print(correlates[index, 1])
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

A = [1101, 2402, 1101, 2402, 101, 2402, 1101, 2402, 1101, 2402, 101, 2402, 201, 2402, 201, 2402, 
     201, 2301, 101, 2402, 101, 2402, 101, 2402, 201, 2402, 101, 2402, 3201, 1101, 3201, 201, 3201, 201]

A = list(set(A))
print(A)