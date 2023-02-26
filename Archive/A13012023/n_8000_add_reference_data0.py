import pandas as pd

filename = 'Analysis/n_80000_signif_immunophenotypes.csv'
signif_phenos = pd.read_csv(filename, index_col=0)

filename = '/Users/chrismbrooks/Documents/Imperial/Asquith Group/Raw Data/Twins UK/4_Trait_Analysis/4. Trait Analysis.xlsx'
twins_trait_analysis = pd.read_excel(filename, usecols="A,D,E")

signif_phenos = signif_phenos.merge(
    twins_trait_analysis, left_on='feature_name', right_on='Trait ID', how='left')

features = ['const', '2DL1', '2DL2s', '2DL2w', '2DL3', '3DL1']

signif_phenos['max_nlog_p'] = signif_phenos[['nlog_p_1','nlog_p_2','nlog_p_3','nlog_p_4','nlog_p_5']].max(axis=1) 

signif_phenos.sort_values('max_nlog_p', 
    ascending=False, inplace=True)

signif_phenos.to_csv('Analysis/n_80000_signif_phenos_w_defs.csv')

"""
Do we include the intercept? Whats the story with the high significance? 
What do we do about the sparcely populated immunophenotypes? 
Do we reprocess but without the zero measurement records? 
Is there a recommended reference database that defines cell type based on CDs?
What does the sparsity mean for the reverse regression strategy? 

Actions ... 
repeat with iKIR score, not that there are both 0.75 and 0.5 weights, need to consult paper ... 
repeat analysis but filter out sparsely populated datasets with 
consult Twins Paper to discern how frequency and cut-off of flow cytof were evaluated. Laura mentioned they focused ona a subset of phenotypes ... 140?

"""