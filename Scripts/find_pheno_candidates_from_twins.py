import pandas as pd

filename = "/Users/chrismbrooks/Documents/Imperial/Asquith Group/Raw Data/Twins UK/4_Trait_Analysis/4. Trait Analysis.xlsx"
phenotype_defintiions =  pd.read_excel(filename)[['Trait ID', 'Subset or Surface Protein', 'Full Subset Name']].copy()

str_of_interest = '158a+'

relevants_traits = []
for index, row in phenotype_defintiions.iterrows():
    if row['Subset or Surface Protein'].find(str_of_interest) > 0 or row['Trait ID'].find(str_of_interest) > 0:
        relevants_traits.append(row['Trait ID'])

rel_traits_df = pd.DataFrame(relevants_traits, columns=['Trait ID'])

rel_traits_df.to_csv('Data/158a+_trait_ids.csv')