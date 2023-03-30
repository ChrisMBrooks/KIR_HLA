import re

import pandas as pd

datasource = '/Users/chrismbrooks/Documents/Imperial/Asquith Group/Raw Data/Twins UK/4_Trait_Analysis/4. Trait Analysis.xlsx'
raw_data = pd.read_excel(datasource)
rename_dict = {'Trait ID':'phenotype_id', 'Subset or Surface Protein':'marker_posession_def', 'Full Subset Name':'full_subset_name'}
raw_data = raw_data.rename(columns=rename_dict)
reference_data = raw_data[['phenotype_id', 'full_subset_name', 'marker_posession_def']].copy()

pattern = '[a-zA-Z0-9]+[\+\-]'
marker_sets = []
for marker_posession_def in reference_data.values[:, 2]:
    matches = re.findall(pattern, marker_posession_def)
    marker_sets.append(matches)

reference_data['relevant_markers'] = marker_sets
print(reference_data)

filename = 'Data/phenotype_marker_reference_data.parquet'
reference_data.to_parquet(filename)

reference_data = pd.read_parquet(filename)
print(reference_data)