from Controllers.ConfigManager import ConfigManager as cm
from Controllers.MySQLManager import MySQLManager as msm

config = cm.ConfigManaager().config
sql = msm.MySQLManager(config=config)

ols_data = sql.read_table_into_data_frame(schema_name='KIR_HLA_STUDY', table_name='model_result_ols')
immunos_details = sql.read_table_into_data_frame(schema_name='KIR_HLA_STUDY', table_name='immunophenotype_definitions')

ols_data = ols_data.merge(immunos_details, left_on='feature_name', right_on='phenotype_id', how='left')

columns = ['feature_name', 'marker_definition', 'parent_population', 'beta_1', 'p_1']

ols_data = ols_data[columns]
filename = 'Data/ols_results_extract_15022023.csv'
ols_data.to_csv(filename)
