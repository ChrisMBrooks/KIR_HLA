import pandas as pd
import numpy as np

from Controllers.ConfigManager import ConfigManager as cm
from Controllers.DefinitionManager import DefinitionManager as dm
from Controllers.MySQLManager import MySQLManager as msm

class DataManager():
    def __init__(self, config, use_full_dataset:bool=False):
        self.config = config
        self.use_full_dataset = use_full_dataset
        self.setup()

    def setup(self):
        self.sql = msm.MySQLManager(config=self.config)
        self.datasources = self.config['setup']['datasources']
        self.schema_name = self.datasources[0]["schema_name"]

        self.datasources = self.config['setup']['datasources']
        self.table_names = [self.datasources[0]["table_name"], 
            self.datasources[1]["table_name"], 
            self.datasources[2]["table_name"]
        ]

        self.pull_data_from_db()

    def pull_data_from_db(self):
        self.data = dict()
        # Pull Data from DB
        A = self.sql.read_table_into_data_frame(
            schema_name=self.schema_name,
            table_name='immunophenotype_assay'
        ).values[:,1:]
        self.data['imm_phenotype_assay'] = [x[0] for x in A]

        self.data['public_mapping'] = self.sql.read_table_into_data_frame(
            schema_name=self.schema_name,
            table_name='public_mapping_vw'
        )

        self.data['kir_genotype'] = self.sql.read_table_into_data_frame(
            schema_name=self.schema_name,
            table_name=self.table_names[1]
        )

        self.data['func_kir_genotype'] = self.sql.read_table_into_data_frame(
            schema_name=self.schema_name,
            table_name='functional_kir_genotype'
        )

        self.data['hla_genotype'] = self.sql.read_table_into_data_frame(
            schema_name=self.schema_name, 
            table_name='raw_hla_genotype'
        )

        self.data['partitions'] = self.sql.read_table_into_data_frame(
            schema_name=self.schema_name, 
            table_name='validation_partition'
        )

        self.data['training_partition'] =  self.data['partitions'][
             self.data['partitions']['validation_partition'] == 'TRAINING'
        ]

        self.data['validation_partition'] =  self.data['partitions'][
             self.data['partitions']['validation_partition'] == 'VALIDATION'
        ]
        
        if self.use_full_dataset:
            filename = "Data/trait_values_transpose.csv"
        else:
            filename = "Data/trait_values_transpose_short.csv"
        self.data['imm_phenotype_measurements'] = pd.read_csv(filename, index_col=0) 

    def features(self, fill_na = 0.0):
        A = self.data['training_partition']
        B = self.data['kir_genotype']
        C = self.data['hla_indicator']

        X = A.merge(B, 
            on='public_id', how='inner').merge(
                C, on='public_id', 
                how='inner'
        )

        X.sort_values(by=['public_id'], inplace=True)
        X.fillna(value=fill_na, inplace=True)
        self.X = X
        return self.X
    
    def features_lookup(self, coefficients:pd.DataFrame):
        named_coefficients = None

        if self.X is not None and not coefficients.empty:
            feature_names = [[int(i), self.X.columns[int(i)+3]] for i in coefficients['feature_id'].values]
            feature_names_df = pd.DataFrame(feature_names, columns=['feature_id', 'feature_name'])
            
            named_coefficients = coefficients.merge(feature_names_df, 
            on='feature_id', how='inner')

        return named_coefficients

    def feature_values(self, fill_na = 0.0):
        return self.features(fill_na = fill_na).values[:,2:].astype(np.float)
    
    def outcomes(self, measurement_id:str, fill_na = 0.0):
        where_clause = "WHERE measurement_id = '{}'".format(measurement_id)
        
        #Get subset of measurements
        Y = self.sql.read_table_into_data_frame(schema_name=self.schema_name, 
            table_name='train_immunophenotype_measurement', 
            where_clause = where_clause
        )
        
        #Get Merge & Sort
        Y = self.data['public_mapping'].merge(
            Y, on='subject_id', how='inner'
        )
        
        Y.sort_values(by=['public_id'], inplace=True)
        Y.fillna(value=fill_na, inplace=True)
        self.Y = Y
        return self.Y

    def outcome_values(self, measurement_id:str, fill_na = 0.0):
        Y = self.outcomes(measurement_id=measurement_id, 
            fill_na=fill_na).values[:,3:]

        Y = np.array([y[0] for y in Y], dtype=np.float)

        return Y

    def record_el_net_results(self, data_frame:pd.DataFrame):
        table_name = 'model_result_elastic_net_primary_coeff'

        column_names = ['coeff', 'feature_name', 'relevance_cut_off', 'predictor_id', 'alpha',
            'l1_ratio', 'measured_abs_error', 'grid_search_alpha_range',
            'grid_search_l1_ratio_range', 'cross_val_n_splits',
            'cross_val_n_repeats', 'run_id'
        ]
        
        valid_values = []
        for index, row in data_frame.iterrows():
            record = [row[col_name] for col_name in column_names]
            valid_values.append(record)

        self.sql.insert_records(schema_name = self.schema_name, 
            table_name=table_name, column_names=column_names, 
            values=valid_values)
    
    def record_lasso_results(self, data_frame:pd.DataFrame):
        table_name = 'model_result_lasso_primary_coeff'

        column_names = ['coeff', 'feature_name', 'relevance_cut_off', 'predictor_id', 'alpha',
            'l1_ratio', 'measured_abs_error', 'grid_search_alpha_range',
            'grid_search_l1_ratio_range', 'cross_val_n_splits',
            'cross_val_n_repeats', 'run_id'
        ]
        
        valid_values = []
        for index, row in data_frame.iterrows():
            record = [row[col_name] for col_name in column_names]
            valid_values.append(record)

        self.sql.insert_records(schema_name = self.schema_name, 
            table_name=table_name, column_names=column_names, 
            values=valid_values)