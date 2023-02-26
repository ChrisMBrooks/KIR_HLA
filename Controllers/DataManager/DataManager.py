import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from Controllers.ConfigManager import ConfigManager as cm
from Controllers.DefinitionManager import DefinitionManager as dm
from Controllers.MySQLManager import MySQLManager as msm

class DataManager():
    def __init__(self, config, use_full_dataset:bool=False):
        self.config = config
        self.use_full_dataset = use_full_dataset
        self.setup()

    def setup(self):
        self.mms = MinMaxScaler(feature_range=(0, 1))
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
        self.data['immunophenotype_assay'] = self.sql.read_table_into_data_frame(
            schema_name=self.schema_name,
            table_name='immunophenotype_assay'
        )

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
            filename = "Data/trait_values_transpose.parquet"
            self.data['imm_phenotype_measurements'] = pd.read_parquet(filename) 
        else:
            filename = "Data/trait_values_transpose_short.csv"
            self.data['imm_phenotype_measurements'] = pd.read_csv(filename, index_col=0) 

    def features(self, fill_na = True, fill_na_value = 0.0, partition = 'training'):

        if partition == 'training':
            f_kir_geno_train_df = self.data['func_kir_genotype'].merge(
                self.data['training_partition'], on='public_id', how='inner'
            )
        elif partition == 'validation':
            f_kir_geno_train_df = self.data['func_kir_genotype'].merge(
                self.data['validation_partition'], on='public_id', how='inner'
            )
        else:
            f_kir_geno_train_df = self.data['func_kir_genotype']

        f_kir_geno_train_df.sort_values(by='public_id', inplace=True)

        if fill_na:
            f_kir_geno_train_df.fillna(fill_na_value, inplace=True)

        self.data['features'] = f_kir_geno_train_df

        return self.data['features']
    
    def feature_values(self, normalise = True, fill_na = True, fill_na_value = 0.0, 
        partition = 'training'
    ):

        self.features(fill_na, fill_na_value, partition)

        values_df = self.data['features']
        values = values_df['f_kir_score'].values.astype(float).reshape(-1,1)

        if normalise:
            values = self.normalise(values)
        return values

    def outcomes(self, fill_na = True, fill_na_value = 0.0, partition = 'training'):

        if partition == 'training':
            pheno_train_df = self.data['imm_phenotype_measurements'].merge(
                self.data['training_partition'], on='subject_id', how='inner'
            )
        elif partition == 'validation':
            pheno_train_df = self.data['imm_phenotype_measurements'].merge(
                self.data['validation_partition'], on='subject_id', how='inner'
            )
        else:
            pheno_train_df = self.data['imm_phenotype_measurements'].merge(
                self.data['partitions'], on='subject_id', how='inner'
            )

        pheno_train_df.sort_values(by='public_id', inplace=True)

        if fill_na:
            pheno_train_df.fillna(fill_na_value, inplace=True)

        self.data['outcomes'] = pheno_train_df

        return  self.data['outcomes']
    
    def outcome_values(self, normalise = True, fill_na = True, fill_na_value = 0.0, 
        partition = 'training'
    ):

        self.outcomes(fill_na, fill_na_value, partition)

        values_df = self.data['outcomes']
        values = values_df.values[:,1:-2].astype(float)

        if normalise:
            values = self.normalise(values)
        
        return values
    
    def outcomes_by_class(self,assay_class:str):
        immuno_classes = self.data['immunophenotype_assay']
        immuno_subset = immuno_classes[immuno_classes['measurement_type'] == 
            assay_class]['measurement_id']
        immuno_subset = list(immuno_subset.values)

        return self.outcomes_subset(desired_columns=immuno_subset)

    def outcomes_subset(self, desired_columns:list, partition = 'training'):
        required_columns = ['subject_id'] + desired_columns + ['public_id', 'validation_partition']
        outcomes = self.outcomes(fill_na = False, partition = partition)
        return outcomes[required_columns].copy()

    def insert_df_to_sql_table(self, df:pd.DataFrame, columns:list, schema:str, table:str, 
        use_batches:bool=True, batch_size:int=10000
    ):
        if not use_batches:
            batch_size = df.shape[0]+1
        
        records_to_insert = []
        for index, row in df.iterrows():
            record = [row[column] for column in columns]
            records_to_insert.append(record)
        
            if len(records_to_insert) >= batch_size:
                self.sql.insert_records(schema_name=schema, table_name=table, 
                    column_names=columns, values = records_to_insert)
                records_to_insert = []

        if len(records_to_insert) > 0:
            self.sql.insert_records(schema_name=schema, table_name=table, 
                column_names=columns, values = records_to_insert)
        
    def normalise(self, values):
        return self.mms.fit_transform(values)
