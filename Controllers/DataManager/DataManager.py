import pandas as pd
import numpy as np
import datetime

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import RepeatedKFold

from Controllers.ConfigManager import ConfigManager as cm
from Controllers.DefinitionManager import DefinitionManager as dm
from Controllers.MySQLManager import MySQLManager as msm

class DataManager():
    def __init__(self, config, use_full_dataset:bool=False, use_db:bool=True):
        self.config = config
        self.use_full_dataset = use_full_dataset
        self.use_db = use_db
        self.setup()

    def setup(self):
        self.sql = msm.MySQLManager(config=self.config)
        self.datasources = self.config['setup']['datasources']
        self.schema_name = "KIR_HLA_STUDY"

        self.datasources = self.config['setup']['datasources']
        self.table_names = ["raw_hla_genotype", 
            "raw_kir_genotype", 
            "raw_public_mapping"
        ]
        if self.use_db:
            self.pull_data_from_db()
        else:
            self.pull_data_from_files()

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
            table_name="raw_kir_genotype"
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

    def pull_data_from_files(self):
        self.data = dict()

        sources = {'immunophenotype_assay': 'Data/immunophenotype_assay.csv',
            'public_mapping':'Data/public_mapping_vw.csv',
            'kir_genotype':'Data/raw_kir_genotype.csv',
            'func_kir_genotype':'Data/functional_kir_genotype.csv',
            'hla_genotype':'Data/raw_hla_genotype.csv',
            'partitions':'Data/validation_partition.csv'
        }
        # Pull Data from DB
        for key in sources:
            source = sources[key]
            self.data[key] = pd.read_csv(source, index_col=0)

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
            f_kir_geno_train_df = self.data['func_kir_genotype'].merge(
                self.data['partitions'], on='public_id', how='inner'
            )

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

    def partition_training_data(self, phenos:pd.DataFrame, scores:pd.DataFrame, n_splits:float, random_state:int):
        cv = RepeatedKFold(n_splits=n_splits, n_repeats=1, random_state=random_state)
        splits_gen = cv.split(phenos)
        split = next(splits_gen)
        train_indeces = split[0]
        test_indeces = split[1]

        phenos_df_t = phenos.iloc[train_indeces, :]
        scores_df_t = scores.iloc[train_indeces, :]

        phenos_df_v = phenos.iloc[test_indeces, :]
        scores_df_v = scores.iloc[test_indeces, :]
        return phenos_df_t, scores_df_t, phenos_df_v, scores_df_v

    def reshape(self, phenos:pd.DataFrame, scores:pd.DataFrame):
        scores = scores['f_kir_score'].values.reshape(-1,1)
        if len(phenos.values.shape) == 1:
            phenos = phenos.values.reshape(-1,1)
        else:
            phenos = phenos.values[:, 0:]
        
        return phenos, scores
    
    def preprocess_data(self, X, Y, impute = True, standardise = True, normalise = True, strategy='mean'):
        if impute:
            X, Y = self.impute_missing_values(X, Y, strategy)
        if standardise:
            X, Y = self.standardise(X, Y)
        if normalise:
            X, Y = self.normalise(X, Y, min=0, max=1) # Makes everything positive. 
        
        return X,Y

    def impute_missing_values(self, X, Y, strategy):
        if strategy.lower() == 'knn':
            imputer = KNNImputer(
                missing_values=np.nan, 
                n_neighbors=2, 
                weights='uniform'
            )
        else:
            imputer = SimpleImputer(
                missing_values=np.nan, 
                strategy=strategy
            )

        X = imputer.fit_transform(X)
        Y = imputer.fit_transform(Y)
        return X, Y

    def standardise(self, X,Y):
        x_std_sclr = StandardScaler()
        y_std_sclr = StandardScaler()

        X = x_std_sclr.fit_transform(X)
        Y = y_std_sclr.fit_transform(Y)
        return X, Y

    def normalise(self, X, Y, min = 0, max = 1):
        x_mms = MinMaxScaler(feature_range=(min, max))
        y_mms = MinMaxScaler(feature_range=(min, max))

        X = x_mms.fit_transform(X)

        # MMS scales and translates each feature individually
        Y = y_mms.fit_transform(Y) 

        return X, Y

    def preprocess_data_v(self, X_t, Y_t, X_v, Y_v, impute = True, standardise = True, normalise = True, strategy='mean'):
        if impute:
            X_t, Y_t, X_v, Y_v, = self.impute_missing_values_v(X_t, Y_t, X_v, Y_v, strategy)
        if standardise:
            X_t, Y_t, X_v, Y_v, = self.standardise_v(X_t, Y_t, X_v, Y_v,)
        if normalise:
            X_t, Y_t, X_v, Y_v,= self.normalise_v(X_t, Y_t, X_v, Y_v, min=0, max=1) # Makes everything positive. 
        
        return X_t, Y_t, X_v, Y_v

    def impute_missing_values_v(self, X_t, Y_t, X_v, Y_v, strategy):
        if strategy.lower() == 'knn':
            imputer = KNNImputer(
                missing_values=np.nan, 
                n_neighbors=2, 
                weights='uniform'
            )
        else:
            imputer = SimpleImputer(
                missing_values=np.nan, 
                strategy=strategy
            )

        imputer = imputer.fit(X_t)
        X_t = imputer.transform(X_t)
        X_v = imputer.transform(X_v)

        imputer = imputer.fit(Y_t)
        Y_t = imputer.transform(Y_t)
        Y_v = imputer.transform(Y_v)
        return X_t, Y_t, X_v, Y_v,

    def standardise_v(self, X_t, Y_t, X_v, Y_v,):
        scaler = StandardScaler()

        scaler = scaler.fit(X_t)
        X_t = scaler.transform(X_t)
        X_v = scaler.transform(X_v)

        scaler = scaler.fit(Y_t)
        Y_t = scaler.transform(Y_t)
        Y_v = scaler.transform(Y_v)
        return X_t, Y_t, X_v, Y_v

    def normalise_v(self, X_t, Y_t, X_v, Y_v, min = 0, max = 1):
        scaler = MinMaxScaler(feature_range=(min, max))

        scaler = scaler.fit(X_t)
        X_t = scaler.transform(X_t)
        X_v = scaler.transform(X_v)

        scaler = scaler.fit(Y_t)
        Y_t = scaler.transform(Y_t)
        Y_v = scaler.transform(Y_v)

        return X_t, Y_t, X_v, Y_v

    def get_date_str(self):
        current_date = datetime.datetime.now().strftime("%d%m%Y")
        return current_date