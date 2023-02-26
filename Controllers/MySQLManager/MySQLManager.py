from mysql.connector import connect, Error
import pandas as pd

class MySQLManager:
    def __init__(self, config):
        self.config = config["my_sql"]
        pass

    def __build_connection(self):
        connection = connect(
                host=self.config["host"],
                user=self.config["user"],
                password=self.config["pwd"]
        )
        return connection

    def execute_query(self, query:str):
        try:
            cxn = self.__build_connection()
            with cxn.cursor() as cursor:
                cursor.execute(query)
            cxn.commit()
        except Error as e:
            print(e)

        finally:
            if cxn: 
                cxn.close()

    def create_table(self, schema_name:str, table_name:str, definition:dict):
        try:
            cxn = self.__build_connection()
            create_db_query = self.get_create_table_str(schema_name, table_name, definition)
            with cxn.cursor() as cursor:
                cursor.execute(create_db_query)

        except Error as e:
            print(e)

        finally:
            if cxn: 
                cxn.close()
        
    def create_mesurements_table(self):
        query = """ CREATE TABLE KIR_HLA_STUDY.raw_immunophenotype_measurement (
        subject_id INTEGER NOT NULL,
        measurement_id VARCHAR(15) NOT NULL,
        measurement_value DECIMAL(20,10),
        PRIMARY KEY (subject_id, measurement_id)
        ) PARTITION BY HASH(subject_id)
        PARTITIONS 100; """

        self.execute_query(query=query)

    def create_partition_table(self):
        query = """ CREATE TABLE KIR_HLA_STUDY.validation_partition (
                        public_id VARCHAR(15) NOT NULL,
                        validation_partition VARCHAR(10) NOT NULL CHECK (
                            validation_partition IN ("TRAINING","VALIDATION"
                        ),
                        subject_id INTEGER NOT NULL
                    )
            );"""

        self.execute_query(query=query)
    
    def get_create_table_str(self, schema_name:str, table_name:str, definition:dict):
        create_table_str = ""
        for key in definition:
            create_table_str = "{}, {} {}".format(create_table_str, key,definition[key])

        create_table_str = "CREATE TABLE {}.{} ({})".format(schema_name, 
            table_name, create_table_str.strip(',')
        )
        return create_table_str
    
    def insert_records(self, schema_name:str, table_name:str, column_names:list, values:list):
        try:
            cxn = self.__build_connection()
            create_db_query = self.get_insert_records_str(schema_name, table_name, 
                column_names, values
            )
            with cxn.cursor() as cursor:
                cursor.executemany(create_db_query, values)
            cxn.commit()

        except Error as e:
            print(e)

        finally:
            if cxn: 
                cxn.close()
    
    def get_insert_records_str(self, schema_name:str, table_name:str, 
        column_names:list, values:list
    ):
        query_param_str = ', '.join(['%s' for x in values[0]])
        column_name_str = ', '.join(column_names)

        insert_str = "INSERT INTO {}.{} ({}) VALUES ({})".format(schema_name, table_name, 
            column_name_str, query_param_str
        )
            
        return insert_str
    
    def read_table_into_data_frame(self, schema_name:str, table_name:str, where_clause = ''):
        df = pd.DataFrame()
        try:
            cxn = self.__build_connection()
            sql_query_str = 'SELECT * FROM {}.{}'.format(schema_name, table_name)

            if where_clause != '':
                sql_query_str = '{} {}'.format(sql_query_str, where_clause)
            
            df = pd.read_sql_query(sql_query_str, cxn)
            cxn.close()

        except Error as e:
            print(e)

        finally:
            if cxn: 
                cxn.close()
            return df
    
    def record_count(self, schema_name:str, table_name:str):
        sql_query_str = """SELECT COUNT(*) FROM {}.{}""".format(schema_name, table_name)
        try:
            cxn = self.__build_connection()
            record_count = None
            with cxn.cursor() as cursor:
                record_count = int(cursor.execute(sql_query_str))

        except Error as e:
            print(e)

        finally:
            if cxn: 
                cxn.close()
            return record_count