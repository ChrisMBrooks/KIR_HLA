import os, sys, logging
import pandas as pd
import time, math
import concurrent.futures
import itertools as it

# evaluate an elastic net model on the dataset

from Controllers.ConfigManager import ConfigManager as cm
from Controllers.DataManager import DataManager as dtm
from Controllers.LearningManager import LearningManager as lm

def setup_logging(enable:bool):
    if enable:
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

        file_handler = logging.FileHandler('logs/debug_logs.txt')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)

        sys.stdout = open('logs/print_logs.txt', 'w')

def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    args = [iter(iterable)] * n
    return it.zip_longest(fillvalue=fillvalue, *args)

def try_tasks(group):
    for item in group:
        try:
            item[0](item[1], item[2])
        except Exception as e:
            print(e)

def en_modelling_task(X:pd.DataFrame, assay_name):

    # Retrieve & Shape Data
    Y = data_mgr.outcome_values(measurement_id=assay_name, fill_na = 0.0)

    # Fit the Model
    primary_coefficients_df = learn_mgr.ElasticNetCoefficientAnalysis(
        x=X, y=Y, predictor_id=assay_name
    )
    if not primary_coefficients_df.empty: 
        primary_coefficients_df = data_mgr.features_lookup(primary_coefficients_df)
        primary_coefficients_df.drop(['feature_id'], axis='columns', inplace=True)

        # Write Data to Output Table
        data_mgr.record_el_net_results(primary_coefficients_df)

def lasso_modelling_task(X:pd.DataFrame, assay_name):

    # Retrieve & Shape Data
    Y = data_mgr.outcome_values(measurement_id=assay_name, fill_na = 0.0)

    # Fit the Model
    primary_coefficients_df = learn_mgr.LassoCoefficientAnalysis(
        x=X, y=Y, predictor_id=assay_name
    )
    if not primary_coefficients_df.empty: 
        primary_coefficients_df = data_mgr.features_lookup(primary_coefficients_df)
        primary_coefficients_df.drop(['feature_id'], axis='columns', inplace=True)

        # Write Data to Output Table
        data_mgr.record_lasso_results(primary_coefficients_df)

def grouped_processes_procedure(task_name, X:pd.DataFrame, n:int):

    m = 10
    x_assay_list = [(task_name, X, assay) for assay in assay_list] 
    executor = concurrent.futures.ProcessPoolExecutor(m)
    futures = [executor.submit(try_tasks, group) 
            for group in grouper(x_assay_list, math.ceil(n/m))]
    
    # Wait here for all processes to complete prior to proceeding
    concurrent.futures.wait(futures)

setup_logging(False)
#Instantiate Controllers
config = cm.ConfigManaager().config
data_mgr = dtm.DataManager(config=config)
learn_mgr = lm.LearningManager(config=config)
assay_list =  data_mgr.data['imm_phenotype_assay'][0:10]
X = data_mgr.feature_values(fill_na = 0.0)

grouped_processes_procedure(
    lasso_modelling_task, X, len(assay_list)
)

#Open Quesstions ... 
# which variables contributed the most? 
# May want to look into MultiTaskElasticNet, RandomForest