
from Controllers.ConfigManager import ConfigManager as cm
from Controllers.MySQLManager import MySQLManager as msm
from Controllers.DataManager import DataManager as dtm
from Controllers.LearningManager import LearningManager as lrn

class DataScienceManager():
    def __init__(self, use_full_dataset=True):
        self.use_full_dataset = use_full_dataset
        self.config = cm.ConfigManaager().config
        self.sql = msm.MySQLManager(config=self.config)
        self.data_mgr = dtm.DataManager(config=self.config, use_full_dataset=use_full_dataset)
        self.lrn_mgr = lrn.LearningManager(config=self.config)