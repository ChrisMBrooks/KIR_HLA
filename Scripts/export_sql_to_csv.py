import pandas as pd
from datetime import date

from Controllers.ConfigManager import ConfigManager as cm
from Controllers.MySQLManager import MySQLManager as msm

use_full_dataset = True
config = cm.ConfigManaager().config
sql = msm.MySQLManager(config=config)

partitions_df = sql.read_table_into_data_frame(
    schema_name='KIR_HLA_STUDY',
    table_name='model_result_ols'
)

print(partitions_df)

date_str = date.today().strftime("%d%m%Y")
filename = "Data/ols_results_extract_{}.csv".format(date_str)
partitions_df.to_csv(filename)

partitions_df = pd.read_csv(filename, index_col=0)

print(partitions_df)
