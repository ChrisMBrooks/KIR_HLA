
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme()

from Controllers.DataScienceManager import DataScienceManager as dsm

use_full_dataset = False
data_sci_mgr = dsm.DataScienceManager(use_full_dataset=use_full_dataset)

summary_stats = data_sci_mgr.sql.read_table_into_data_frame(schema_name='KIR_HLA_STUDY', 
    table_name='immunophenotype_summary_stats'
)

sns.histplot(data=summary_stats, x="nans_count", cumulative=True)
plt.title('Cumulative Histogram of NaN Count')
plt.xlabel('Immunophenotypes Grouped by NaN Count')

#plt.show()
date_str = data_sci_mgr.data_mgr.get_date_str()
filename = 'Analysis/Data Characterisation/immunos_nans_count_hist_{}.png'.format(date_str)
plt.savefig(filename)



