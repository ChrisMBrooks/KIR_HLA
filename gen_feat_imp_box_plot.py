
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme()

import pandas as pd
from Controllers.DataScienceManager import DataScienceManager as dsm

#Instantiate Controllers
use_full_dataset = False
use_database = False
data_sci_mgr = dsm.DataScienceManager(
    use_full_dataset=use_full_dataset, 
    use_database=use_database
)

#Declare Config Params
perm_type = 'train-validate'
source_filename = 'Analysis/Multivariate/feature_importance_perm_values_21042023.csv'
date_str = data_sci_mgr.data_mgr.get_date_str()
plot_filename = "Analysis/Multivariate/feature_import_box_plot_{}.png".format(date_str)

#Import Data
importances_df = pd.read_csv(source_filename, index_col=0)

#Format Data
importances_indeces = importances_df.values.mean(axis=0) -2*importances_df.values.std(axis=0) > 0
importances_indeces = np.where(importances_indeces == 1)[0]
columns = list(importances_df.columns)
columns = [columns[x] for x in importances_indeces]
importances = importances_df[columns].copy()

sorted_indeces = np.argsort(importances_df.values.mean(axis=0))
columns = list(importances_df.columns)
columns = [columns[x] for x in sorted_indeces]
importances_df = importances_df[columns].copy()
  
#Plot Box Plots
ax = importances.plot.box(vert=False, whis=1.5)
ax.set_title("Permutation Importances ({})".format(perm_type))
ax.axvline(x=0, color="k", linestyle="--")
ax.set_xlabel("Decrease in accuracy score")
ax.figure.tight_layout()

#Export Plot
plt.savefig(plot_filename)