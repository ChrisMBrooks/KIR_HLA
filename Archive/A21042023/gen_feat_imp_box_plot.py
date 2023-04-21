
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

#Import Data
filename = 'Analysis/Multivariate/11042023_c_rc3/feature_importance_perm_rankings_11042023.csv'
importances = pd.read_csv(filename, index_col=0)

#Format Data
importances = importances[importances['importance_mean'] -0*importances['importance_std'] > 0].copy()


importances = importances.sort_values(by='importance_mean', ascending=False)
values_t = np.stack([
    np.flip(importances['importance_mean'].values.T), 
    np.flip(importances['importance_std'].values.T)], axis=0)
importances = pd.DataFrame(values_t, columns= reversed(list(importances['feature'])))
  
#Plot Box Plots
ax = importances.plot.box(vert=False, whis=10)
ax.set_title("Permutation Importances (training set)")
ax.axvline(x=0, color="k", linestyle="--")
ax.set_xlabel("Decrease in accuracy score")
ax.figure.tight_layout()

date_str = data_sci_mgr.data_mgr.get_date_str()
filename = "Analysis/Multivariate/feature_import_box_plot_{}.png".format(date_str, date_str)
plt.savefig(filename)