import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from Controllers.ConfigManager import ConfigManager as cm
from Controllers.MySQLManager import MySQLManager as msm
from Controllers.DataManager import DataManager as dtm
from Controllers.LearningManager import LearningManager as lrn

def retrieve_subset_of_data(partition = 'training', use_full_dataset = True):
    # Get iKIR Scores
    ikir_scores = data_mgr.feature_values(normalise = False, fill_na = False, 
        fill_na_value = 0.0, partition = partition
    )

    # Read in List of Desired Phenos
    filename = "Data/candidate_phenos_qc_09022023.csv"
    phenos_subset = pd.read_csv(filename, index_col=0)
    phenos_subset['filter_criteria'] = phenos_subset['forward_selected'] & phenos_subset['backward_selected']

    phenos_subset = phenos_subset[phenos_subset['filter_criteria'] == True] 

    phenos_subset = list(phenos_subset['label'].values)

    # Filter Master Dataset on Desired Subset
    if use_full_dataset: 
        immunos_maxtrix_subset = data_mgr.outcomes_subset(desired_columns=phenos_subset, 
            partition = partition
        )
    else:
        # Logic to Handle When the Small Traits file is used. 
        pheno_labels_small = data_mgr.outcomes(fill_na=False, partition='everything').columns[1:-2]
        phenos_subset_overlap = np.intersect1d(pheno_labels_small, phenos_subset)
        immunos_maxtrix_subset = data_mgr.outcomes_subset(desired_columns=phenos_subset_overlap, 
            partition = partition
        )

    immunos_labels = immunos_maxtrix_subset.columns[1:-2]
    immunos_maxtrix_subset = immunos_maxtrix_subset.values[:,1:-2].astype(float)
    return ikir_scores, immunos_labels, immunos_maxtrix_subset

#Instantiate Controllers
use_full_dataset = True
config = cm.ConfigManaager().config
sql = msm.MySQLManager(config=config)
data_mgr = dtm.DataManager(config=config, use_full_dataset=use_full_dataset)
lrn_mgr = lrn.LearningManager(config=config)

# Pull Data from DB
ikir_scores_t, immunos_labels_t, immunos_maxtrix_subset_t = \
    retrieve_subset_of_data(partition = 'training', use_full_dataset=use_full_dataset)

# Prep Data
df = pd.DataFrame(immunos_maxtrix_subset_t, columns=immunos_labels_t)
df = df.reindex(sorted(df.columns), axis=1)
immunos_labels_t = df.columns

# calculate the correlation matrix
corr = df.corr()

# plot the heatmap
plt.figure() 
sns.heatmap(corr, cmap="Blues", annot=True, annot_kws={"fontsize":4}, xticklabels=immunos_labels_t, yticklabels=immunos_labels_t)

plt.show()

#https://numpy.org/doc/stable/reference/generated/numpy.cov.html
#https://stackabuse.com/covariance-and-correlation-in-python/
#https://stackoverflow.com/questions/39409866/correlation-heatmap
#https://stackoverflow.com/questions/43335973/how-to-generate-high-resolution-heatmap-using-seaborn