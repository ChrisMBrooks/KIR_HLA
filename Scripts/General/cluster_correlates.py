import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from collections import defaultdict

from Controllers.DataScienceManager import DataScienceManager as dsm

print('Starting')

#Instantiate Controllers
use_full_dataset = True
use_database = False
data_sci_mgr = dsm.DataScienceManager(
    use_full_dataset=use_full_dataset, 
    use_database=use_database
)

phenos = data_sci_mgr.data_mgr.outcomes(fill_na=False, fill_na_value=None, partition='training')
phenos_subset = list(phenos.columns[1:-2])
phenos = phenos[phenos_subset]

#fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
corr = phenos.corr(method='pearson').values
corr = np.nan_to_num(corr)


# Ensure the correlation matrix is symmetric
corr = (corr + corr.T) / 2
np.fill_diagonal(corr, 1)

# We convert the correlation matrix to a distance matrix before performing
# hierarchical clustering using Ward's linkage.
distance_matrix = 1 - np.abs(corr)



distance_matrix = squareform(distance_matrix)

print('dist matrix 2: ', distance_matrix.shape)

dist_linkage = hierarchy.ward(distance_matrix)
dist_linkage = np.clip(dist_linkage, a_min=0)

"""
dendro = hierarchy.dendrogram(
    dist_linkage, labels=phenos_subset, ax=ax1, leaf_rotation=90
)
dendro_idx = np.arange(0, len(dendro["ivl"]))

ax2.imshow(corr[dendro["leaves"], :][:, dendro["leaves"]])
ax2.set_xticks(dendro_idx)
ax2.set_yticks(dendro_idx)
ax2.set_xticklabels(dendro["ivl"], rotation="vertical")
ax2.set_yticklabels(dendro["ivl"])
fig.tight_layout()
date_str = data_sci_mgr.data_mgr.get_date_str()
filename = 'Analysis/General/pearson_clustering_dendo_diagram_{}.png'.format(date_str)
plt.savefig(filename,dpi=1200)
"""

cluster_ids = hierarchy.fcluster(dist_linkage, 0.001, criterion="distance")
cluster_id_to_feature_ids = defaultdict(list)
for idx, cluster_id in enumerate(cluster_ids):
    cluster_id_to_feature_ids[cluster_id].append(idx)
selected_features = \
    [[phenos_subset[v[0]], [phenos_subset[x] for x in v]] for v in cluster_id_to_feature_ids.values()]

selected_features = pd.DataFrame(selected_features, columns=['representative_features', 'cohort'])

date_str = data_sci_mgr.data_mgr.get_date_str()
filename = 'representative_features_after_clustering_{}.parquet'.format(date_str)

selected_features.to_parquet(filename)

print('Initial Size: ', len(phenos_subset))
print('Final Size: ', selected_features.shape)
print('Complete')