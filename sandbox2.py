import os
import math, random, itertools
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme()

import pandas as pd
from Controllers.DataScienceManager import DataScienceManager as dsm

filename = 'Analysis/ElasticNet/April/11042023_rc/grid_search_results_11042023.csv'
df = pd.read_csv(filename, index_col=0)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the data
ax.scatter(df['alpha'], df['l1_ratio'], df['mae'], c=df['mae'], cmap='viridis')

# Set labels and title
ax.set_xlabel('Alpha')
ax.set_ylabel('L1 Ratio')
ax.set_zlabel('MAE',  labelpad=15)
ax.set_title('3D Plot: Alpha vs L1 Ratio vs MAE')

# Add a color bar
cbar = fig.colorbar(ax.scatter(df['alpha'], df['l1_ratio'], df['mae'], c=df['mae'], cmap='viridis'), pad=.05, shrink=0.6)
cbar.set_label('MAE')

fig.tight_layout()

# Show the plot
plt.show()

plt.clf()


filename = 'Analysis/RandomForest/May/26052023/Test7/rf_2r_gs_results_7_26052023.csv'

df = pd.read_csv(filename, index_col=0)

df = df[df['max_depth'] <=10].copy()
df = df[df['n_estimators'] >=40].copy()

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the data
ax.scatter(df['n_estimators'], df['max_depth'], df['mean_neg_mae'], c=df['mean_neg_mae'], cmap='viridis')

# Set labels and title
ax.set_xlabel('Num Estimators')
ax.set_ylabel('Max Depth')
ax.set_zlabel('MAE',  labelpad=15)
ax.set_title('3D Plot: Num Estimators vs Max Depth vs MAE')

# Add a color bar
cbar = fig.colorbar(ax.scatter(df['n_estimators'], df['max_depth'], df['mean_neg_mae'], c=df['mean_neg_mae'], cmap='viridis'), pad=.05, shrink=0.6)
cbar.set_label('MAE')

fig.tight_layout()

# Show the plot
plt.show()

plt.clf()
