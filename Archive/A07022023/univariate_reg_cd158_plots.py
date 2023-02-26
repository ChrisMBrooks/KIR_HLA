import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import statsmodels.api as sm

from Controllers.ConfigManager import ConfigManager as cm
from Controllers.MySQLManager import MySQLManager as msm
from Controllers.DataManager import DataManager as dtm
from Controllers.LearningManager import LearningManager as lrn

#Instantiate Controllers
use_full_dataset = True
config = cm.ConfigManaager().config
sql = msm.MySQLManager(config=config)
data_mgr = dtm.DataManager(config=config, use_full_dataset=use_full_dataset)
lrn_mgr = lrn.LearningManager(config=config)

# Pull Data from DB

X = data_mgr.features(fill_na = False, partition = 'training')

Y = data_mgr.outcomes(fill_na = False, partition = 'training')

if use_full_dataset:
    filename = 'Data/158a+_trait_ids.csv'
    relevant_phenos = list(pd.read_csv(filename, index_col=0).values[:, 0])
else: 
    relevant_phenos = ['Lin:1', 'Lin:10', 'Lin:11', 'Lin:12', 'Lin:13', 'Lin:14', 'Lin:15', 'Lin:16', 'Lin:17', 'Lin:18', 'Lin:19']

gene_indicator_name = 'kir2dl1'

# CD158a - KIR2DL1
# CD158b1 - KIR2DL2 or KIR2DL3

X0 = X[['public_id', gene_indicator_name]].copy()
Y0 = Y[['public_id'] + relevant_phenos].copy()
Z0 = X0.merge(Y0, on='public_id', how='inner')
Z1 = Z0.dropna()

X1 = Z1[gene_indicator_name].values.reshape(-1,1)
Y1 = Z1[relevant_phenos].values

cutoff = -1*np.log10(0.05)

np.seterr(invalid='raise')

regression_results = lrn_mgr.UnivariateRegressions(X1, Y1)

neg_log_p_score_records = []

for index, row in enumerate(regression_results):
    beta_hat = row[1:]
    p_score_results = lrn_mgr.regression_p_score(X1, Y1[:,0], beta_hat=beta_hat)
    neg_log_p_scores = -1*np.log10(p_score_results)
    neg_log_p_score_records.append(neg_log_p_scores)

Z = np.stack(neg_log_p_score_records)

print('Computation complete. Now on to plotting.')

fig, ax = plt.subplots(nrows=1, ncols=2)
for i in range(0, Z.shape[1]):
    ax[0].scatter(np.ones(Z.shape[0])*(i+1), Z[:, i],  color = '#88c999')
    if i > 0:
        ax[1].scatter(np.ones(Z.shape[0])*(i+1), Z[:, i],  color = '#88c999')

ax[0].set_xticks(np.arange(1, 3 ,1))
ax[1].set_xticks(np.arange(1, 3, 1))

ax[0].tick_params(axis='both', which='major', labelsize=6)
ax[1].tick_params(axis='both', which='major', labelsize=6)

ticks = ['const', gene_indicator_name]
ax[0].set_xticklabels(ticks)
ax[1].set_xticklabels(ticks)

fig.supxlabel(gene_indicator_name)
fig.supylabel('-log10(p-value)')

plt.plot([0, 6], [cutoff, cutoff], 'r-')
plt.savefig('Analysis/cd158a+_signif_plot.png')
plt.clf()
plt.cla()

""" regression_results = lrn_mgr.UnivariateRegressions(X1, Y1)
print(regression_results)

beta_1 = regression_results[0][1:]
print(beta_1)

p_score_results = lrn_mgr.regression_p_score(X1, Y1[:,0], beta_hat=beta_1)
print('p-scoes', p_score_results)
print('cutoff', -1*np.log10(0.05), '-log10(p)', -1*np.log10(p_score_results))

X2 = sm.add_constant(X1)
est = sm.OLS(Y1[:, 0], X2)
est2 = est.fit()
print(est2.summary())   
"""

for i in range(0, 1, 2):
#for i in range(0, 1, Y1.shape[1]):
    plt.scatter(X1[:,0], Y1[:,i])
    plt.savefig('Analysis/kir2dl1_{}_ols_scatter_plot.png'.format(relevant_phenos[i]))