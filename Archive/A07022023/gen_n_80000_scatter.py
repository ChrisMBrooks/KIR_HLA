import os, random, math, time
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt

from Controllers.ConfigManager import ConfigManager as cm
from Controllers.MySQLManager import MySQLManager as msm
from Controllers.DataManager import DataManager as dtm
from Controllers.LearningManager import LearningManager as lrn

#Instantiate Controllers
config = cm.ConfigManaager().config
sql = msm.MySQLManager(config=config)
data_mgr = dtm.DataManager(config=config, use_full_dataset=True)
lrn_mgr = lrn.LearningManager(config=config)

X = data_mgr.feature_values(normalise = True, fill_na = True, fill_na_value = 0.0, 
        partition = 'training')

Y = data_mgr.outcome_values(normalise = True, fill_na = True, fill_na_value = 0.0, 
        partition = 'training')

ols_results = sql.read_table_into_data_frame(schema_name='KIR_HLA_STUDY', 
    table_name='model_result_ols_kir_count')

cutoff = -1*np.log10(0.05/80000)

np.seterr(invalid='raise')

neg_log_p_score_records = []
failed_immunophenotypes = []
significant_immunophenotypes = []
remaining_immunophenotypes = []
for index, row in ols_results.iterrows():
    feature_name = row['feature_name']
    idx = np.where(data_mgr.data['outcomes'].columns[1:-2] == feature_name)[0][0]
    beta_hat = [row[column] for column in ols_results.columns[2:]]
    try:
        p_scores = lrn_mgr.regression_p_score(X, Y[:,idx], beta_hat=beta_hat)
        neg_log_p_scores = -1*np.log10(p_scores)
        neg_log_p_score_records.append(neg_log_p_scores)

        for score in neg_log_p_scores[1:]:
            if score > cutoff:
                significant_immunophenotypes.append([feature_name] + list(neg_log_p_scores))
                break
        remaining_immunophenotypes.append([feature_name] + list(neg_log_p_scores))

    except:
        neg_log_p_score_records.append(np.zeros(len(beta_hat)))
        failed_immunophenotypes.append([feature_name] + list(beta_hat))
        #print('Failed to transform {} values.'.format(feature_name))

Z = np.stack(neg_log_p_score_records)

print(Z)
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

ticks = ['const', 'kir score']
ax[0].set_xticklabels(ticks)
ax[1].set_xticklabels(ticks)

fig.supxlabel('Functional iKIR')
fig.supylabel('-log10(p-value)')

plt.plot([0, 6], [cutoff, cutoff], 'r-')
plt.savefig('Analysis/n_80000_scatter_plot.png')

sig_phenos_df = pd.DataFrame(significant_immunophenotypes, columns=['feature_name', 'nlog_p_0', 'nlog_p_1'])
sig_phenos_df.to_csv('Analysis/n_80000_signif_immunophenotypes.csv')

failed_phenos_df = pd.DataFrame(failed_immunophenotypes, columns=['feature_name', 'beta_0', 'beta_1'])
failed_phenos_df.to_csv('Analysis/n_80000_failed_immunophenotypes.csv')

remaining_phenos_df = pd.DataFrame(remaining_immunophenotypes, columns=['feature_name', 'nlog_p_0', 'nlog_p_1'])
remaining_phenos_df.to_csv('Analysis/n_80000_other_immunophenotypes.csv')