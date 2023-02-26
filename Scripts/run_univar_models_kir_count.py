import numpy as np
import pandas as pd

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

# Pull Data from DB

features = data_mgr.features(fill_na = True, fill_na_value = 0.0, 
        partition = 'training')

features['kir_count'] = features['kir2dl1'] + features['kir2dl2'] + features['kir2dl3'] + features['kir3dl1']
X = features['kir_count'].values.astype(float).reshape(-1,1)

Y = data_mgr.outcome_values(normalise = True, fill_na = True, fill_na_value = 0.0, 
        partition = 'training')

columns = ["run_id" , "feature_name", "beta_0", "beta_1"]

results = lrn_mgr.UnivariateRegression(X, Y)

results_df = pd.DataFrame(results, columns=[columns[0]]+columns[2:])
results_df.insert(1, columns[1], data_mgr.data['outcomes'].columns[1:-2])

data_mgr.insert_df_to_sql_table(df=results_df, columns=columns, schema='KIR_HLA_STUDY', 
    table='model_result_ols_kir_count', use_batches=True, batch_size=5000)

# Generate Histogram
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

ticks = ['const', 'kir count']
ax[0].set_xticklabels(ticks)
ax[1].set_xticklabels(ticks)

fig.supxlabel('iKIR Count')
fig.supylabel('-log10(p-value)')

plt.plot([0, 6], [cutoff, cutoff], 'r-')
plt.savefig('Analysis/n_80000_scatter_plot_kir_count.png')

sig_phenos_df = pd.DataFrame(significant_immunophenotypes, columns=['feature_name', 'nlog_p_0', 'nlog_p_1'])
sig_phenos_df.to_csv('Analysis/n_80000_signif_immunophenotypes_kir_count.csv')

failed_phenos_df = pd.DataFrame(failed_immunophenotypes, columns=['feature_name', 'beta_0', 'beta_1'])
failed_phenos_df.to_csv('Analysis/n_80000_failed_immunophenotypes_kir_count.csv')

remaining_phenos_df = pd.DataFrame(remaining_immunophenotypes, columns=['feature_name', 'nlog_p_0', 'nlog_p_1'])
remaining_phenos_df.to_csv('Analysis/n_80000_other_immunophenotypes_kir_count.csv')