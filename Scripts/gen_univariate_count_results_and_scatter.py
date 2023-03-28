import uuid
from datetime import date
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme()

from Controllers.ConfigManager import ConfigManager as cm
from Controllers.MySQLManager import MySQLManager as msm
from Controllers.DataManager import DataManager as dtm
from Controllers.LearningManager import LearningManager as lrn

def get_univariate_anlysis_df(write_to_table:bool):
    run_id = str(uuid.uuid4().hex)

    # Pull Data from DB
    X = data_mgr.features(fill_na = False, partition = 'training')
    X['ikir_count'] = X['kir2dl1'] + X['kir2dl2'] + X['kir2dl3'] + X['kir3dl1']
    X0 = X[['public_id', 'ikir_count']]

    Y = data_mgr.outcomes(fill_na = False, partition = 'training')

    results = []
    for feature_name in Y.columns[1:-2]:
        Y0 = Y[['public_id', feature_name]]
        Z = X0.merge(Y0, on='public_id', how='inner')
        
        #Filter NAs
        Z0 = Z[~Z.isna().any(axis=1)]

        Z1 = Z0[['ikir_count', feature_name]].values
        X1 = Z1[:, 0]
        Y1 = Z1[:, 1]

        beta_hat = lrn_mgr.UnivariateRegression(X1, Y1)[1:]
        #p_scores = lrn_mgr.regression_p_score(X1, Y1, beta_hat=beta_hat)
        p_scores = lrn_mgr.regression_p_score2(X1, Y1)

        #Out of the Box Check of P-value calculation
        #sp_stats_r = sp.stats.linregress(x=X1, y=Y1, alternative='two-sided')
        #print(sp_stats_r)

        results.append([run_id, feature_name] + beta_hat + [p_scores[1]])
        
    columns = ["run_id" , "feature_name", "beta_0", "beta_1", "p_1"]
    results_df = pd.DataFrame(results, columns=columns)
    
    if write_to_table:
        data_mgr.insert_df_to_sql_table(df=results_df, columns=columns, schema='KIR_HLA_STUDY', 
            table='model_result_ols_kir_count', use_batches=True, batch_size=5000)

    return results_df

def generate_scatter(ols_results:pd.DataFrame, filename:str):
    #Drop NAs (some p_values may have nulls)
    ols_results = ols_results[~ols_results.isna().any(axis=1)].copy()
    ols_results['-log10(p-val)'] = -1*np.log10(ols_results['p_1'])

    cutoff = -1*np.log10(0.05)
    cutoff2 = -1*np.log10(0.05/80000)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(np.ones(ols_results['-log10(p-val)'].shape), ols_results['-log10(p-val)'])
    ax.axhline(cutoff, ls='--')
    ax.axhline(cutoff2, ls='--')
    ax.text(1.005, cutoff + 0.05, "-log10(0.05)", c='blue')
    ax.text(1.005, cutoff2 + 0.05, "-log10(0.05/80000)", c='blue')
    plt.xticks([1], [''])
    plt.ylabel('-log10(p-val)')
    plt.title('Univariate Null Hypothesis Significance Testing - iKIR Count')

    #plt.show()
    plt.savefig(filename)

#Instantiate Controllers
config = cm.ConfigManaager().config
sql = msm.MySQLManager(config=config)
data_mgr = dtm.DataManager(config=config, use_full_dataset=True)
lrn_mgr = lrn.LearningManager(config=config)

ols_results = get_univariate_anlysis_df(write_to_table=True)

# ols_results = sql.read_table_into_data_frame(schema_name='KIR_HLA_STUDY', 
#     table_name='model_result_ols')

date_str = date.today().strftime("%d%m%Y")
filename = 'Analysis/Univariate/univar_count_p_value_scatter_{}.png'.format(date_str)
generate_scatter(ols_results=ols_results, filename=filename)
print('Complete.')
