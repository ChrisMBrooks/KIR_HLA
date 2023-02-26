import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from Controllers.ConfigManager import ConfigManager as cm
from Controllers.MySQLManager import MySQLManager as msm
from Controllers.DataManager import DataManager as dtm
from Controllers.LearningManager import LearningManager as lrn

#Instantiate Controllers
config = cm.ConfigManaager().config
sql = msm.MySQLManager(config=config)
data_mgr = dtm.DataManager(config=config, use_full_dataset=False)
lrn_mgr = lrn.LearningManager(config=config)

ols_results = sql.read_table_into_data_frame(schema_name='KIR_HLA_STUDY', 
    table_name='model_result_ols')

#Drop NAs (some p_values may have nulls)
ols_results = ols_results[~ols_results.isna().any(axis=1)]

ols_results['-log10(p-val)'] = -1*np.log10(ols_results['p_1'])

#print(ols_results[ols_results['p_1'] < 0.05])

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
plt.title('Univariate Null Hypothesis Significance Testing')

#plt.show()
plt.savefig('Analysis/Univariate/univar_p_value_scatter_20022023.png')