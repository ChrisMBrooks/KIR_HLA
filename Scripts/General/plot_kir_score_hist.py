import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme()

from Controllers.ConfigManager import ConfigManager as cm
from Controllers.DataManager import DataManager as dtm

#Instantiate Controllers
config = cm.ConfigManaager().config
data_mgr = dtm.DataManager(config=config, use_full_dataset=True)

# Pull Data from DB

X0 = data_mgr.features(fill_na = False, partition = 'training')
X1 = data_mgr.features(fill_na = False, partition = 'validation')
X2 = pd.concat([X0, X1])

#num_bins = 8
#n, bins, patches = plt.hist(X, num_bins, facecolor='blue', alpha=0.5)

sns.histplot(data=X2, x="f_kir_score", binwidth=0.25)

#plt.show()
plt.title('Histogram of Functional iKIR Score')
plt.savefig('Analysis/Data Characterisation/iKIR_score_hist20022023.png')