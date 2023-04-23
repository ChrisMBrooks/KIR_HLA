
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Controllers.DataScienceManager import DataScienceManager as dsm

#Instantiate Controllers
use_full_dataset=True
data_sci_mgr = dsm.DataScienceManager(use_full_dataset=use_full_dataset)

Y = data_sci_mgr.data_mgr.outcomes(fill_na = False, partition = 'training')
m = Y.shape[1]

# Get Non NaN Immunos and write to CSV
Z0  = Y.isna().sum()
for i in range(0, 60, 10):
    Z1  = Z0[Z0 > i]
    columns = set(Y.columns[1:-2]).difference(set(Z1.index))
    series = pd.Series(list(columns))
    filename = "Data/Subsets/na_filtered_phenos_less_thn_{}_zeros.csv".format(str(i))
    series.to_csv(filename)

# Make a Histogram
#Zn  = [Z0[Z0 >  n ].shape[0] for n in range(0,210,10)] 
#Znn  = [100.0*(Z0[Z0 >  n ].shape[0])/m for n in range(0,210,10)] 

#Z.plot.bar(x='lab', y='val', rot=0)
#Z.plot.hist(bins=200)
#plt.show(block=True)