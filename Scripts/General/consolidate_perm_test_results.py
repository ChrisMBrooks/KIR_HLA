import os, datetime
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme()

directory = "Data/perm_test_results/"
results = []
for idx, filename in enumerate(os.listdir(directory)):
    full_path = os.path.join(directory, filename)
    values = list(pd.read_csv(full_path, index_col=0).values[:, 0])
    results.extend(values)

results = pd.DataFrame(results, columns=['p_vals'])

current_date = datetime.datetime.now().strftime("%d%m%Y")
filename = "Data/perm_test_results_{}.csv".format(current_date)
results.to_csv(filename)

filename = "Data/perm_test_results_{}.csv".format(current_date)

results = pd.read_csv(filename, index_col=0)

sns.histplot(data=results, x='p_vals', kde = True)

q05 = results.quantile(q=0.05).values[0]

"""

The 5% quantile, indicates the point where 5% percent of the data have values less than the number. 
More generally, the pth percentile is the number n for which p% of the data is less than n

"""

plt.axvline(x=q05, color='r', linestyle='--')
plt.text(x=q05, y=800, s=format(q05, '.2e'))
plt.title(label='Permutation Test Results')

filename = 'Analysis/Univariate/perm_test_hist{}.png'.format(current_date)

plt.savefig(fname=filename)