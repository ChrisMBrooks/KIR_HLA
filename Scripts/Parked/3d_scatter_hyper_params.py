
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

columns = ['mean_abs_err', 'non_zero_coeff_count', 'l1_ratio', 'alpha', 'filter_str', 'filter']

data = [[-0.47557083143776097, 6, 0.4, 0.1, '<50 NaNs', 50],
[ -0.47331256293336516, 22, 0.30, 0.1, '<40 NaNs', 50],
[-0.4738885507910488, 18, 0.30, 0.1, '<30 NaNs', 30],
[-0.4738885507910488, 18, 0.30, 0.1, '<20 NaNs', 20],
[-0.4732130434986372, 18, 0.30, 0.1, '<10 NaNs', 20],
[-0.4780030665046235, 5, 0.1, 0.4, '<0 NaNs', 0]]

df = pd.DataFrame(data, columns=columns)
susbset_df = df[['l1_ratio', 'alpha', 'mean_abs_err']].copy()

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')

x = df['l1_ratio']
y = df['alpha']
z = df['mean_abs_err']

ax.set_xlabel("L1 Ratio")
ax.set_ylabel("Alpha")
ax.set_zlabel("Neg Mean Abs Error")

ax.scatter(x, y, z)

for index, row in df.iterrows():
    ax.text(x=row['l1_ratio'] + 0.02, y=row['alpha'], z=row['mean_abs_err'], 
        s=str(row['non_zero_coeff_count']), horizontalalignment='left', size='medium', color='black', weight='semibold')

plt.show()