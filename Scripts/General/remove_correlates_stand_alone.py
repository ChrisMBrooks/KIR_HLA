import pandas as pd
import datetime

def get_phenotypes(partition = 'training'):
    data = {}
    filename = "Data/validation_partition.csv"
    data['partitions'] = pd.read_csv(filename)

    filename = "Data/trait_values_transpose.parquet"
    data['imm_phenotype_measurements'] = pd.read_parquet(filename) 

    if partition == 'training':
        pheno_train_df = data['imm_phenotype_measurements'].merge(
            data['training_partition'], on='subject_id', how='inner'
        )
    elif partition == 'validation':
        pheno_train_df = data['imm_phenotype_measurements'].merge(
            data['validation_partition'], on='subject_id', how='inner'
        )
    else:
        pheno_train_df = data['imm_phenotype_measurements'].merge(
            data['partitions'], on='subject_id', how='inner'
        )

    return  pheno_train_df

def get_correlation_dict(corr_df:pd.DataFrame, threshold:float):
    corr_dict = {}
    for row_index, row in corr_df.iterrows():
        for col_index, col in row.items():
            corr_val = row[col_index]
            if row_index != col_index and corr_val >= threshold:
                if not row_index in corr_dict:
                    corr_dict[row_index] = []
                corr_dict[row_index].append(col_index)
    return corr_dict

def overlap(A:list, B:list):
    return list(set(A) & set(B))

def remove_correlates(corr_dict:dict):
    keys_to_check = list(corr_dict.keys())
    uniques = []

    while len(keys_to_check) > 0:
        current_key = keys_to_check[-1]
        current_correlates = corr_dict[current_key]
        uniques.append(current_key)
        
        items_to_remove = [current_key] + current_correlates
        for item in items_to_remove:
            if item in keys_to_check:
                keys_to_check.remove(item)

    return uniques

print('Starting...')

#Instantiate Controllers

phenos_df = get_phenotypes(partition = 'everything')
y_labels = phenos_df.columns[1:-2]
phenos_df = phenos_df[y_labels].copy()

# Prep Data
phenos_df = phenos_df.reindex(sorted(phenos_df.columns), axis=1)

print('Computing Correlation Matrix...')
corr_df = phenos_df.corr()

#clean-up 2
del phenos_df

print('Computing Correlation Dict')
threshold = 0.95
corr_dict = get_correlation_dict(corr_df, threshold)

print('Removing Correlates')
uniques = remove_correlates(corr_dict)
uniques = pd.Series(uniques, name='unlike_phenos')

date = datetime.datetime.now().strftime("%d%m%Y")
filename = "unlike_phenos_{}.csv".format(date)
uniques.to_csv(filename)