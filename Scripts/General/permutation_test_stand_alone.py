import time, sys 
import numpy as np
import pandas as pd
import statsmodels.api as sm

def import_data():
    part_filename = "Data/validation_partition.csv"
    trait_filename = "Data/trait_values_transpose.parquet"
    kir_filename = "Data/ikir_scores.csv"
    data = {}
    data['partitions'] = pd.read_csv(part_filename)

    data['training_partition'] =  data['partitions'][
        data['partitions']['validation_partition'] == 'TRAINING'
    ]

    data['validation_partition'] =  data['partitions'][
        data['partitions']['validation_partition'] == 'VALIDATION'
    ]

    data['func_kir_genotype'] = pd.read_csv(kir_filename)
    data['imm_phenotype_measurements'] = pd.read_parquet(trait_filename) 
    return data

def ikir_scores(data:dict, fill_na = True, fill_na_value = 0.0, partition = 'training'):

    if partition == 'training':
        f_kir_geno_train_df = data['func_kir_genotype'].merge(
            data['training_partition'], on='public_id', how='inner'
        )
    elif partition == 'validation':
        f_kir_geno_train_df = data['func_kir_genotype'].merge(
            data['validation_partition'], on='public_id', how='inner'
        )
    else:
        f_kir_geno_train_df = data['func_kir_genotype']

    f_kir_geno_train_df.sort_values(by='public_id', inplace=True)

    if fill_na:
        f_kir_geno_train_df.fillna(fill_na_value, inplace=True)

    data['features'] = f_kir_geno_train_df

    return data['features']

def phenotypes(data:dict, partition = 'training'):
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

def regression_p_score2(X, Y):
    X2 = sm.add_constant(X)
    model = sm.OLS(Y, X2)
    results = model.fit()
    p_vals = results.pvalues
    coeff = results.params

    return p_vals

job_id = sys.argv[1]
start_time = time.time()
data = import_data()

Xa = ikir_scores(data, fill_na = False, partition = 'training')
Xb = ikir_scores(data, fill_na = False, partition = 'validation')
X = pd.concat([Xa, Xb])

Ya = phenotypes(data, partition = 'training')
Yb = phenotypes(data, partition = 'validation')
Y = pd.concat([Ya, Yb])

Z = X.merge(Y, on='public_id', how='inner')

p_vals = []
n = 10
for i in range(n):
    p_vals_i = []

    # Shuffle your independent variables
    feature_set = Z['f_kir_score'].copy().values
    np.random.shuffle(feature_set)
    Z['f_kir_score'] = feature_set
    
    for feature_name in Y.columns[1:-2]:
        Z1 = Z[['public_id', feature_name, 'f_kir_score']]
        
        #Filter NAs
        Z1 = Z1[~Z1.isna().any(axis=1)].copy()
        Z1 = Z1[['f_kir_score', feature_name]].values
        
        feature_set = Z1[:, 0]
        outcome_set = Z1[:, 1]

        p_val = regression_p_score2(feature_set, outcome_set)[-1]
        p_vals_i.append(p_val)
    min_p = min(p_vals_i)
    p_vals.append(min_p)

p_vals = np.array(p_vals)
p_vals_s = pd.Series(p_vals, name='permutation_p_vals')

filename = "Results/perm_test_raw_data_{}.csv".format(job_id)
p_vals_s.to_csv(filename)

elapsed_time = time.time() - start_time
print(elapsed_time)