import numpy as np
import pandas as pd
import time, uuid, random, math

from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor

from Controllers.DataScienceManager import DataScienceManager as dsm

def import_data():
    part_filename = "Data/validation_partition.csv"
    trait_filename = "Data/trait_values_transpose.parquet"
    kir_filename = "Data/ikir_scores.csv"
    data = {}
    data['partitions'] = pd.read_csv(part_filename, index_col=0)

    data['training_partition'] =  data['partitions'][
        data['partitions']['validation_partition'] == 'TRAINING'
    ]

    data['validation_partition'] =  data['partitions'][
        data['partitions']['validation_partition'] == 'VALIDATION'
    ]

    data['func_kir_genotype'] = pd.read_csv(kir_filename, index_col=0)
    ikir_cols = data['func_kir_genotype'].columns

    data['imm_phenotype_measurements'] = pd.read_parquet(trait_filename)
    phenos_cols = data['imm_phenotype_measurements'].columns

    master_df =  data['partitions'].merge(data['imm_phenotype_measurements'], 
        on='subject_id', how='inner'
    )

    master_df = master_df.merge(
        data['func_kir_genotype'], on='public_id', how='inner'
    )

    data['func_kir_genotype'] = master_df[ikir_cols]
    data['imm_phenotype_measurements'] = master_df[phenos_cols]

    return data

def construct_distribution(population:np.ndarray):
    counts = {}
    for i in range(population.shape[0]):
        item = population[i, 0]
        if item not in counts:
            counts[item] = 1
        else:
            counts[item] += 1

    counts = {key:value/population.shape[0] for (key,value) in counts.items()}
    keys = tuple(sorted(list(counts.keys())))

    distribution = {}
    distribution[keys[0]] = (0, counts[keys[0]])
    for i in range(1, len(keys)):
        min = distribution[keys[i-1]][1]
        max = min + counts[keys[i]]
        distribution[keys[i]] = (min, max)

    return distribution

def get_random_ikir_score(distribution):
    i = random.randint(1, 1000000)/1000000
    for key in distribution:
        tup = distribution[key]
        if i > tup[0] and i <= tup[1]:
            return key
    return 0

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

        pheno_train_df.sort_values(by='public_id', inplace=True)

    return  pheno_train_df

def ikir_scores(data:dict, partition = 'training'):

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

    data['features'] = f_kir_geno_train_df

    return data['features']

def generate_psuedo_data(ikir_distribution, length:int):
    row_count = length
    psuedo_pheno_values = rng.normal(loc=0, scale=1, size = row_count).reshape(-1,1)
    psuedo_ikir_scores = np.array([get_random_ikir_score(ikir_distribution) for x in range(row_count)]).reshape(-1,1)

    psuedo_pheno_values, psuedo_ikir_scores = data_sci_mgr.data_mgr.preprocess_data(
        psuedo_pheno_values, psuedo_ikir_scores
    )
    
    df = pd.DataFrame(psuedo_pheno_values, columns=['psuedo_phenotype_measurement'])
    df['psuedo_ikir_score'] = psuedo_ikir_scores

    return df

def evaluate_model(phenos, scores):
    # Data Prep
    scores = scores['f_kir_score'].values.reshape(-1,1)
    if len(phenos.values.shape) == 1:
        phenos = phenos.values.reshape(-1,1)
    else:
        phenos = phenos.values[:, 0:]

    phenos, scores = data_sci_mgr.data_mgr.preprocess_data(phenos, scores)
    scores = scores.ravel()
    if len(phenos.shape) == 1:
        phenos = phenos.reshape(-1,1)

    # Cross Validation
    n_splits = 2
    n_repeats = 2
    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1)

    # Grid Search
    max_nodes_min = 3
    max_nodes_max = 40
    max_nodes_step = 1

    num_trees_min = 10
    num_trees_max = 400

    gsc = GridSearchCV(
        estimator=RandomForestRegressor(),
        param_grid={
            'max_depth': range(max_nodes_min, max_nodes_max, max_nodes_step),
            'n_estimators': (num_trees_min, 50, 100, num_trees_max),
            # max_depth: the maximum depth of the tree 
            # n_estimators: the number of trees in the forest
        },
        cv=cv, 
        scoring='neg_mean_squared_error', 
        verbose=1, 
        n_jobs=-1 # parallelism, -1 means using all processors
    )

    grid_result = gsc.fit(phenos, scores) # Regress Immunos on iKIR Score
    h_params = grid_result.best_params_

    model = RandomForestRegressor(
        max_depth=h_params["max_depth"], 
        n_estimators=h_params["n_estimators"],
        bootstrap=True,
        max_samples=0.8,
        random_state=False, 
        verbose=1
    )

    ikir_score_predictions = cross_val_score(model, phenos, scores, 
    cv=cv, scoring='neg_mean_absolute_error'
    )

    score = round(ikir_score_predictions.mean(), 3)
    return score, h_params

start_time = time.time()
run_id = str(uuid.uuid4().hex)

#Instantiate Controller
use_full_dataset = False
data_sci_mgr = dsm.DataScienceManager(use_full_dataset=use_full_dataset)

rng = np.random.default_rng(seed=42)

start_time = time.time()
data = import_data()

population = ikir_scores(data, partition = 'all')[['f_kir_score']].values.astype(float).reshape(-1,1)
ikir_distribution = construct_distribution(population)

scores = ikir_scores(data, partition = 'training')
phenos = phenotypes(data, partition = 'training')

phenos = phenos[['public_id'] + list(phenos.columns[1:-2])]
scores = scores[['public_id', 'f_kir_score']]

pheno_labels = phenos.columns[1:]
for pheno_label in pheno_labels:

    pheno = phenos[pheno_label].copy()
    score, h_params = evaluate_model(pheno, scores)
    print('first score:')
    print(h_params)
    print(score)
    
    deltas = []
    for i in range(25):
        # Get Fake Data
        row_count = pheno.shape[0]
        pseudo_data = generate_psuedo_data(ikir_distribution, row_count)
        phenos['psuedo_phenotype_measurement'] = pseudo_data['psuedo_phenotype_measurement']
        pheno = phenos[[pheno_label, 'psuedo_phenotype_measurement']].copy()
        print(pheno)
        
        sec_score, sec_h_params = evaluate_model(pheno, scores)
        print('second score:')
        print(sec_h_params)
        print(sec_score)

        print('difference')
        delta = round(math.fabs(score-sec_score),3)
        print(delta)
    break
np.array(deltas)

run_time = time.time() - start_time 

print(run_time)
