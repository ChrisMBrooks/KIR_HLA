import pandas as pd
import numpy as np
import math, uuid
import time

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LassoCV
from scipy import stats

class LearningManager():
    def __init__(self, config:dict):
        self.config = config
        self.setup()

    def setup(self):
        pass

    def ElasticNetCoefficientAnalysis(self, x:pd.DataFrame, y:pd.DataFrame, predictor_id:str, 
        grid_search=False, default_alpha = 1, default_l1 = 0.5
        ):

        self.model = ElasticNet()

        if grid_search:
            # Perform Grid Search for Hyper Params
            #start_time = time.time()

            # Instantiate evaluation method model
            n_splits = 10
            n_repeats = 10
            cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1)

            hyper_params_grid = dict()
            hyper_params_grid['alpha'] = [0.01, 0.1, 1.0, 10.0, 100.0]
            hyper_params_grid['l1_ratio'] = np.arange(0.01, 1, 0.1)

            search = GridSearchCV(self.model, hyper_params_grid, scoring='neg_mean_absolute_error', 
                cv=cv, n_jobs=-1
            )

            grid_search_results = search.fit(x, y)
            best_fit_hyper_params = grid_search_results.best_params_
            best_fit_mae = grid_search_results.best_score_

            #elapsed_time = time.time()-start_time
            #print("Grid Search Elapsed Time: {}".format(elapsed_time))
        
        else:
            best_fit_hyper_params = {'alpha':default_alpha, 'l1_ratio': default_l1}
            best_fit_mae = 0

        # Instaniate EN model w/ Chosen Hyper Params
        #start_time = time.time()

        model = ElasticNet(alpha=best_fit_hyper_params['alpha'], 
            l1_ratio=best_fit_hyper_params['l1_ratio']
        )

        #Fit Model
        model.fit(x, y)

        #elapsed_time = time.time()-start_time
        #print("Fit Mode Elapsed Time: {}".format(elapsed_time))

        #Prep Results
        coefficients = np.array([[i,v,math.fabs(v)] for i,v in enumerate(model.coef_)])

        coefficients_df = pd.DataFrame(
            coefficients, columns=['feature_id', 'coeff', 'absv_coeff']
        )
        coefficients_df.sort_values(by=['absv_coeff'], ascending=False, inplace=True)

        pp_cut_off = 0.01
        cut_off = pp_cut_off*coefficients_df.iloc[0]['absv_coeff']

        if coefficients_df.iloc[0]['absv_coeff'] == 0:
            print('No correlation was found for predictor: {}'.format(predictor_id))
            return pd.DataFrame()
        else: 
            primary_coefficients_df = coefficients_df[coefficients_df['absv_coeff'] > cut_off].copy()
            primary_coefficients_df.drop(['absv_coeff'], axis='columns', inplace=True)

            primary_coefficients_df['relevance_cut_off'] = pp_cut_off
            primary_coefficients_df['predictor_id'] = predictor_id
            primary_coefficients_df['alpha'] = best_fit_hyper_params['alpha']
            primary_coefficients_df['l1_ratio'] = best_fit_hyper_params['l1_ratio']

            if grid_search:
                primary_coefficients_df['measured_abs_error'] = best_fit_mae

                alpha_range_str = '{} to {}'.format(
                    hyper_params_grid['alpha'][0], hyper_params_grid['alpha'][-1]
                )
                primary_coefficients_df['grid_search_alpha_range'] = alpha_range_str

                l1_ratio_range = '{} to {}'.format(
                    hyper_params_grid['l1_ratio'][0], hyper_params_grid['l1_ratio'][-1]
                )

                primary_coefficients_df['grid_search_l1_ratio_range'] = l1_ratio_range
                primary_coefficients_df['cross_val_n_splits'] = n_splits
                primary_coefficients_df['cross_val_n_repeats'] = n_repeats

            else:
                primary_coefficients_df['measured_abs_error'] = 0
                primary_coefficients_df['grid_search_alpha_range'] = str(default_alpha)
                primary_coefficients_df['grid_search_l1_ratio_range'] = str(default_l1)
                primary_coefficients_df['cross_val_n_splits'] = 0
                primary_coefficients_df['cross_val_n_repeats'] = 0

            primary_coefficients_df['run_id'] = str(uuid.uuid4().hex)

        return primary_coefficients_df
    
    def LassoCoefficientAnalysis(self, x:pd.DataFrame, y:pd.DataFrame, predictor_id:str):
        
        #Significance testing, ridge regression, genetic data: https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-12-372

        n_splits=3
        n_repeats=3

        cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, 
            random_state=1
        )

        alphas = [0.5, 1, 10]

        model = LassoCV(n_alphas=1, alphas=alphas, 
            random_state=0, cv=cv
        )

        # LassoCV leads to different results than a hyperparameter search using GridSearchCV with a Lasso model. 
        # In LassoCV, a model for a given penalty alpha is warm started using the coefficients of the closest model 
        # (trained at the previous iteration) on the regularization path. It tends to speed up the hyperparameter search.

        #Fit Model
        model.fit(x, y)

        #Prep Results
        coefficients = np.array([[i,v,math.fabs(v)] for i,v in enumerate(model.coef_)])

        coefficients_df = pd.DataFrame(
            coefficients, columns=['feature_id', 'coeff', 'absv_coeff']
        )
        coefficients_df.sort_values(by=['absv_coeff'], ascending=False, inplace=True)

        cut_off = 0.0

        if coefficients_df.iloc[0]['absv_coeff'] == 0:
            print('No correlation was found for predictor: {}'.format(predictor_id))
            return pd.DataFrame()
        else: 
            primary_coefficients_df = coefficients_df[coefficients_df['absv_coeff'] > cut_off].copy()
            primary_coefficients_df.drop(['absv_coeff'], axis='columns', inplace=True)

            primary_coefficients_df['relevance_cut_off'] = cut_off
            primary_coefficients_df['predictor_id'] = predictor_id
            primary_coefficients_df['alpha'] = model.alpha_
            primary_coefficients_df['l1_ratio'] = 1.0


            primary_coefficients_df['measured_abs_error'] = mean_squared_error(y_true = y, y_pred = model.predict(x))
            primary_coefficients_df['grid_search_alpha_range'] = "{} to {}".format(alphas[0], alphas[-1])
            primary_coefficients_df['grid_search_l1_ratio_range'] = 1.0
            primary_coefficients_df['cross_val_n_splits'] = n_splits
            primary_coefficients_df['cross_val_n_repeats'] = n_repeats

            primary_coefficients_df['run_id'] = str(uuid.uuid4().hex)

        return primary_coefficients_df
    
    