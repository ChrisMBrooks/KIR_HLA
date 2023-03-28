import pandas as pd
import numpy as np
import scipy as sp, random
import uuid

import statsmodels.api as sm

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

class LearningManager():
    def __init__(self, config:dict):
        self.config = config
        self.setup()

    def setup(self):
        self.ikir_dist_filename = './Data/iKIR_scores_distribution.csv'
        self.ikir_score_dist_df = pd.read_csv(self.ikir_dist_filename)

    def get_random_iKIR_score(self):
        n = 100000.0
        random_probability = random.randint(0,n)/n

        for index, row in self.ikir_score_dist_df.iterrows():
            if random_probability >= row['Min Threshold'] and \
                random_probability <= row['Max Threshhold']:
                return row['iKIR Score']
        return 0

    def generate_iKIR_scores(self,n:int, normalise = False):
        mms = MinMaxScaler(feature_range=(-1, 1))
        mms.fit(np.arange(0.0, 4.26, 0.25).reshape(-1, 1))

        rand_scores = []
        for i in range(0, n, 1):
            x = self.get_random_iKIR_score()
            rand_scores.append(x)
        rand_scores = np.array(rand_scores)
        if normalise:
            rand_scores = mms.transform(rand_scores.reshape(-1,1))
        return rand_scores
    
    def regression_p_score(self, X, Y, beta_hat):
        #This doesn't seem to work for multivar. Need to explore why. Use statsmodel instead. 

        # Code Source: https://stackoverflow.com/questions/27928275/find-p-value-significance-in-scikit-learn-linearregression
        # Another source with maths: https://stats.stackexchange.com/questions/352383/how-to-calculate-p-value-for-multivariate-linear-regression
        # add ones column to accomodate the (constant) intercept 
        n = X.shape[0]
        X1 = np.column_stack((np.ones(n), X))

        # standard deviation of the noise.
        sigma_hat = np.sqrt(np.sum(np.square(Y - X1@beta_hat)) / (n - X1.shape[1]))

        # estimate the covariance matrix for beta 
        beta_cov = np.linalg.inv(X1.T@X1)

        # the t-test statistic for each variable from the formula from above figure
        t_vals = beta_hat / (sigma_hat * np.sqrt(np.diagonal(beta_cov)))

        # compute 2-sided p-values.
        p_vals = sp.stats.t.sf(np.abs(t_vals), n-X1.shape[1])*2 
        return p_vals
    
    def regression_p_score2(self, X, Y):
        X2 = sm.add_constant(X)
        model = sm.OLS(Y, X2)
        results = model.fit()
        p_vals = results.pvalues
        coeff = results.params

        #p_vals included p_val for constant, use pvals[1:] to skip. 
        return p_vals

    def UnivariateRegressions(self, X, Y):
        results = []
        run_id = str(uuid.uuid4().hex)
        for i in range(0, Y.shape[1]):
            Y1 = Y[:,i]

            model = LinearRegression().fit(X, Y1)
            beta_hat = [model.intercept_] + model.coef_.tolist()
            beta_hat = [float(x) for x in beta_hat]

            result = [run_id] + beta_hat
            results.append(result)
        
        return results
    
    def UnivariateRegression(self, X, Y):
        X = np.array(X).reshape(-1, 1)
        Y = np.array(Y).reshape(-1, 1)
        run_id = str(uuid.uuid4().hex)
        model = LinearRegression().fit(X, Y)

        beta_hat = [model.intercept_] + model.coef_[0].tolist()
        beta_hat = [float(x) for x in beta_hat]

        result = [run_id] + beta_hat
        
        return result
            

    