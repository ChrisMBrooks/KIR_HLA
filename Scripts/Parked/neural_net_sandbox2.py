# References
# https://pytorch.org/tutorials/beginner/nn_tutorial.html

import numpy as np
import pandas as pd
import torch as t

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from Controllers.ConfigManager import ConfigManager as cm
from Controllers.MySQLManager import MySQLManager as msm
from Controllers.DataManager import DataManager as dtm
from Controllers.LearningManager import LearningManager as lrn

class NeuralNet(t.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.fc1 = t.nn.Linear(input_size, hidden_size, bias=False)
        self.fc2 = t.nn.Linear(input_size, hidden_size, bias=False)
        self.fc3 = t.nn.Linear(hidden_size, output_size, bias=False)
    
    def forward(self, x):
        out0 = t.nn.functional.relu(self.fc1(x))
        out1 = t.nn.functional.relu(self.fc1(out0))
        out2 = self.fc2(out1)
        return out2

class SimpleNeuralNetRegressor():
    def __init__(self, epochs = 20, batch_size = 64, learning_rate = 0.000000001):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    def loss_func(self):
        return t.nn.L1Loss()

    def loss_batch(self, model, loss_func, xb, yb, opt=None):
        loss = loss_func()(model(xb), yb)

        if opt is not None:
            loss.backward()
            opt.step()
            opt.zero_grad()

        value, count = loss.item(), len(xb)
        return value, count

    def fit(self, epochs, model, loss_func, opt, train_dl, valid_dl):
        epoch_losses = []
        for epoch in range(epochs):
            model.train()
            for xb, yb in train_dl:
                self.loss_batch(model, loss_func, xb, yb, opt)

            model.eval()
            with t.no_grad():
                losses, nums = zip(
                    *[self.loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
                )
            val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

            epoch_losses.append((epoch, val_loss))
        return epoch_losses

    def format_data(self, t_features, t_observations, v_features, v_observations, bs):
        t_features_tnsr = t.Tensor(t_features)
        t_observations_tnsr = t.Tensor(t_observations)

        v_features_tnsr = t.Tensor(v_features)
        v_observations_tnsr = t.Tensor(v_observations)

        training_ds = t.utils.data.TensorDataset(t_features_tnsr, t_observations_tnsr)
        validation_ds = t.utils.data.TensorDataset(v_features_tnsr, v_observations_tnsr)
        return (
            t.utils.data.DataLoader(training_ds, batch_size=bs, shuffle=True),
            t.utils.data.DataLoader(validation_ds, batch_size=bs * 2),
        )

    def instantiate_model(self, input_size, hidden_size, output_size, learning_rate):
        model = NeuralNet(input_size, hidden_size, output_size)
        return model, t.optim.SGD(model.parameters(), lr=learning_rate)

    def run(self, t_features, t_observations, v_features, v_observations
        ):

        train_dl, valid_dl = self.format_data(t_features, t_observations, 
            v_features, v_observations, self.batch_size
        )
        
        input_size = t_features.shape[1]
        output_size = t_observations.shape[1]
        hidden_size = t_features.shape[1]
        model, optimiser = self.instantiate_model(input_size, hidden_size, output_size, 
            learning_rate=self.learning_rate
        )

        epoch_losses = self.fit(self.epochs, model, self.loss_func, optimiser, train_dl, 
            valid_dl
        )

        return model, epoch_losses

def preprocess_data(X, Y):
    X, Y = impute_missing_values(X, Y, strategy='mean')
    X, Y = standardise(X, Y)
    X, Y = normalise(X, Y) #Centeres about the origin
    return X, Y

def impute_missing_values(X, Y, strategy= 'mean'):
    imputer = SimpleImputer(missing_values=np.nan, strategy=strategy)
    X = imputer.fit_transform(X)
    Y = imputer.fit_transform(Y)
    return X, Y

def standardise(X,Y):
    x_std_sclr = StandardScaler()
    y_std_sclr = StandardScaler()

    X = x_std_sclr.fit_transform(X)
    Y = y_std_sclr.fit_transform(Y)
    return X, Y

def normalise(X, Y, min=-1, max=1):
    x_mms = MinMaxScaler(feature_range=(min, max))
    y_mms = MinMaxScaler(feature_range=(min, max))

    X = x_mms.fit_transform(X)

    # MMS scales and translates each feature individually
    Y = y_mms.fit_transform(Y) 

    return X, Y

def retrieve_subset_of_data(partition = 'training'):
    # Get iKIR Scores
    ikir_scores = data_mgr.feature_values(normalise = False, fill_na = False, 
        fill_na_value = 0.0, partition = partition
    )

    # Read in List of Desired Phenos
    filename = "Data/candidate_phenos_09022023.csv"
    phenos_subset = list(pd.read_csv(filename, index_col=0).values[:, 0])

    # Filter Master Dataset on Desired Subset 
    immunos_maxtrix_subset = data_mgr.outcomes_subset(desired_columns=phenos_subset, 
        partition = partition
    )

    immunos_labels = immunos_maxtrix_subset.columns[1:-2]
    immunos_maxtrix_subset = immunos_maxtrix_subset.values[:,1:-2].astype(float)
    return ikir_scores, immunos_labels, immunos_maxtrix_subset

#Instantiate Controllers
config = cm.ConfigManaager().config
sql = msm.MySQLManager(config=config)
data_mgr = dtm.DataManager(config=config, use_full_dataset=True)
lrn_mgr = lrn.LearningManager(config=config)

# Pull Data from DB
ikir_scores_t, immunos_labels_t, immunos_maxtrix_subset_t = \
    retrieve_subset_of_data(partition = 'training')

ikir_scores_v, immunos_labels_v, immunos_maxtrix_subset_v = \
    retrieve_subset_of_data(partition = 'validation')

# Normalise Data 
ikir_scores_t, immunos_maxtrix_subset_t = preprocess_data(ikir_scores_t, immunos_maxtrix_subset_t)
ikir_scores_v, immunos_maxtrix_subset_v = preprocess_data(ikir_scores_v, immunos_maxtrix_subset_v)

nnr = SimpleNeuralNetRegressor(epochs = 20, batch_size = 64, learning_rate=0.0001)

model, epoch_losses = nnr.run(
    t_features=immunos_maxtrix_subset_t, t_observations=ikir_scores_t, 
    v_features=immunos_maxtrix_subset_v, v_observations=ikir_scores_v
)

print(epoch_losses)