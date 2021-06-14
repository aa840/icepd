# This script gives examples of the main functions available.
# The Boston housing dataset is used as an example
import numpy as np
import random
import pandas as pd
from sklearn.datasets import load_boston
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from statsmodels.formula.api import ols
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm 

import tensorflow as tf                                 
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras import regularizers, Input, initializers
from tensorflow.keras.models import load_model

from test_partial_dep import test_pd
from test_ice import test_ice
from test_interactions import test_interactions

from icepd.data_struct import Ensemble

def model_init(N):
    l = 0.05
    # Construct a simple feedforward NN with one hidden layer 
    model = Sequential()
    model.add(Input(shape=(N,)))
    # Initializing weights at zero is much better as expect sparse solution
    model.add(Dense(((10)), activation='sigmoid'))
    model.add(Dense(1, activation='linear', kernel_regularizer=regularizers.l1(l)))
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy']) 
    return model

#Load example data
X, y = load_boston(return_X_y=True)
bost_dic = load_boston()

# Change to pd dataframe, center and scale to unit std
X = pd.DataFrame(scale(X))
y = pd.DataFrame(scale(y))
X.columns = bost_dic['feature_names']

# OHE RAD variable
one_hot = pd.get_dummies(X['RAD'], prefix=('RAD--'))
X = X.copy().drop(columns = 'RAD')
X = X.merge(one_hot, left_index=True, right_index=True, how='left')

new_c = []

for c in X.columns:
    new_c.append(c.replace('_',''))

X.columns = new_c
    
# NN initalized 
model = model_init(len(X.columns))

# Create and fit an ensemble model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
model_ensem = Ensemble(model)                   

# The model ensemble can also be set with an existing array of regression models
model_array = [load_model('./Models/model_ensem_Boston_OHE_' + str(i) + '.h5') for i in range(0,10)]
model_ensem.set_models_array(model_array)

print('Starting tests for interactions')
# Test interactions
test_interactions(X,y, model_ensem, './Graphs/')
print('Finished tests for interactions')

# Smaller X for tests
X = X.iloc[:100,:]
y = y.iloc[:100]

print('Starting tests for PD')
# Test PD
test_pd(X,y, model_ensem, './Graphs/')
print('Finished tests for PD')

print('Starting tests for ICE')
# Test ICE
test_ice(X,y, model_ensem, './Graphs/')
print('Finished tests for ICE')

print('All tests completed.')
print('Graphs produced can be viewed in Graphs folder.')
