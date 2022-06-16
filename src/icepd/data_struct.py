# Datastructure classes for handling the artificial data and model ensemble
import numpy as np
import random
import pandas as pd
import datetime

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.utils import resample
from tensorflow.keras.models import load_model

import os
import pickle


class Ensemble:
    def __init__(self, reg):
        self.models_array = []
        self.model_func = reg

    def set_models_array(self, models_array):
        # Sets the models array with an array of prefit models
        self.models_array = models_array

    def save(self, name):
        n = 0 
        for m in self.models_array:
            try:
                m.save(name + str(n) + '.h5')
            except:
                pickle.dump(m, open(name + str(n) + '.h5', 'wb'))
            n = n + 1

            
    def predict(self, X):
        predict_array = []
        
        # Predict for the entire ensemble
        for r in self.models_array:
            predict_array.append(r.predict(X))

        return np.array(predict_array).mean(axis=0)

        
    def fit(self, X,y, N = 25, epochs=50, batch_size=20, verbose=1, bootstrap = 'Y', callbacks=None):
        # Fit an ensemble of models to the bootstrapped data
        # N is the number of models in the ensemble

        date = datetime.datetime.now().date()  
            
        self.models_array = []
        n = 0

        for i in range(N):
            # Save the regression model so weights are reintialized
            date = str(datetime.datetime.now().date()) + '_' + "{:.0f}".format(np.random.uniform()*10000)

            try:
                self.model_func.save('tmp_' + str(date) + '.h5') 
                reg = load_model('tmp_' + str(date) + '.h5')
            except:
                pickle.dump(self.model_func, open('tmp_' + str(date) + '.h5', 'wb'))
                reg = pickle.load(open('tmp_' + str(date) + '.h5', 'rb'))
            
            if bootstrap == 'Y':
                # Bootstrap resample
                train = resample(X, random_state=i)
            else:
                train = X
            
            # Try to fit for NN, otherwise fit without settings
            try:
                if callbacks is None:
                    reg.fit(train, y.loc[train.index], epochs=epochs, batch_size=batch_size, verbose=verbose)
                else:
                    reg.fit(train, y.loc[train.index], epochs=epochs, batch_size=batch_size, verbose=verbose, callback=callbacks)
            except:
                reg.fit(train, y.loc[train.index].values.ravel())
   
            self.models_array.append(reg)
            n = n + 1

            os.remove(('tmp_' + str(date) + '.h5'))
                
                

class Artificial_data:
    # Artificial data for H stat, pass data and artifical data generated
    def __init__(self, X, y, N_null):
        self.y = pd.DataFrame(calc_artif_y(X,y, N_null=N_null))
        self.X = X        
        
def calc_artif_y(X,y, N_null=10):
    # Calculates y for the artificial data

    # GBR used with max depth one to give an additive model with NL present
    #  The parameters of this may want to be tuned
    gb = GradientBoostingRegressor(n_estimators=100, max_depth=1, loss='squared_error', random_state=1)
    gb_gs = GridSearchCV(gb,
                         cv=5, n_jobs=1, verbose=1,
                       param_grid={"learning_rate": np.logspace(-2, 0, 5, base=10),
                                   "min_samples_leaf": np.arange(2, 35, 5) })
    gb_gs.fit(X,y.values.ravel())
    print('Finished fitting artificial data')
    
    # Additive model X
    fa_x = gb_gs.predict(X)

    y_art_array = []
    for i in range(0,N_null):
        # Random permutation of integers
        p_i = np.random.permutation(len(X))
        y_pi = y.iloc[p_i]
        fa_pi = gb_gs.predict(X.iloc[p_i])

        # Artificial data
        y_art = fa_x + (y_pi.values.ravel() - fa_pi)
        y_art_array.append(np.array(y_art.ravel()))
        
    return np.array(y_art_array)
    
