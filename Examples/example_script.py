import numpy as np
import pandas as pd
import pickle

import sklearn.datasets 
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

from sklearn.metrics import r2_score

import matplotlib
import matplotlib.pyplot as plt

from icepd.interactions import OLS_interactions, l1_interactions
from icepd.partial_dep import pd_plot, pd_hstat_compare, plot_interactions, pd_hstat, pd_hstat_null, pd_OHE_1D_plot, plot_format

from icepd.ice import ice_plot
from icepd.data_struct import Ensemble
from sklearn.kernel_ridge import KernelRidge

data = sklearn.datasets.load_diabetes()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.DataFrame(data.target, columns=['Progression'])

# KRR initalized 
model = KernelRidge(kernel='rbf', alpha=0.01, gamma=0.1)

# Create and fit an ensemble model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
model_ensem = Ensemble(model) # When initializing pass Ensemble class the model type                  
model_ensem.fit(X_train, y_train, N=10) # Number of items in ensemble

print('Training error ' + str(abs(y_train.values.ravel() - model_ensem.predict(X_train).ravel()).mean()))
print('Testing error ' + str(abs(y_test.values.ravel() - model_ensem.predict(X_test).ravel()).mean()))

print('R2 ' + str(r2_score(y_test.values.ravel(), model_ensem.predict(X_test).ravel())))

plot_format()
plt.scatter(y_test.values.ravel(), model_ensem.predict(X_test).ravel())
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()


# Save ensemble
model_ensem.save('./Models/model_ensem_Diabetes_KRR_')

# The model ensemble can also be set with an existing array of regression models
model_array = [pickle.load(open('./Models/model_ensem_Diabetes_KRR_' + str(i) + '.h5', 'rb')) for i in range(0,10)]
model_ensem.set_models_array(model_array)


# Plot 1D PD plot, unscaled
pdp = pd_plot(X, y, [2], model_ensem, center='N', ensemble='Y', grid_size = 0.05, linear='N', grad='N', filename='./Graphs/Diabetes_', show='Y')
plt.show()

# Plot 1D PD plot for OHE variable
pdp = pd_plot(X,y, [1], model_ensem, ensemble='Y', filename='./Graphs/', show='Y')
plt.show()


# Plot 2D PD plot
plt.close()
pdp = pd_plot(X, y, [2,3], model_ensem, ensemble='Y', filename='./Graphs/', show='Y')  
plt.show()

ice = ice_plot(X, [0], model_ensem.models_array[0], ensemble='N', grad='N', alpha=0.5, center='Y', filename='./Graphs/', show='Y')
plt.show()

h_stat_one, pd_dict = pd_hstat([1,2], X, model_ensem, N=400)

# Null distribution H stats
mean_h_null_one, std_h_null_one, h_null_one, art_data= pd_hstat_null(X, y, model_ensem.models_array[0], features=[0,1], N_pd=400, N_null=10)

print('The H-statistic for the actual dataset: ' + str(h_stat_one))
print('The H-statistic for the null dataset: ' + str(mean_h_null_one) + ' with std. ' + str(std_h_null_one))

# Interactions for all variables
h_stat, h_stat_null_m, h_stat_null_std = pd_hstat_compare(X,y, model_ensem, N=400) # The number of points sampled for the H-statistic can be set here 

print(h_stat_null_m)
print(h_stat_null_std)

print(h_stat)

# Subtract the null value and set those less than 4 std. from mean to zero
params = h_stat - h_stat_null_m
params[(params) < 4*h_stat_null_std] = 0 

plot_interactions(params, X, 'H-Stat', show='Y')
plt.show()
