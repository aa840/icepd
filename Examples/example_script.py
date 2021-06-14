# This script gives examples of the main functions available.
# The Boston housing dataset is used as an example

import numpy as np
import pandas as pd
import pickle

from sklearn.datasets import load_boston
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

import matplotlib
import matplotlib.pyplot as plt

from icepd.interactions import OLS_interactions, l1_interactions
from icepd.partial_dep import pd_plot, pd_hstat_compare, plot_interactions, pd_hstat, pd_hstat_null, pd_OHE_1D_plot

from icepd.ice import ice_plot
from icepd.data_struct import Ensemble
from sklearn.kernel_ridge import KernelRidge

#Load example data
X, y = load_boston(return_X_y=True)
bost_dic = load_boston()

# Change to pd dataframe, center and scale to unit std
[mean_x4, std_x4] = [X[:,4].mean(), X[:,4].std()] 
[mean_y, std_y] = [y.mean(), y.std()] 
X = pd.DataFrame(scale(X))
y = pd.DataFrame(scale(y))
X.columns = bost_dic['feature_names']

# OHE RAD variable
one_hot = pd.get_dummies(X['RAD'], prefix=('RAD'), prefix_sep=' ')
X = X.copy().drop(columns = 'RAD')
X = X.merge(one_hot, left_index=True, right_index=True, how='left')

X.columns = [c.replace('_', ' ')[:10] for c in X.columns]

# NN initalized 
model = KernelRidge(kernel='rbf', alpha=0.01, gamma=1)

# Create and fit an ensemble model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
model_ensem = Ensemble(model) # When initializing pass Ensemble class the model type                  
model_ensem.fit(X, y, N=10) # Number of items in ensemble

# Save ensemble
model_ensem.save('./Models/model_ensem_Boston_KRR_')

# The model ensemble can also be set with an existing array of regression models
model_array = [pickle.load(open('./Models/model_ensem_Boston_KRR_' + str(i) + '.h5', 'rb')) for i in range(0,10)]
model_ensem.set_models_array(model_array)

# Plot 1D PD plot and gradients of PD
pd = pd_plot(X, y, [4], model_ensem, ensemble='Y', linear='N', grad='Y', filename='./Graphs/')

# Plot 1D PD plot, unscaled
pd = pd_plot(X, y, [4], model_ensem, ensemble='Y', linear='N', grad='Y', filename='./Graphs/Unscaled_', unscale_x=[mean_x4, std_x4], unscale_y=[mean_y, std_y])

# Plot 1D PD plot for OHE variable
pd = pd_OHE_1D_plot(X,y, [12,13,14,15,16,17,18,19,20], model_ensem, ensemble='Y', linear='N', filename='./Graphs/', xtick_labels=['a','b','c','d','e','f','g', 'h', 'i'])

# Plot 2D PD plot
pd = pd_plot(X, y, [3,4], model_ensem, ensemble='Y', linear='Y', filename='./Graphs/')  

# Plot ICE graph and gradients of ICE for one NN model
ice_plot(X, [4], model_ensem.models_array[0], ensemble='N', grad='Y', filename='./Graphs/')

# Interaction for pair of variables
# H stat between [3,4]
h_stat, pd_dict = pd_hstat([3,4], X, model_ensem.models_array[0], N=200)
# Null distribution H stats
mean_h_null, std_h_null, h_null, art_data= pd_hstat_null(X, y, model_ensem.models_array[0], N_null=5, features=[3,4])


# Interactions for all variables
h_stat, h_stat_null, h_stat_null_std = pd_hstat_compare(X,y, model_ensem, N=50) # The number of points sampled for the H-statistic can be set here 
plot_interactions(h_stat - h_stat_null, X, 'H-Stat')

# OLS/ANOVA Interactions
params_OLS = OLS_interactions(X,y, pvalue=0.01)
plot_interactions(params_OLS, X, filename='./Graphs/OLS')

# l1 Interactions
params_l1 = l1_interactions(X,y, 0.05)
plot_interactions(params_OLS, X, filename='./Graphs/l1')
