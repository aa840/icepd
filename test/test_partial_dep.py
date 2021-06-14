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

from icepd.interactions import OLS_interactions, l1_interactions
from icepd.partial_dep import pd_plot, pd_hstat_compare, plot_interactions, pd_hstat, pd_hstat_null, pd_OHE_1D_plot
from icepd.data_struct import *


def test_pd(X,y,model_ensem, filename):

    # No ensemble
    pd_plot(X, y, [4], model_ensem.models_array[0], ensemble = 'N', linear='Y', grad='Y', filename=filename)
    
    # Plot 1D PD plot and gradients of PD
    pd_plot(X, y, [4], model_ensem, ensemble='Y', linear='Y', grad='Y', filename=filename)

    # No linear, no gradients
    pd_plot(X, y, [4], model_ensem.models_array[0], ensemble = 'N', linear='N', grad='N', filename=filename)

    # OHE PD
    pd_OHE_1D_plot(X,y, [12,13,14,15,16,17,18,19,20], model_ensem, ensemble='Y', linear='Y', filename='./Graphs/', xtick_labels=['a','b','c','d','e','f','g', 'h', 'i'])
    
    # Plot 2D, No ensemble
    pd_plot(X, y, [1,4], model_ensem, ensemble='N', linear='Y', filename=filename)  
    
    # Plot 2D PD plot
    pd_plot(X, y, [1,4], model_ensem, ensemble='Y', linear='Y', filename=filename)

    # H stat between [3,4]
    h_stat = pd_hstat([3,4], X, model_ensem.models_array[0], N=100)
    # Null distribution H stats
    mean_h_null, std_h_null, h2_null, art_data = pd_hstat_null(X, y, model_ensem.models_array[0], N_null=5, features=[3,4])

