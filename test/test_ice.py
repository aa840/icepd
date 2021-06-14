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

from icepd.partial_dep import pd_plot, pd_hstat_compare, plot_interactions, pd_hstat, pd_hstat_null
from icepd.data_struct import Ensemble
from icepd.ice import ice_plot

def test_ice(X,y, model_ensem, filename):
    # Plot ice 
    ice_plot(X, [4], model_ensem, ensemble='Y', grad='Y', filename=filename)

    # Without ensemble and gradient
    ice_plot(X, [4], model_ensem.models_array[0], ensemble='N', grad='N', filename=filename)

    
