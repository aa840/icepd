#Testing of the interactions graphs

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
from icepd.partial_dep import pd_plot, pd_hstat_compare, plot_interactions, pd_hstat, pd_hstat_null
from icepd.data_struct import *

def test_interactions(X,y, model_ensem, filename):
    # OLS/ANOVA Interactions
    params_OLS = OLS_interactions(X,y, pvalue=0.01)
    plot_interactions(params_OLS, X, filename = filename + '/OLS_')

    # l1 Interactions
    params_l1 = l1_interactions(X,y, 0.05)
    plot_interactions(params_OLS, X,  filename = filename + '/l1_')


