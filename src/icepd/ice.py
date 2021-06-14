# The ICE functions 
import numpy as np
import random
import pandas as pd
from sklearn import datasets, linear_model

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm 

from sklearn.utils import resample
from icepd.partial_dep import calc_pdp_grad, pd_xs, plot_format


def ice_plot(X, features, reg, ensemble='N', grid_size=None, grad='N', filename='./', N = None, alpha = None, center = None, unscale_x = None, show='Y'):
    # Plot the ice graph
    # Gradients plotted if grad = Y
    # alpha controls the transparency of the lines

    plt.figure(figsize=(7.5, 6.1))
    
    if grid_size is None:
        grid_size = (X.iloc[:,features[0]].max() - X.iloc[:,features[0]].min())/10

    
    if X.iloc[:,features[0]].nunique() < 20:
        grid_x = X.iloc[:,features[0]].unique()
        grid_x.sort()
    else:
        grid_x = np.arange(X.iloc[:,features[0]].min() - 0.05, (X.iloc[:,features[0]].max() +0.05), grid_size)
            
    ice = []
    ice_std = []
    ice_grad = []
    pdp_local = []

    for j in range(0,len(grid_x)):
        if ensemble == 'Y':
            # All ice points at point grid_x[j] are calculated for all models
            ice_tmp = []
            for r in reg.models_array:
                pd_zero = pd_xs(features[0], X.iloc[:,features[0]].mean(), X.copy(), r) # PD at zero is calculated to shift all ICE values

                if N is not None:
                    ice_tmp.append(ice_xs(features, grid_x[j], X.copy().iloc[:N,:], r) - pd_zero)
                else:
                    ice_tmp.append(ice_xs(features, grid_x[j], X.copy(), r) - pd_zero)
            ice_tmp = np.array(ice_tmp)

            # Mean model value is then plotted
            ice.append(ice_tmp.mean(axis=0)) 
            # Std of models is also recorded
            ice_std.append(ice_tmp.std(axis=0))
        else:
            pd_zero = pd_xs(features[0], X.iloc[:,features[0]].mean(), X.copy(), reg) # PD at zero is calculated to shift all ICE values
            ice_tmp = ice_xs(features, grid_x[j], X.copy().iloc[:N, :], reg)
            ice.append(ice_tmp - pd_zero) 
 
    ice = np.array(ice)

    # Pandas array can sometimes be 3D, try change
    try:
        ice = ice[:,:,0]
    except:
        pass
    
    if center == 'Y':
        if N is None:
            ice_zero = ice_xs(features, X.iloc[:,features[0]].mean(), X.copy().iloc[:,:], reg) 
        else:
            ice_zero = ice_xs(features, X.iloc[:,features[0]].mean(), X.copy().iloc[:N,:], reg)

#        ice = ice - ice_zero

        # For all ensemble models remove ice_zero
        for n in range(0,len(ice[:,0])):
            try:
                ice[n,:] = ice[n,:] - ice_zero[:,0]
            except:
                ice[n,:] = ice[n,:] - ice_zero[:]
                
            ice[n,:] = ice[n,:] + pd_zero

    no_inc = len(X)

    if N is None:
        N = len(X)

    if alpha is None:
        alpha = max(0.01, (50.0/N))
        alpha = min(alpha, 1)
        
    # Plot ICE Graph
    if unscale_x is None:
        plt.plot(grid_x,ice[:,:no_inc], 'b', linewidth=0.10, alpha=alpha) # Transparency will increase as more data points are added
    else:
        plt.plot(((grid_x*unscale_x[1]) + unscale_x[0]),ice[:,:no_inc], 'b', linewidth=0.10, alpha=alpha) # Transparency will increase as more data points are added
        
    plt.xlabel(str(X.columns[features[0]]))
    plt.ylabel('ICE')
    plot_format()
    plt.tight_layout()
    plt.savefig(filename + 'ICE_plot_' + str(features[0]) + '.pdf')

    if show !='Y':
        plt.close()

    if grad == 'Y':
        plt.close()

        # Calculates the gradients of the ICE
        for k in range(0,len(ice[0,:])):
            grad_mean, grid_mid = calc_pdp_grad(ice[:,k], grid_x)
            ice_grad.append(grad_mean)

        ice_grad = np.array(ice_grad)

        # Plot ICE gradient graph
        plt.plot(grid_mid[:],ice_grad[:no_inc,:].T, 'b', alpha=min(50.0/N, 1)) 
        plt.xlabel(str(X.columns[features[0]]).replace('_', '-'))
        plt.ylabel('Gradient')
        plot_format()
        plt.tight_layout()
        plt.savefig(filename + 'ICE_plot_grads_' + str(features[0]) + '.pdf') 


        if show !='Y':
            plt.close()
    
        
    return ice


def ice_xs(i, xs, X, clf):
    X.iloc[:,i] = xs
    ice = clf.predict(X)
    return ice

