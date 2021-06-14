# The PD functions, including H-stats 
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.utils import resample

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.style as style

from matplotlib import cm 

from icepd.data_struct import *
import datetime

import os


def pd_xs(i, xs, X, reg):
    # PD calculated for one variable
    X.iloc[:,i] = xs
    pd = reg.predict(X).mean() 
    return pd



def pd_xs2(i, j, xsi, xsj, X, reg):
    # PD calculated for two variables
    X.iloc[:,i] = xsi
    X.iloc[:,j] = xsj
    pd = reg.predict(X).mean() 
    return pd

def pd_ohe(i, xs_ohe, X, reg):
    # PD calculated for OHE values
    X.iloc[:,i] = X.iloc[:,i].max() # This is the 1 value of OHE

    # Need to set all other OHE values to 0
    for xs in xs_ohe:
        if xs != i:
            X.iloc[:,xs] = X.iloc[:,xs].min() # This is the 0 value of OHE

    pd = reg.predict(X).mean() 
    return pd

def pd_xs2_ohe(i, j, xi_ohe, xj_ohe,  X, reg):
    # PD calculated for OHE values
    X.iloc[:,i] = X.iloc[:,i].max() # This is the 1 value of OHE
    X.iloc[:,j] = X.iloc[:,j].max() # This is the 1 value of OHE

    # Need to set all other OHE values to 0, xi_ohe
    for xs in xi_ohe:
        if xs != i:
            X.iloc[:,xs] = X.iloc[:,xs].min() # This is the 0 value of OHE

    # Need to set all other OHE values to 0
    for xs in xj_ohe:
        if xs != j:
            X.iloc[:,xs] = X.iloc[:,xs].min() # This is the 0 value of OHE

            
    pd = reg.predict(X).mean() 
    return pd


def pd_grid(reg, X, features, grid):
    # Calculates PD for a grid of points

    pd_array = np.zeros((len(grid), len(grid[0])))

    for i in range(0,len(pd_array)):
        for j in range(0, len(pd_array[0])):
            pd_array[i,j] = pd_xs2(features[0], features[1], grid[i,j,0], grid[i,j,1], X.copy(), reg)

    return pd_array

def pd_hstat(features, X, reg, N=None, prev_dict=None,  ohe_j=None, ohe_k=None):
    # Returns sqrt of H2
    h2_stat, prev_dict = pd_hstat2(features, X, reg, N=N, prev_dict=prev_dict, ohe_j=ohe_j, ohe_k=ohe_k)
    return np.sqrt(h2_stat), prev_dict


def pd_hstat2(features, X, reg, N=None, prev_dict=None,  ohe_j=None, ohe_k=None):
    # Calculates the PD H-stat**2, a measure of the interactions present between the two variables contained in features
    # N is the number of datapoints considered when calculating the H-stat
    # prev_dict contains previously calculated values, this can speed up calc. by factor of 3
    # All are centered as in Friedman, so F(x) has mean 0
    
    if N is None:
        N = len(X.index)

    if N > len(X):
        print('N cannot be larger than the size of the dataset. N is now len(X).')
        N = len(X.index)
        
       
    pd_jk = np.zeros(N)
    pd_j = np.zeros(N)
    pd_k = np.zeros(N)

    # Dict speeds up computation considerably for discrete variables
    prev_values_xsj = {}
    prev_values_xsk = {}
    prev_values_xsjk = {}

    if prev_dict is not None:
        if 'Feature ' + str(features[0]) in prev_dict.keys():
            prev_values_xsj = prev_dict['Feature ' + str(features[0])]
        if 'Feature ' + str(features[1]) in prev_dict.keys():
            prev_values_xsk = prev_dict['Feature ' + str(features[1])]
        if 'Mean PD' in prev_dict.keys():
            mean_pd = prev_dict['Mean PD'] 
        else:
            mean_pd =reg.predict(X.iloc[:N,:]).mean() # mean of function 

    else:
        prev_dict = {}    
        mean_pd =reg.predict(X.iloc[:N,:]).mean() # mean of function 

        
    for i in range(0,N):
        xsj = X.iloc[i,features[0]]
        xsk = X.iloc[i,features[1]]

        # Adds to dictionary if not previously calculated
        if (xsj, xsk) not in prev_values_xsjk.keys():
            if ohe_j is None:
                prev_values_xsjk[(xsj,xsk)] = pd_xs2(features[0], features[1], xsj, xsk, X.copy().iloc[:N, :], reg)
            else:
                prev_values_xsjk[(xsj,xsk)] = pd_xs2_ohe(features[0], features[1], ohe_j, ohe_k, X.copy().iloc[:N, :], reg)

        if xsj not in prev_values_xsj.keys():
            if ohe_j is None:
                prev_values_xsj[xsj] =  pd_xs(features[0], xsj, X.copy().iloc[:N, :], reg)
            else:
                prev_values_xsj[xsj] =  pd_ohe(features[0], ohe_j, X.copy().iloc[:N, :], reg)
                
        if xsk not in prev_values_xsk.keys():
            if ohe_k is None:
                prev_values_xsk[xsk] =  pd_xs(features[1], xsk, X.copy().iloc[:N, :], reg)
            else:
                prev_values_xsk[xsk] =  pd_ohe(features[1], ohe_k, X.copy().iloc[:N, :], reg)
                
        pd_jk[i] = prev_values_xsjk[(xsj,xsk)]
        pd_j[i] = prev_values_xsj[xsj]
        pd_k[i] = prev_values_xsk[xsk]

        
    pd_jk = pd_jk - mean_pd #Set mean of pd to 0
    pd_j = pd_j - mean_pd #Set mean of pd to 0
    pd_k = pd_k - mean_pd #Set mean of pd to 0

    
    # Prevents division by zeros
    if (pd_jk**2).sum() == 0.00 :
        h2_stat = 0.00
    else:
        h2_stat = ((pd_jk - pd_j - pd_k)**2).sum()/ (pd_jk**2).sum() 

    # If there is no variety in the dataset the h-stat will go to 1.0, to prevent, below sets it to 0.0
    if  X.iloc[:,features[0]].std() * X.iloc[:,features[1]].std() == 0.0:
        h2_stat = 0.0
        
    # Update dictionary 
    prev_dict['Feature ' + str(features[0])] = prev_values_xsj
    prev_dict['Feature ' + str(features[1])] = prev_values_xsk
    prev_dict['Mean PD'] = mean_pd
        
    # The squared value is returned, as is dictionary
    return h2_stat, prev_dict

def pd_hstat_null(X, y, reg, N_null=5, epochs=200, batch_size=20, features=None, N_pd=None, art_data=None):
    # PD H-stat for the null distribution
    # N_null is the number of points found for the null distribution

    if N_pd is None:
        N_pd = len(X)
    
    # Artificial data described in Friedman, contains NL but no interactions
    if art_data is None:
        art_data = Artificial_data(X,y,N_null)
    
    h_stat_array = []

 #   print(art_data.y)

    for i in range(0,N_null):
        h_stat2_n = pd_art_data(art_data.X, art_data.y.iloc[i,:], reg, epochs=epochs, batch_size=batch_size, features=features, N_pd=N_pd)
        h_stat_array.append(np.sqrt(h_stat2_n)) # Sqrt of H2 to H

    h_stat_array = np.array(h_stat_array)    
    std_h_stat = h_stat_array.std(axis=0)

    # Mean H stat, std and full array are returned
    return h_stat_array.mean(axis=0), std_h_stat, h_stat_array, art_data


def pd_art_data(X, y, reg_orig, epochs=200, batch_size=20, features=None, ohe_j=None, ohe_k = None, N_pd=None):
    # Returns PD for artificial data
    # Do not pass ensemble here
    # Pass full dataset
    # N_pd is the number of points for the PD

    date = str(datetime.datetime.now().date()) + '_' + "{:.0f}".format(np.random.uniform()*10000)

    # Reinitialize model so weights are not retained 
    try:
        reg_orig.save('tmp_' + str(date) + '.h5') 
        reg = load_model('tmp_' + str(date) + '.h5')
    except:
        pickle.dump(reg_orig, open('tmp_' + str(date) + '.h5','wb'))
        reg = pickle.load(open('tmp_' + str(date) + '.h5','rb'))

    # Fit regression model to artifical data, if NN set epochs and batch size 
    try:
        reg.fit(X, y, epochs=epochs, batch_size=batch_size)
    except:
        reg.fit(X, y)

    # Find h stat for all interactions with model fit to art data
    h_stat2_n = []

    # Initialize prev dict
    prev_dict = {}
    
    if features == None:
        for i in range(0,len(X.iloc[0,:])):
            for j in range(0,i):
                f = (i,j)
#                print(f)
                h_stat2, prev_dict = pd_hstat2(f, X.iloc[:N_pd, :], reg, prev_dict=prev_dict)
                h_stat2_n.append(h_stat2)
    else:
        h_stat2, _ = pd_hstat2(features, X.iloc[:N_pd, :], reg, ohe_j=ohe_j, ohe_k=ohe_k)
        h_stat2_n = h_stat2            
                
    os.remove('tmp_' + str(date) + '.h5') 

    return np.array(h_stat2_n)


def pd_hstat_real(X, reg, N=None):
    # PD for real data

    # Dict. stores prev calculated PD
    prev_dict = {}
    
    h2_stat_array = []
    for i in range(0,len(X.columns)):
        for j in range(0,i):
            features = (i,j)
#            print(features)
            h2_stat, prev_dict = pd_hstat2(features, X, reg, prev_dict=prev_dict, N=N) #PD H2 returned
            h2_stat_array.append(h2_stat)

    h2_stat_array = np.array(h2_stat_array)   
    h_stat_array = np.sqrt(h2_stat_array) # Change from H2 to H
 
    return h_stat_array

def pd_hstat_compare(X, y, reg, ensemble='Y', N_null=10, N=None):
    # Computes the actual and null distributions for the H-stat
    # Reg is the regression method
    # N_null is the number of smaples drawn from the null distribution
    # N is the number of samples used for the H-stat

    
    # Returns the h stat for the real data 
    h_stat_actual = pd_hstat_real(X,reg, N=N)
    
    # Start by calculating null stats, std returned for h stat not h2
    if ensemble == 'Y':
        h_stat_null_m, h_stat_null_std, h_stat_array, art_data = pd_hstat_null(X, y, reg.models_array[0], N_null=N_null, N_pd=N)
    else:
        h_stat_null_m, h_stat_null_std, h_stat_array, art_data = pd_hstat_null(X, y, reg, N_null=N_null, N_pd=N)

                                                                               
    return h_stat_actual, h_stat_null_m, h_stat_null_std


def pd_plot_2D(X, features, reg, ensemble='N', linear='N', y=None, grid_size=0.25, filename='./', show=None, binar=None):
    # PD plot for 2D graphs

    # Check first there are sufficient values
    if X.iloc[:, features[0]].nunique() == 1:
        print('There is only one unique value for feature ' + str(features[0]) + ' a 2D PD plot is not possible.')
        return
    if X.iloc[:, features[1]].nunique() == 1:
        print('There is only one unique value for feature ' + str(features[1]) + ' a 2D PD plot is not possible.')
        return 
        
    # Produce grid for plots, if discrete variable then limit points
    if X.iloc[:, features[0]].nunique() < 20:
        grid_x = X.iloc[:,features[0]].unique()
        grid_x.sort()
    else: 
        grid_x = np.arange(X.iloc[:,features[0]].min(), (X.iloc[:,features[0]].max() + 0.05), grid_size)

    if X.iloc[:,features[1]].nunique() < 20:
        grid_y = X.iloc[:,features[1]].unique()
        grid_y.sort()
    else: 
        grid_y = np.arange(X.iloc[:,features[1]].min(), (X.iloc[:,features[1]].max() + 0.05), grid_size)

    N_x = len(grid_x)
    N_y = len(grid_y)
    grid = np.zeros((N_x, N_y,2))

    for i in range(0, N_x):
        for j in range(0,N_y):
            grid[i,j] = [grid_x[i], grid_y[j]]

    pdp = pd_grid(reg, X.copy(), features, grid)
    plot_format()
    plt.contourf(pdp, 20, extent=[grid_y.min(), grid_y.max(), grid_x.min(), grid_x.max()], cmap=plt.get_cmap('viridis'), origin='lower') 
    plt.xlabel(str(X.columns[features[1]]).replace('_', '-'))
    plt.ylabel(str(X.columns[features[0]]).replace('_', '-'))
    plt.colorbar()
#    plt.tight_layout()
    plot_format()
    plt.savefig(filename + '/PD_plot_' + str(features[0]) + '_' + str(features[1]) + '.pdf',bbox_inches='tight')

    if show != 'Y':
        plt.close()

    # If first target feature is binary then plot PD lines
    if X.iloc[:,features[0]].nunique() == 2 and binar == 'Y':

        plt.close()
        
        if ensemble == 'Y': 
            # For ensemble
            no_models = len(reg.models_array)
            n = 0
            pdp = []
            for r in reg.models_array:
                pdp_zero = pd_xs(features[1], X.iloc[:,features[1]].mean(), X.copy(), r) # PD at mean
                pdp.append(pd_grid(r, X.copy(), features, grid) - pdp_zero)                        
            pdp = np.array(pdp)
                
            # Mean PDP, with upper and lower uncertain for first variable
            mean_pdp = pdp[:,0,:].mean(axis=0)
            min_pdp = mean_pdp - 1.96*pdp[:,0,:].std(axis=0)
            max_pdp = mean_pdp + 1.96*pdp[:,0,:].std(axis=0)

            label_name = str(X.columns[features[0]]) + ' = ' + '%.2f' % grid_x[0]

            plt.plot(grid_y, mean_pdp, 'C0-' , label = label_name)
            plt.plot(grid_y, max_pdp, 'C0--')
            plt.plot(grid_y, min_pdp, 'C0--')

            # Mean PDP, with upper and lower uncertain for first variable
            mean_pdp = pdp[:,1,:].mean(axis=0)
            min_pdp = mean_pdp - 1.96*pdp[:,1,:].std(axis=0)
            max_pdp = mean_pdp + 1.96*pdp[:,1,:].std(axis=0)

            label_name = str(X.columns[features[0]]) + ' = ' + '%.2f' % grid_x[1]

            plt.plot(grid_y, mean_pdp, 'C2-', label = label_name)
            plt.plot(grid_y, max_pdp, 'C2--')
            plt.plot(grid_y, min_pdp, 'C2--')

        else:
            # For non-ensemble
            n = 0 
            for g in grid_x:
                label_name = str(X.columns[features[0]]) + ' = ' + '%.2f' % g
                plt.plot(grid_y, pdp[n,:], label = label_name)
                n = n + 1

        plt.xlabel(str(X.columns[features[1]]))
        plt.ylabel('Partial Dependence')
        plt.legend()
#        plt.tight_layout()
        plot_format()
        plt.savefig(filename + '/PD_plot_lines_bin_' + str(features[0]) + '_' + str(features[1]) + '.pdf',bbox_inches='tight')

        if show != 'Y':
            plt.close()


def pd_plot(X,y, features, reg, ensemble='N', linear='N', grid_size=None, type_reg_model='NN', grad='N', filename = './', unscale_x = None, unscale_y = None, center='Y', lin_alpha=0.000001, log_y='Y', show=None, binar=None):
    # Plots the partial dependence with the distribution of variables shown
    # Pass target features as array
    # If ensemble = 'Y', uncertainity for PD shown 
    # If linear = 'Y', the linear solution is also shown and y must be passed
    # Type of regression, gives legend name
    # If grad is Y the gradient is also plotted
    # Unscale is used to unstandardize a variable, [mean, std.]

    plot_format()
    
    if grid_size is None:
        grid_size = (X.iloc[:,features[0]].max() - X.iloc[:,features[0]].min())/10
    
    if len(features) == 2: # If 2D plot        
        pdp = pd_plot_2D(X, features, reg, ensemble=ensemble, linear=linear, y=None, grid_size=grid_size, filename=filename, show=show, binar=binar)
    if type(features) == int or len(features) == 1: # If 1D plot
        # 1D PD Plot

        # First check more than one unique value
        if X.iloc[:, features[0]].nunique() == 1:
            print('There is only one unique value for feature ' + str(features[0]) + ' a 1D PD plot is not possible.')
            return

        # One target feature

        if X.iloc[:,features[0]].nunique() < 20:
            grid_x = X.iloc[:,features[0]].unique()
            grid_x.sort()
        else:
            grid_x = np.arange(X.iloc[:,features[0]].min() - 0.05, (X.iloc[:,features[0]].max() +0.05), grid_size)
            
        if ensemble == 'Y':

            no_models = len(reg.models_array)
            pdp = np.zeros((len(grid_x), no_models))
            n = 0
            for r in reg.models_array:

                if center == 'Y':
                    if X.iloc[:,features[0]].nunique() == 2 :
                        pdp_zero = pd_xs(features[0], X.iloc[:,features[0]].min(), X.copy(), r) 
                    else:
                        pdp_zero = pd_xs(features[0], X.iloc[:,features[0]].mean(), X.copy(), r) # PD at mean
                else:
                    pdp_zero = pd_xs(features[0], X.iloc[:,features[0]].mean(), X.copy(), reg) # Mean for whole ensemble
                        
                for j in range(0, len(pdp)):
                    pdp[j, n] = pd_xs(features[0], grid_x[j], X.copy(), r) - pdp_zero

                n = n +1

        else:
            no_models = 10
            pdp = np.zeros(len(grid_x))
            
            if center == 'Y':
                if X.iloc[:,features[0]].nunique() == 2 :
                    pdp_zero = pd_xs(features[0], X.iloc[:,features[0]].min(), X.copy(), reg) # pdp at zero (mean)
                else: 
                    pdp_zero = pd_xs(features[0], X.iloc[:,features[0]].mean(), X.copy(), reg) # pdp at zero (mean)

                    
            else:
                pdp_zero = 0.0
                
            for j in range(0, len(grid_x)):
                pdp[j] = pd_xs(features[0], grid_x[j], X.copy(), reg) - pdp_zero

        pdp = np.array(pdp)
                
        # Add linear solution 
        pdp_linear = np.zeros((len(grid_x), no_models))
#        lin_reg = linear_model.LinearRegression()
        lin_reg = linear_model.Lasso(alpha=lin_alpha)
        n = 0
        
        # Set number of resampling
        if ensemble == 'Y':
            n_iterations = no_models
        else:
            n_iterations = 10

        for i in range(n_iterations):
            if linear == 'Y':
                train = resample(X, random_state=i)
                lin_reg.fit(train, y.loc[train.index])

                if center == 'Y':
                    if X.iloc[:,features[0]].nunique() == 2:
                        pdp_zero_lin = pd_xs(features, X.iloc[:,features[0]].min(), X.copy(), lin_reg) # PD at min
                    else:
                        pdp_zero_lin = pd_xs(features[0], X.iloc[:,features[0]].mean(), X.copy(), lin_reg) # PD at mean
                else:
                    pdp_zero_lin = 0.0
                        
                for j in range(0, len(pdp)):
                    pdp_linear[j,n] = pd_xs(features[0], grid_x[j], X.copy(), lin_reg) - pdp_zero_lin
            n = n + 1
                
        # Plot PD and hist
        if ensemble == 'Y':
            graph_pd_plot(pdp, pdp_linear, grid_x, no_models, X, features, ensemble, linear, type_reg_model, filename=filename, unscale_x=unscale_x, unscale_y=unscale_y, log_y=log_y, show=show)
            if grad == 'Y':
                plt.close()
                graph_pd_grad_plot(pdp, pdp_linear, grid_x, no_models, X, features, ensemble, linear, type_reg_model, filename=filename, unscale_x=unscale_x, unscale_y=unscale_y, show=show)
        else:
            graph_pd_plot(pdp, pdp_linear, grid_x, 0, X, features, ensemble, linear, type_reg_model, filename=filename, unscale_x=unscale_x, unscale_y=unscale_y, log_y=log_y, show=show)
            if grad == 'Y':
                plt.close()
                graph_pd_grad_plot(pdp, pdp_linear, grid_x, 0, X, features, ensemble, linear, type_reg_model, filename=filename, unscale_x=unscale_x, unscale_y=unscale_y, show=show)
            
#    plt.close()

    return  pdp


def pd_OHE_1D_plot(X,y, ohe_features, reg, ensemble='N', linear='N', grid_size=None, type_reg_model='NN', grad=None, filename = './', xtick_labels = None, unscale_y = None, center='N', log_y=None, show=None):
    # OHE function version of PD plot
    # Plots the partial dependence with the distribution of variables shown
    # Pass target features as array
    # If ensemble = 'Y', uncertainity for PD shown 
    # If linear = 'Y', the linear solution is also shown and y must be passed
    # Type of regression, gives legend name
    # If grad is Y the gradient is also plotted
    # x_tick_labels are the values for the OHE

    plot_format()
    
    if grid_size is not None:
        print('There is no grid size for this function, the argument is ignored.')
   
    if grad is not None:
        print('There is no grad for this function, the argument is ignored.')
 

        
    if len(ohe_features) == 1: 
        print('The OHE features passed must be greater than one.')
        return
    else:
        # 1D PD Plot        
        ohe_values = np.zeros(len(ohe_features)) # Set arrays for ohe_values 
        
        if ensemble == 'Y':
            no_models = len(reg.models_array)
            pdp = np.zeros((len(ohe_features), no_models))

            # PD for OHE
            n = 0
            
            for r in reg.models_array:
                # For each OHE feature the PD is calculated
                m = 0 
                for j in ohe_features:
                    # The OHE PD ensures that the other OHE features are set to 0 
                    pdp[m, n] = pd_ohe(j, ohe_features, X.copy(), r) 
                    m = m + 1
                    
                n = n + 1

            pdp = np.array(pdp)
            min_pdp = pdp.mean(axis=1).min() # minimum
            
            if center == 'Y':
                pdp = pdp - min_pdp # Set the zero point as the minimum OHE value
                    
        else:
            no_models = 10
            pdp = np.zeros(len(ohe_features))

            m = 0 
            for j in ohe_features:
                # The OHE PD ensures that the other OHE features are set to 0 
                pdp[m] = pd_ohe(j, ohe_features, X.copy(), reg) 
                m = m + 1
                
            pdp = np.array(pdp)
            min_pdp = pdp.min() # minimum

            if center == 'Y':
                pdp = pdp - min_pdp # Set the zero point as the minimum OHE value

                
        # Add linear solution 
        pdp_linear = np.zeros((len(ohe_features), no_models))
#        lin_reg = linear_model.LinearRegression()
        lin_reg = linear_model.Lasso(alpha=0.000001)
        n = 0
        
        # Set number of resampling
        if ensemble == 'Y':
            n_iterations = no_models
        else:
            n_iterations = 10

        # Linear PDP
        for i in range(n_iterations):
            if linear == 'Y':
                train = resample(X, random_state=i)
                lin_reg.fit(train, y.loc[train.index])

                m = 0
                for j in ohe_features:
                    # The OHE PD ensures that the other OHE features are set to 0 
                    pdp_linear[m, n] = pd_ohe(j, ohe_features, X.copy(), lin_reg)
                    m = m +1
                    
                n = n + 1

        pdp_linear = np.array(pdp_linear)
        min_pdp_lin = pdp_linear.mean(axis=1).min() # minimum  

        if center == 'Y':
            pdp_linear = pdp_linear - min_pdp_lin # Set the zero point as the minimum OHE value

        # Unscale PD
        if unscale_y is not None:
            pdp = (pdp * unscale_y[1]) + unscale_y[0]

            if linear == 'Y':
                pdp = (pdp * unscale_y[1]) + unscale_y[0]
        
                
        # Plot PD and hist
        if ensemble == 'Y':
            graph_pd_plot(pdp, pdp_linear, np.arange(0,len(ohe_features)), no_models, X, ohe_features, ensemble, linear, type_reg_model, filename=filename, xtick_labels=xtick_labels, ohe='Y', unscale_x = None , unscale_y = unscale_y, log_y=log_y, show=show)
        else:
            graph_pd_plot(pdp, pdp_linear, np.arange(0,len(ohe_features)), 0, X, ohe_features, ensemble, linear, type_reg_model, filename=filename, xtick_labels=xtick_labels, ohe='Y', unscale_x = None , unscale_y = unscale_y, log_y=log_y, show=show )

    if show != 'Y':
        plt.close()

    return  pdp, pdp_linear



def graph_pd_plot(pdp, pdp_linear, grid_x, no_models, X, features, ensemble, linear, type_reg_model='NN', filename='./', xtick_labels=None, ohe='N', unscale_x = None, unscale_y = None, log_y='Y', show=None):
    # Produce graph for pdp and hist 
    # Unscale is used to unstandardize a variable, [mean, std.]

    if unscale_y is not None:
        pdp = (pdp * unscale_y[1]) + unscale_y[0]

        if linear == 'Y':
            pdp_linear = (pdp_linear * unscale_y[1]) + unscale_y[0]
    
    plt.figure(figsize=(7.5, 6.1))
    plt.rcParams["axes.linewidth"] = 1.25

    # definitions for the axes
    left, width = 0.2, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.135
    
    rect_scatter = [left + 0.06,0.1 + bottom + spacing, width, height]
    rect_histx = [left + 0.06, spacing, width, 0.2]
    
    ax_scatter = plt.axes(rect_scatter)
    ax_scatter.tick_params(direction='in', top=True, right=True, labelbottom=False)
    # And histogram of prob density 
    ax_histx = plt.axes(rect_histx)
    ax_histx.tick_params(direction='in', labelbottom=True)
    
    ax_scatter.set_ylabel('Partial Dependence')

    if no_models == 0 or ensemble != 'Y':
        # Plot non-ensemble
        # Plot linear
        if linear == 'Y':
            # Plot linear model
            if X.iloc[:,features[0]].nunique() < 5 or ohe == 'Y': 
                ax_scatter.errorbar(grid_x, pdp_linear.mean(axis=1), 1.96 * pdp_linear.std(axis=1), fmt='x', markersize=12, markeredgewidth=2.0, linewidth=2.0, color='C2', label='Linear')    
            else:
                ax_scatter.plot(grid_x, pdp_linear.mean(axis=1), 'C2-', label='Linear')
                ax_scatter.plot(grid_x, pdp_linear.mean(axis=1) - 1.96*pdp_linear.std(axis=1), 'C2--')
                ax_scatter.plot(grid_x, pdp_linear.mean(axis=1) + 1.96*pdp_linear.std(axis=1), 'C2--')
                
            # Plot NN model
            if X.iloc[:,features[0]].nunique() < 5 or ohe == 'Y':
                # Marker shown
                ax_scatter.plot(grid_x, pdp, 'x', markersize=12, markeredgewidth=2.0, label=type_reg_model)
            else:
                # Line shown
                ax_scatter.plot(grid_x, pdp, '-', label=type_reg_model)
        else:
            # If no linear plot
            # Plot NN model
            if X.iloc[:,features[0]].nunique() < 5 or ohe == 'Y':
                # Marker shown
                ax_scatter.plot(grid_x, pdp, 'x', markersize=12, markeredgewidth=2.0, label=type_reg_model)
            else:
                # Line shown
                ax_scatter.plot(grid_x, pdp, '-', linewidth=2.0, label=type_reg_model)    
            
    else:        
        # Ensemble plots
        # Plot linear
        if linear == 'Y':
            if X.iloc[:,features[0]].nunique() < 5 or ohe == 'Y': 
            # If discrete then use errorbars
                ax_scatter.errorbar(grid_x, pdp_linear.mean(axis=1), 1.96 * pdp_linear.std(axis=1), fmt='x', markersize=12, markeredgewidth=2.0, color='C2', label='Linear')                
            else:
                ax_scatter.plot(grid_x, pdp_linear.mean(axis=1), 'C2-', label='Linear')
                ax_scatter.plot(grid_x, pdp_linear.mean(axis=1) - 1.96*pdp_linear.std(axis=1), 'C2--')
                ax_scatter.plot(grid_x, pdp_linear.mean(axis=1) + 1.96*pdp_linear.std(axis=1), 'C2--')
 
        if X.iloc[:,features[0]].nunique() < 5 or ohe == 'Y': 
            # If discrete then use errorbars
            ax_scatter.errorbar(grid_x, pdp.mean(axis=1), 1.96 * pdp.std(axis=1), fmt='x', markersize=12, markeredgewidth=2.0, color='C0', label=type_reg_model)                
        else:
            # If continous then use line graph
            ax_scatter.plot(grid_x, pdp.mean(axis=1), 'C0', label=type_reg_model)

            # 95% interval is 1.96
            ax_scatter.plot(grid_x, pdp.mean(axis=1) - 1.96 * pdp.std(axis=1), 'C0--')
            ax_scatter.plot(grid_x, pdp.mean(axis=1) + 1.96 * pdp.std(axis=1), 'C0--')
            
    # Add Legend
    if linear == 'Y':
        ax_scatter.legend()

    
    # Figure sizes and histogram plotted
    if X.iloc[:,features[0]].nunique() > 5:
        binwidth = grid_x[1] - grid_x[0]
    else:
        binwidth = (grid_x.max() - grid_x.min())/20
        
    lim_x = np.ceil(np.abs(grid_x).max() / binwidth) * binwidth

    diff_grid = (grid_x.max() - grid_x.min())/10
    
    ax_scatter.set_xlim(grid_x.min() - diff_grid, grid_x.max() + diff_grid)


    if ohe == 'Y':
        # The bar graph shows the populations of the OHE
        ax_histx.bar(np.arange(0,len(features)), (X.iloc[:, features] == X.iloc[:, features].max()).sum())
        ax_histx.set_yscale('log')
        ax_histx.set_xticks(np.arange(0,len(features)))

        cur_y_ticks = ax_histx.get_yticks()

        if len(cur_y_ticks > 2):
            ax_histx.set_yticks(cur_y_ticks[::2])

        # xtick labels
        if xtick_labels is not None:
            ax_histx.set_xticklabels(xtick_labels)
            
        plot_format()
        plt.savefig(filename + 'PD_OHE_plot_' + str(features[0]) + '.pdf')
    else:

        if unscale_x is not None:
            bins = np.arange((-lim_x * unscale_x[1]) + unscale_x[0], (lim_x * unscale_x[1]) + unscale_x[0] + binwidth* unscale_x[1], binwidth* unscale_x[1])
            
            # Unscales the X values and plots histogram 
            n,_,_ = ax_histx.hist((X.iloc[:,features[0]].values * unscale_x[1]) + unscale_x[0], bins=bins)
            ax_histx.set_xlim([(ax_scatter.get_xlim()[0] * unscale_x[1]) + unscale_x[0], (ax_scatter.get_xlim()[1] * unscale_x[1]) + unscale_x[0]])
        else:
            bins = np.arange(-lim_x, lim_x + binwidth, binwidth)

            n,_,_ = ax_histx.hist(X.iloc[:,features[0]].values, bins=bins)
            ax_histx.set_xlim(ax_scatter.get_xlim())
        
        ax_histx.set_xlabel(str(X.columns[features[0]]))    #.replace('_', '-'))

        
        if log_y == 'Y':
            ax_histx.set_yscale('log')
            ax_histx.set_ylim(0.9, 1.1*len(X) )
            ymin = 1.0
            ymax = max(1.1 * n.max(), 11)
            ax_histx.set_yticks(10**(np.arange(np.floor(np.log10(ymin)), np.ceil(np.log10(ymax)), 2)) )

        #            ax_histx.set_yticks(10**(np.arange(np.floor(np.log10(ymin)) + 1, np.ceil(np.log10(ymax)), 2)) , minor=True)
            
#        ymin, ymax = ax_histx.get_ylim()
#        ax_histx.set_yticks(np.round(np.linspace((ymin), (ymax), 3)))
#        ax_histx.yaxis.set_major_locator(plt.MaxNLocator(5))
#        plt.tight_layout()
        plot_format()
        plt.savefig(filename + 'PD_plot_' + str(features[0]) + '.pdf',bbox_inches='tight')
#        plt.close()

    if show != 'Y':
        plt.close()


    
def graph_pd_grad_plot(pdp, pdp_linear, grid_x, no_models, X, features, ensemble, linear, type_reg_model='NN', filename='./', unscale_x = None, unscale_y = None, show = None):

    # Unscale PD and grid_x
    if unscale_y is not None:
        pdp = (pdp * unscale_y[1]) + unscale_y[0]
        if linear == 'Y':
            pdp_linear = (pdp_linear * unscale_y[1]) + unscale_y[0]

    if unscale_x is not None:
        grid_x = (grid_x * unscale_x[1]) + unscale_x[0]

        
    #Plots the gradients ofthe partial dependence
    if X.iloc[:,features[0]].nunique() > 10 and ensemble == 'Y':

        # ML model
        grad, grid_mid = calc_pdp_grad(pdp, grid_x)
        grad_mean = grad.mean(axis=1) 
        grad_up = grad.mean(axis=1) + 1.96*grad.std(axis=1) 
        grad_down = grad.mean(axis=1) - 1.96*grad.std(axis=1)

        plt.plot(grid_mid, grad_mean, 'C0-')
        plt.plot(grid_mid, grad_up, 'C0--')
        plt.plot(grid_mid, grad_down, 'C0--')
            
        if linear == 'Y':
            # Linear model
            lin_grad, grid_mid = calc_pdp_grad(pdp_linear, grid_x)
            lin_grad_mean = lin_grad.mean(axis=1) 
            lin_grad_up = lin_grad.mean(axis=1) + 1.96*lin_grad.std(axis=1) 
            lin_grad_down = lin_grad.mean(axis=1) - 1.96*lin_grad.std(axis=1)

            # Linear plot
            plt.plot(grid_mid, lin_grad_mean, 'C2-')
            plt.plot(grid_mid, lin_grad_up, 'C2--')
            plt.plot(grid_mid, lin_grad_down, 'C2--')
        
        plt.xlabel(str(X.columns[features[0]]).replace('_', '-'))
        plt.ylabel('PD Grad.')
        plt.rcParams["axes.linewidth"] = 1.25
        
#        plt.tight_layout()
        plot_format()
        plt.savefig(filename + 'Grad_PD_plot_' + str(features[0]) + '.pdf',bbox_inches='tight')

        if show != 'Y':
            plt.close()
            
def calc_pdp_grad(pdp, grid):
    # Calc the pdp grad for 1D and 2D array
    
    # If 1D array
    if len(pdp.shape) == 1:
        grad, grid_mid = pd_grad(pdp, grid)

    # If 2D array
    if len(pdp.shape) == 2:
        grad = []

        for i in range(0,len(pdp[0])): # No of models
            grad_m, grid_mid = pd_grad(pdp[:,i], grid)
            grad.append(grad_m)
    
        grad = np.array(grad).T
        grid_mid = np.array(grid_mid)

    return grad, grid_mid

def pd_grad(pdp, grid):
    # The grad is calculated
    grad = []
    grid_mid = []

    for i in range(1,len(pdp)):
        diff = (pdp[i] - pdp[i - 1])/(grid[i] - grid[i - 1])
        mid = (grid[i] + grid[i - 1])/2
            
        #if i != abs(grid).argmin() and (i - 1) != abs(grid).argmin(): 
        grid_mid.append(mid)
        grad.append(diff)

    grid_mid = np.array(grid_mid)
    grad = np.array(grad)

    return grad, grid_mid


def plot_interactions(inter, X, filename='./', show=None):
    # Plots a heatmap of the interactions present
    # Name is the name of the graph pdf file

    plot_format()
    
    N = len(X.columns)
    
    # Change to 2D matrix
    interactions = np.zeros((N,N))
    n = 0 
    for i in range(0,N):
        for j in range(0,i):
            interactions[i,j] = inter[n]
            interactions[j,i] = inter[n]
            n = n + 1

            
    # Plot interactions
    plt.imshow(interactions, cmap=plt.get_cmap('seismic'))
    plt.colorbar()
    plt.xticks(np.arange(0.0,len(X.columns), 1.0), X.columns, fontsize=18, rotation=90)   
    plt.yticks(np.arange(0.0,len(X.columns), 1.0), X.columns, fontsize=18)
#    plt.grid()
    
    # If all zero set colorbar to 0-1.0
    if inter.sum().sum() == 0:
        plt.clim(0,0.10) 

    plot_format()
    plt.savefig(filename + '_Interactions.pdf', bbox_inches='tight') 

    if show != 'Y':
        plt.close()

def plot_format():
    # Format for graphs     

    # Matplotlib settings
    plt.rcParams.update({
            'font.size': 21,
            'legend.fontsize': 21,
            'axes.labelsize': 21,
            'axes.titlesize': 21,
            'figure.figsize': (8.5, 6.5),
            'text.usetex': True,
            'errorbar.capsize': 5.0

        })

    plt.rc('font', family='serif') 
    
    matplotlib.rcParams['xtick.labelsize'] = 21 
    matplotlib.rcParams['ytick.labelsize'] = 21 
    matplotlib.rcParams['font.size'] = 21




