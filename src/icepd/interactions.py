import numpy as np
import pandas as pd

from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split

from statsmodels.formula.api import ols
import statsmodels.api as sm

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm 

from scipy.stats import chisquare, chi2_contingency


def l1_interactions(X, y, alpha):
    # Lasso method for detecting interactions
    reg = linear_model.Lasso()
    reg.alpha = alpha

    X_cross = calc_X_cross(X)
           
    reg.fit(X.join(X_cross),y)
    
    params = reg.coef_

    params = params[(len(X.columns) -1):]

    return params

def stats_summary_lr(X,y):
    X2 = sm.add_constant(X)
    est = sm.OLS(y, X2)
    est2 = est.fit()
    print(est2.summary())

    return est2

def OLS_interactions(X,y, pvalue=0.01):
    # The OLS/ANOVA interaction detection
    # Statistical significance to detect interactions
    # The p value below which an interactions is detected can be set 
    X_cross = calc_X_cross(X)
    est = stats_summary_lr(X.join(X_cross),y)    
    params = est.params.values

    # Only include coefficients which have p value below certain value
    params[est.pvalues > pvalue] = 0

    params = params[(len(X.columns)):]
    
    return params

def calc_X_cross(X):

    x_cross = pd.DataFrame(index=X.index)

    for i in range(0,len(X.columns)):
        for j in range(0,i) :
            x_cross = x_cross.join(pd.DataFrame((X.iloc[:,i]*X.iloc[:,j]), columns=['_cross_'+str(i) + '_' + str(j)]))

    return x_cross

    

def guide(X,y):
    # GUIDE interaction detection
    # This method does not work well for collinear data

    # Additive GBR fit 
    rf = GradientBoostingRegressor(n_estimators=100, max_depth=1, learning_rate=0.75, loss='ls', random_state=1)
    rf.fit(X,y)
    resid = rf.predict(X) - y.values.ravel()

    N = len(X.iloc[0,:])

    chi_array = []
    min_array = []
    min_p_array = []

    for i in range(0,N):
        for j in range(0,i):
            # For binary
            if X[i].nunique() == 2:
                mid_i = X[i].mean()
            else:
                mid_i = X[i].median()
      
            if X[j].nunique() == 2:
                mid_j = X[j].mean()
            else:
                mid_j = X[j].median()

            # Table of positive, negative
            table = np.zeros((4,2))

            table[0,0] = sum(resid[ (X[i] >= mid_i)  & (X[j] >= mid_j)] > 0)
            table[0,1] = sum(resid[ (X[i] >= mid_i)  & (X[j] >= mid_j)] < 0)
            table[1,0] = sum(resid[ (X[i] >= mid_i)  & (X[j] < mid_j)] > 0)
            table[1,1] = sum(resid[ (X[i] >= mid_i)  & (X[j] < mid_j)] < 0)
            table[2,0] = sum(resid[ (X[i] < mid_i)  & (X[j] >= mid_j)] > 0)
            table[2,1] = sum(resid[ (X[i] < mid_i)  & (X[j] >= mid_j)] < 0)
            table[3,0] = sum(resid[ (X[i] < mid_i)  & (X[j] < mid_j)] > 0)
            table[3,1] = sum(resid[ (X[i] < mid_i)  & (X[j] < mid_j)] < 0)

            # Drop row with zero entries
            table = table[np.all(table != 0, axis=1)]

            chi_array.append(chi2_contingency(table.T)[1])
    
    params = (np.array(chi_array) < 0.01)

    # True/false interactions shown rather than strength
    return params



def bidirectional_stepwise_regression(X,y):
    # Stepwise addition of interactions
    # Stepwise has known issues, included for completeness
    
    X_cross = calc_X_cross(X)
    
    all_var = X.join(X_cross)
    current_model = pd.DataFrame()

    while True:
        p_value = []

        # Check p value of terms added
        for i in range(0,len(all_var.columns)):
            if all_var.columns[i] not in current_model.columns:
                est = stats_summary_lr((pd.DataFrame(all_var.iloc[:,i]).join(current_model)),y)   
                p_value.append(est.pvalues.values[1]) # 0 is constant, 1 is added
            else:
                p_value.append(100) # High value

        # Add minimum term if less than thres otherwise leave loop
        if np.array(p_value).min() < 0.15: # Threshold
            indices_add = np.array(p_value).argmin()
            print('Added ' + str(indices_add))
            current_model = pd.DataFrame(all_var.iloc[:,indices_add]).join(current_model) 
        else:
            break

        # Remove any variables that are now below threshold
        est = stats_summary_lr(current_model,y)   
        current_model = current_model.drop(columns=current_model.columns[est.pvalues.values[1:] > 0.15])

    # Get indices
    params = np.zeros(len(all_var.columns))

    # Final model remove all with pvalue greater than 0.05
    current_model = current_model.drop(columns=current_model.columns[est.pvalues.values[1:] > 0.05])
    
    for i in range(0,len(all_var.columns)):
        if all_var.columns[i] in current_model.columns:
            params[i] = 1

    params = bool(params)
            
    return params
