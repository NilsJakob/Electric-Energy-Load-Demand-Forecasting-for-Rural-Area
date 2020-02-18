import pandas as pd
import numpy as np

def variance(x, sample=False): 
    if isinstance(x, pd.core.frame.DataFrame):
        x=x.to_numpy()
        
    if isinstance(x, pd.core.series.Series):
        x=x.to_numpy()
    
    if not isinstance(x, np.ndarray):
        raise ValueError("x is not a numpy array.")
    
    #Count:
    if sample:
        n_x = len(x) - 1
    else: 
        n_x = len(x)
    
    mean_x = np.sum(x)/(n_x)
     
    squared_deviation = np.sum((x-mean_x)**2)
    variance = squared_deviation/n_x
    sigma = np.sqrt(variance)
    
    return sigma      

def total_variance(x, y):
    if isinstance(x, pd.core.frame.DataFrame):
        x=x.to_numpy()
        
    if isinstance(x, pd.core.series.Series):
        x=x.to_numpy()
    
    if not isinstance(x, np.ndarray):
        raise ValueError("x is not a numpy array.")
        
    n_x = len(x)
    n_y = len(y)
    
    mean_x = np.sum(x)/(n_x)
    mean_y = np.sum(y)/(n_y)
    
    tot_var_xy=np.sum((x-mean_x)*(y-mean_y))
    sigma = tot_var_xy/n_x
    
    return sigma