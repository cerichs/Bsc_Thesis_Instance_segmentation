
import numpy as np
import pandas as pd

def mean_centering(data, ref = None):
    
    mean = []
    
    mean_list = data.mean(axis=0)
    data_mean = [data - mean_list if ref is None else data - ref]
    for i in range(len(data)):
        mean.append( np.array(data_mean[0].loc[i].tolist()) )
    
    return pd.DataFrame(mean)

def msc_hyp(hyperspectral_dataframe, ref = None):
    
    
    if ref is None:
        # Get the reference spectrum. Estimate it from the mean. This is computed during training    
        ref = hyperspectral_dataframe.mean(axis=0)
    else:
        # If reference is not None, we are running on test-data
        ref = ref
    
    # Define a new array and populate it with the data    
    data_msc = np.zeros_like(hyperspectral_dataframe)
    for i in range(hyperspectral_dataframe.shape[0]):
        ref = list(ref)
        hyp = list(hyperspectral_dataframe.iloc[i,:])
        
        # Run regression
        fit = np.polyfit(ref, hyp, 1, full=True)

        # Apply correction
        data_msc[i,:] = (hyp - fit[0][1]) / fit[0][0]
        
        
    return pd.DataFrame(data_msc), ref