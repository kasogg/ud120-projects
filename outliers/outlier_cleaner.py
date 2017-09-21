#!/usr/bin/python

import numpy as np

def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []


    ### your code goes here

    combined_data = np.hstack([ages, net_worths, predictions - net_worths])
    index = np.lexsort(combined_data.T)
    
    cleaned_data = combined_data[index[:-9]]

    return cleaned_data

