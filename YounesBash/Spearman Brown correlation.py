# -*- coding: utf-8 -*-
"""
Created on Sun May 28 11:39:43 2023

@author: yvalib
"""

import scipy.io
import numpy as np
from scipy.stats import pearsonr

# Specify the path to your .mat file
mat_file = "C:/Users/yvalib/Documents/Data/3_Results/a018/Dataset_Bashivan_5stim_newArrangment_a018.mat"

# Load the .mat file
data = scipy.io.loadmat(mat_file)

# Function to convert struct to dictionary
def struct_to_dict(struct):
    dictionary = {}
    for field_name in struct.dtype.names:
        field_value = struct[field_name][0, 0]
        if field_value.dtype.names is not None:
            dictionary[field_name] = struct_to_dict(field_value)
        else:
            dictionary[field_name] = field_value
    return dictionary

# Access the struct
struct_data = data['Dataset_Bashivan_5stim_newArrangment']

# Convert struct to dictionary
dataset = struct_to_dict(struct_data)

# Access the converted dictionary
print(dataset)

neural_data = dataset['neuralData']

avg_neuralData = np.mean(neural_data, axis=0)


# Number of splits and repetitions
num_splits = 1000
split_size = 5
sb_corr = np.zeros((1000, 32))
for neuron_num in range(avg_neuralData.shape[1]):
    neuron_data = avg_neuralData[:, neuron_num]

    for split_index, _ in enumerate(range(num_splits)):
    # Generate a random split location
        split_location = np.random.randint(0, 10 - split_size + 1)  # Random index between 0 and (10 - 5)
        #print(split_location)
        # Split the array
        dataset1 = neuron_data[split_location:split_location + split_size]
        dataset2 = np.concatenate((neuron_data[:split_location], neuron_data[split_location + split_size:]), axis=0)
        #print(dataset1.shape, dataset2.shape)
        r, p_value = pearsonr(dataset1, dataset2)
        r_sb = (2*r)/(1+r)
        sb_corr[split_index, neuron_num] = r_sb
        
avg_sb_corr = np.mean(np.abs(sb_corr), axis=0)
        

import matplotlib.pyplot as plt   
plt.hist(avg_sb_corr.T)
