# -*- coding: utf-8 -*-
"""
Created on Fri May 12 11:01:42 2023
Changed Thur June 8, 2023

@author: yvalib
"""

import torchvision.models as models
import torch 
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import logging
import h5py
from Mapper import Mapper
from Mapper2 import Mapper2

# In[]:
## Only to visualize the data
import nexusformat.nexus as nx
hf =nx.nxload('npc_v4_data.h5')
print(hf.tree)
print(hf.readme)
del hf

# In[]:
torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

# In[]:
# variable for adjusting the input data
transform = transforms.Compose([transforms.Resize((224, 224))])

# loading the input data
hf = h5py.File('npc_v4_data.h5', 'r')
natural_img = torch.from_numpy(np.array(hf.get('images/naturalistic')))
monkey_n_s1 = np.array(hf.get('/neural/naturalistic/monkey_s/stretch/session_1'))
hf.close()

# resizing the input data
stim = natural_img.repeat(3,1,1,1)
stim = torch.permute(stim, (1, 0, 2, 3))
stim = transform(stim).to(device, dtype=torch.float32) # stim is the input images  

# Averaging over the repetitions to only get mean firing rates per image
target = np.mean(monkey_n_s1, axis=0) 
target = torch.tensor(target)
target.shape # target is the avg firing rates for each image


# In[]: Normalizing the input:
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

neural_data = np.array(dataset['neuralData'])
natural_img = torch.from_numpy(np.array(dataset['stimuli']))

target = np.mean(neural_data, axis=1)
target = torch.tensor(target)

# resizing the input data
stim = natural_img.repeat(3,1,1,1)
stim = torch.permute(stim, (1, 0, 2, 3))
stim = transform(stim).to(device, dtype=torch.float32) # stim is the input images  


# In[]: Normalizing the input:
import torchvision.transforms as transforms

# Define normalization parameters for ImageNet dataset
#mean = [0.485, 0.456, 0.406]
#std = [0.229, 0.224, 0.225]

mean = torch.mean(stim, dim=(0, 2, 3))
std = torch.std(stim, dim=(0, 2, 3))
# Create normalization transform
normalize = transforms.Normalize(mean=mean, std=std)

# Apply normalization to your input images
stim = normalize(stim)


# In[]:
# Choose the model
model = models.resnet50(pretrained=True)
model = model.to(device)

model.eval()
    
# In[]:
'''
import pickle
#from corr_score import corr_score

# Empty list to store the average layer responses
layer_responses = []
layer_names = []
r_scores = []
cc=0
# Loop through each layer in the model
for name, layer in model.named_modules():
    # Register a hook to get the layer output
    current_layer_response = []
    def hook_fn(module, input, output):
        current_layer_response.append(output.cpu().detach().numpy())
    layer.register_forward_hook(hook_fn)

    # Make a forward pass through the model with one batch to get the layer output
    with torch.no_grad():
        _ = model(stim)
        
    # Get the average layer response
    layer_responses.append(current_layer_response[0])
    
    # Store the layer name
    layer_names.append(name)

    # Remove the hook to free up memory
    layer._forward_hooks.clear()
    print("cc:", cc)
    cc=cc+1
# In[]:
   
# Saving the layer responses and layer names
with h5py.File('layer_data_myNorm.h5', 'w') as f:
    indices_group = f.create_group('indices')
    for i, arr in enumerate(layer_responses):
        # Create a subgroup for each index
        index_group = indices_group.create_group(str(i))
        # Store the layer name and layer response in the subgroup
        index_group.create_dataset('layer_name', data=layer_names[i])
        index_group.create_dataset('layer_response', data=arr)
        
'''

# In[]:
import nexusformat.nexus as nx
h5_layers =nx.nxload('layer_data_myNorm.h5')
print(h5_layers.tree)
del h5_layers

index = 119#70#119
h5_layers = h5py.File('layer_data_myNorm.h5', 'r')
X = torch.from_numpy(np.array(h5_layers.get('/indices/' + str(index) + '/layer_response')))
layerName = (h5_layers.get('/indices/' + str(index) + '/layer_name'))
layerName = layerName[()].decode()
h5_layers.close()

# In[]:

from scipy.stats import pearsonr

print(X.shape)
print(layerName)
Y = target
num_neurons = Y.shape[1]
mapper2=Mapper2(graph=None, max_epochs=120
                , tol=0.1,ls=1000,ld=1, # weights for different losses
                  init_lr=1e-1, batch_size=40, num_neurons=num_neurons,
                  map_type='separable', inits=None, decay_rate=10, log_rate=10)
mapper2.fit(X.to(device),Y.to(device))
plt.matshow((torch.squeeze(mapper2.s_w.cpu().detach()).permute(2,0,1))[1])
prediction=(mapper2.predict(X.to(device))).cpu().detach().numpy()
scores = np.array([pearsonr(prediction[:, i], Y[:, i])[0] for i in range(prediction.shape[-1])])
print('convmap score: '+ str(np.median(scores)))


# In[]:
'''
# loop over all layers and calculate the scores
from scipy.stats import pearsonr
import nexusformat.nexus as nx
h5_layers =nx.nxload('layer_data_myNorm.h5')
indices_subgroup = h5_layers['indices']
num_elements = len(indices_subgroup.entries)
Y = target
num_neurons = Y.shape[1]
score_array = []
s_w_array = []
del h5_layers
for index in range(1, num_elements):
    h5_layers = h5py.File('layer_data_myNorm.h5', 'r')
    X = torch.from_numpy(np.array(h5_layers.get('/indices/' + str(index) + '/layer_response')))
    layerName = (h5_layers.get('/indices/' + str(index) + '/layer_name'))
    layerName = layerName[()].decode()
    h5_layers.close()

       
    #print(X.shape)
    #print(layerName)
    mapper=Mapper2(graph=None, max_epochs=80
                    , tol=0.1,ls=1000,ld=1,
                      init_lr=1e-1, batch_size=5, num_neurons=num_neurons,
                      map_type='separable', inits=None, decay_rate=10, log_rate=10)
    mapper.fit(X.to(device),Y.to(device))
    torch.cuda.empty_cache()

    current_s_w = (torch.squeeze(mapper.s_w.cpu().detach()).permute(2,0,1))[1]
    plt.matshow(current_s_w)
    s_w_array.append(current_s_w)
    prediction=(mapper.predict(X.to(device))).cpu().detach().numpy()
    #prediction=(mapper.predict(X.to('cpu')))
    scores = np.array([pearsonr(prediction[:, i], Y[:, i])[0] for i in range(prediction.shape[-1])])
    score_array.append(np.median(scores))
    print('convmap score: '+ str(np.median(scores))+ '=============================')
'''

# In[]: eqv V4 model
for idx, (name, layer) in enumerate(model.named_modules()):
    # if idx == 0:
    #     continue
    print(idx, ':', name)
    if idx == 10:
        break
cc = 0
for idx, layer in enumerate(model.children()):
    for i, sublayer in enumerate(layer.children()):
        print(cc, ':  ', sublayer)
        cc = cc+1
    print(idx, ':', layer)
    #if idx == 10:
     #   break
# Loop through layers and sub-layers
for module in enumerate(model.children()):
    print(name, module)
    if len(list(module.children())) > 0:
        for sub_name, sub_module in module.named_children():
            print("\t", sub_name, sub_module)
    print('=============================')
    if idx == 10:
        break
################################################  
layer_index = 70
layerlist = []
for idx, (name, layer) in enumerate(model.named_modules()):
    if idx == 0:
        continue
    if idx>layer_index:
        break
    print(idx, ':', name, ',  ', layer)
    layerlist.append(layer)

# from CustomModel import CustomModel
# new_model = CustomModel(layerlist)


v4_eqv = torch.nn.Sequential(*layerlist) # the first layer include the whole resnet
################################################

v4_eqv = torch.nn.Sequential(*list(model.modules())[1:71]) # the first layer includes the whole resnet

for idx, (name, layer) in enumerate(v4_eqv.named_modules()): 
    print(idx, ':', name, ':  ', layer)

###############################################
v4_eqv = torch.nn.Sequential(*list(model.children())[:7])
###############################################


# Move the new model to the same device as the original model
v4_eqv = v4_eqv.to(device)
# # Freeze the parameters in the new model
# for param in v4_eqv.parameters():
#     param.requires_grad = False
# Freeze the parameters in the new model except for BatchNorm layers
for param in v4_eqv.parameters():
    if not isinstance(param, torch.nn.BatchNorm2d):
        param.requires_grad = False

##############################################
for idx, (name, layer) in enumerate(v4_eqv.children()): #<==================== why named_module() gives wrong answer
    print(idx, ':', name, ':  ', layer)
    
############################
activation = {}
def get_activation(name):
    def hook(model, input, output):
        #activation[name] = output.detach()
        activation[name] = output
    return hook
activation['layer3.0']

model.register_forward_hook(get_activation('layer3.0'))

for idx, (name, layer) in enumerate(model.named_modules()):
    if idx == 70:
        module.register_forward_hook(get_activation(name, activation))
# In[]: V4_mapper model
v4_mapper = torch.nn.Sequential(v4_eqv, mapper2)

Y1 = v4_eqv(stim)
Y2 = mapper2.predict(Y1)
#Y3 = mapper2.predict(v4_eqv(stim))

Y = v4_mapper(stim)

# GPU name
gpu_name = torch.cuda.get_device_name(0)
print("GPU Type:", gpu_name)
# In[]: Lucent




# In[]:
for idx, layer in enumerate(model.modules()):
    print(idx, ':  ', layer)
    


# In[]:


# 1. How to isolate the number of channels I want
# 2. Compute Canada Or how to free up the memory
# 3. The v4_mapper does not work
# 4. consistency



# Resnet Copy













