# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 14:21:37 2023

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

torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: ", device)
mapper2=Mapper2(graph=None, max_epochs=120
                , tol=0.1,ls=1000,ld=1, # weights for different losses
                  init_lr=1e-1, batch_size=40, num_neurons=22,
                  map_type='separable', inits=None, decay_rate=10, log_rate=10)


# variable for adjusting the input data
transform = transforms.Compose([transforms.Resize((224, 224))])

# loading the input data
hf = h5py.File('npc_v4_data.h5', 'r')
natural_img = torch.from_numpy(np.array(hf.get('images/naturalistic')))
monkey_n_s1 = np.array(hf.get('/neural/naturalistic/monkey_n/stretch/session_1'))
hf.close()

# resizing the input data
stim = natural_img.repeat(3,1,1,1)
stim = torch.permute(stim, (1, 0, 2, 3))
stim = transform(stim).to(device, dtype=torch.float32) # X is the input images  

target = np.mean(monkey_n_s1, axis=0)
target = torch.tensor(target)
model = models.resnet50(pretrained=True)
model = model.to(device)

model.eval()
v4_eqv = torch.nn.Sequential(*list(model.children())[:7])

v4_mapper = torch.nn.Sequential(v4_eqv, mapper2)

Y = model(stim)
#Y = v4_mapper(stim[0:10, :, :, :])
print(Y.shape)