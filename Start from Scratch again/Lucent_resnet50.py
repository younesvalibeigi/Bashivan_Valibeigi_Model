# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 17:01:55 2023

@author: yvalib
"""
# https://lucent.readthedocs.io/en/latest/tutorials/first_steps.html
# In[]: 
# !pip install --quiet git+https://github.com/greentfrapp/lucent.git
import torch

from lucent.optvis import render, param, transform, objectives
from lucent.modelzoo.util import get_model_layers

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[]: 
import torchvision.models as models

resnet50 = models.resnet50(pretrained=True)
resnet50 = resnet50.to(device)
resnet50.eval()

get_model_layers(resnet50)

for idx, (name, layer) in enumerate(resnet50.named_modules()):
    print(idx, ':  ', name, ',  ', layer)




# In[]:

# render_vis works in the same way! Just substitute the appropriate layer name!
list_of_images = render.render_vis(resnet50, "layer4_1_conv1:121", show_inline=True)
list_of_images = render.render_vis(resnet50, "layer3_0:100", show_inline=True)


import matplotlib.pyplot as plt
import numpy as np
im = list_of_images[0]
max_im = np.zeros((128,128))
for i in range(im.shape[3]):
    max_im = im[0,:,:,i]
    plt.matshow(max_im, cmap='gray')
    plt.show()

# I read up until batching
# https://lucent.readthedocs.io/en/latest/tutorials/first_steps.html
def show_generated_images(sequence):
    im = sequence[0]
    max_im = np.zeros((128,128))
    for i in range(im.shape[3]):
        max_im = im[0,:,:,i]
        plt.matshow(max_im, cmap='gray')
        plt.show()

# In[]: USe the direction functino in objectives.py in optvis
direction = torch.rand(1024, device=device)
obj = objectives.direction(layer='layer3_0', direction=direction)
sequence = render.render_vis(resnet50, obj, show_inline=True)
#from lucent.misc.io.showing import animate_sequence
show_generated_images(sequence)



# In[]: Try the lucent only with the mapper
from Mapper2 import Mapper2
# You need to run clean_convmap.py first to run this part
# for now Y.shape is 22 and X is 1024, 14, 14
direction = torch.rand(22,device=device)
obj = objectives.direction(layer='s_w', direction=direction)
sequence = render.render_vis(mapper2, obj, show_inline=True) # <=== Does not work




