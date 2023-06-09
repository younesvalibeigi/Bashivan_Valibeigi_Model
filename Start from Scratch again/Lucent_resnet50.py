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

import matplotlib.pyplot as plt
im = list_of_images[0]
im = im[1:]
plt.matshow(im[:,:,:,1])

# I read up until batching
# https://lucent.readthedocs.io/en/latest/tutorials/first_steps.html

