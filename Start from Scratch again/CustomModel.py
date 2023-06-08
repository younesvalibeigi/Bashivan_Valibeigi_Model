# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 11:27:51 2023

@author: yvalib
"""

import torch
import torch.nn as nn

class CustomModel(nn.Module):
    def __init__(self, layers):
        super(CustomModel, self).__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
