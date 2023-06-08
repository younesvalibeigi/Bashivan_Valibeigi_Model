# -*- coding: utf-8 -*-
"""
Created on Thu May 18 17:34:23 2023

@author: yvalib
"""


import torch
import h5py
# import sys
# sys.path.insert(1,'/Users/apple/Documents/u2_winter_term/comp396/cubemap')
from basemapper import BaseMapper
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import pdb
import torch.nn.functional as F
from torch.optim import Adam


torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


np.random.seed(123)
npa = np.array

class Mapper2(BaseMapper):
    def __init__(self, graph=None, num_neurons=39, batch_size=50, init_lr=0.01,
               ls=0, ld=0, tol=1e-2, max_epochs=10, map_type='separable', inits=None,input_ph=None,target_ph=None,
               log_rate=100, decay_rate=200, gpu_options=None):
        """
        Mapping function class.
        :param graph: tensorflow graph to build the mapping function with
        :param num_neurons: number of neurons (response variable) to predict
        :param batch_size: batch size
        :param init_lr: initial learning rate
        :param ls: regularization coefficient for spatial parameters
        :param ld: regularization coefficient for depth parameters
        :param tol: tolerance - stops the optimization if reaches below tol
        :param max_epochs: maximum number of epochs to train
        :param map_type: type of mapping function ('linreg', 'separable')
        :param inits: initial values for the mapping function parameters. A dictionary containing
                    any of the following keys ['s_w', 'd_w', 'bias']
        :param log_rate: rate of logging the loss values
        :param decay_rate: rate of decay for learning rate (#epochs)
        """
        super(Mapper2, self).__init__(graph=graph, num_neurons=num_neurons, batch_size=batch_size, init_lr=init_lr,
                                    ls=ls, ld=ld, tol=tol, max_epochs=max_epochs, map_type=map_type, inits=inits,
                                    input_ph=input_ph,
                                     #target_ph=target_ph,
                                     log_rate=log_rate, decay_rate=decay_rate, gpu_options=gpu_options)   
        #input_shape = self._input_ph.shape
#         if self._input_ph is None:

#             input_shape = torch.Size([640, 384, 15, 15])
#         else:
    # def backward(self, grad_output):
    #     # Override the backward method to include the necessary hooks
      #  return grad_output
    def _make_separable_map(self,X): # updated function
        input_shape = X.shape
        if self._inits is None:
            self.s_w=torch.randn((1, 1, input_shape[2], input_shape[3], self._num_neurons), dtype=torch.float,requires_grad=True,device=device) 
            self.d_w=torch.randn(((1, input_shape[1], self._num_neurons)), dtype=torch.float,requires_grad=True,device=device) 
            self.bias=torch.randn(((1, self._num_neurons)), dtype=torch.float,requires_grad=True,device=device)
        else:
            if 's_w' in self._inits:
                self.s_w=Variable(torch.from_numpy(self._inits['s_w'].reshape((1, 1, input_shape[2], input_shape[3], self._num_neurons)),dtype=torch.float), requires_grad=True)
            else:
                self.s_w=Variable(torch.from_numpy(np.random.randn(1, 1, input_shape[2], input_shape[3], self._num_neurons),dtype=torch.float), requires_grad=True)
            if 'd_w' in self._inits:
                self.d_w=Variable(torch.from_numpy(self._inits['d_w'].reshape(1, input_shape[1], self._num_neurons),dtype=torch.float), requires_grad=True)
            else:
                self.d_w = Variable(torch.from_numpy(np.random.randn(1, input_shape[1], self._num_neurons),dtype=torch.float), requires_grad=True)
            if 'bias' in self._inits:
                self.bias=Variable(torch.from_numpy(self._inits['bias'].reshape(1, self._num_neurons),dtype=torch.float), requires_grad=True)
            else:
                self.bias = Variable(torch.from_numpy(np.zeros((1, self._num_neurons),dtype=torch.float)), requires_grad=True)
    
    def _make_loss(self,Y):
        #self._s_vars,self._d_vars,self._biases=self.forward()
        #self._predictions=self.forward()
        self._target_ph=Y
        self.l2_error=torch.norm(self._predictions - self._target_ph,p=2)
        if self._map_type =='separable':
            laplace_filter=torch.from_numpy(np.array([0, -1, 0, -1, 4, -1, 0, -1, 0],dtype=np.float).reshape((1,1,3,3))).to(device)
            laplace_filter=laplace_filter.type(torch.cuda.FloatTensor)
            # laplace_filter=laplace_filter.type(torch.FloatTensor)
            conv =nn.Conv2d(1, 1, kernel_size=1,bias=False,padding="same").to(device)
            conv.weight = torch.nn.Parameter(laplace_filter)
            # torch.nn.Parameter( torch.FloatTensor(7, 32, 32, device="cuda") )
            conv.requires_grad_(False)
            #laplace_loss=self._l2_loss(conv(torch.squeeze(self.s_w.permute(4, 3, 1, 2, 0),4))) #neurosn, ch, spatial dim, bacth
            laplace_loss=self._l2_loss(conv(torch.squeeze(self.s_w.permute(4, 1, 2, 3, 0),4))) #<=============
            #l2_loss_s = self._l2_loss(self.s_w.permute(4, 3, 1, 2, 0))
            l2_loss_s = self._l2_loss(self.s_w.permute(4, 1, 2, 3, 0))
            l2_loss_d = self._l2_loss(self.d_w.permute(2, 1, 0))
            self.reg_loss=self._ls*laplace_loss+self._ld*(l2_loss_s+l2_loss_d)
            self._total_loss = self.l2_error + self.reg_loss
            return self._total_loss

    def forward(self,X,Y):
        self.predict(X)
        loss=self._make_loss(Y)
        self._is_initialized = True
        return loss
    def fit(self, X, Y):
        self._make_separable_map(X)
        # print(self.s_w)
        # optimizer = SGD( [self.s_w,self.d_w,self.bias],lr=self._lr , momentum=0.9)
        optimizer = Adam( [self.s_w,self.d_w,self.bias],lr=self._lr )
        for e in range(self._max_epochs):
            for counter, batch in enumerate(self._iterate_minibatches(X, Y, batchsize=self._batch_size, shuffle=True)):                 
                optimizer.zero_grad()
                # print(batch[0].shape,batch[1].shape)
                self.forward(batch[0],batch[1])
                # pdb.set_trace()
                
                self._total_loss.backward()
                
                optimizer.step()
            if (e % self._log_rate == 0) or (e == self._max_epochs - 1):
              print('Epoch: %d, Err Loss: %.2f, Reg Loss: %.2f' % (e + 1, self.l2_error, self.reg_loss))  
            if e % self._decay_rate == 0 and e != 0:
              self._lr /= 10.
            if self._total_loss < self._tol:
              print('Converged.')
              break
              
    def predict(self,X):
        # self._input_ph=X
        #import pdb;pdb.set_trace()
        
        out=self.s_w*X.unsqueeze(-1)

        #out=torch.sum(out,(1,2)) # <<<<<----------------------------------------------------------------Changed
        out=torch.sum(out,(2,3))
        out=out*self.d_w
        out=torch.sum(out,1)+self.bias
        self._predictions=out
        
#         print(out)
        return out