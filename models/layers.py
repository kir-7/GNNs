import torch
from torch import nn
import torch.functional as F
from torch.nn import ReLU, SiLU

import numpy as np

import os

# gLinear is a linear layer of GNN that is of format graph-in graph-out and does'nt do any sort of pooling
# GNN have 3 types of pooling(information gathering from other nodes, edges and global) - message passing, convolutions and Pooling

class gLinear(nn.Module):
    def __init__(self, emb_dim=25, activation='relu', norm='layer', aggr='none', device='cpu'):
        
        super(gLinear, self).__init__()

        self.emb_dim = emb_dim
        self.activation = {"silu": SiLU(), "relu": ReLU()}[activation]

        self.norm = {"layer": torch.nn.LayerNorm, "batch": torch.nn.BatchNorm1d}[norm]

    
    def forward(self, x):
        pass
    
class gPool(nn.Module):
    
    def __init__(self, aggr):
        super(gPool, self).__init__()
