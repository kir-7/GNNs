import torch
import torch.nn.functional as F
import torch_geometric

import numpy as np
import os
import random
import time

from layers import gLayer



if __name__ == "__main__":

    #  sanity checking if the layer works input is h{i} = (node, node_dim) 
    #  output is also of same shape but the h{i+1} 

    torch.manual_seed(0)

    layer = gLayer()
    print("layer created successfully!")
    trial_h = torch.randn(10, 25)
    trial_edge = torch.randint(1, 10, (2, 10))
    trial_edge_emb = torch.randn(10, 25)
    with torch.no_grad():
        out = layer(trial_h, trial_edge, trial_edge_emb)
        print(out.shape)
    
    print(layer)


    del layer