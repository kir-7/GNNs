import torch
import torch.nn.functional as F
import torch_geometric

import numpy as np
import os
import random
import time

from layers import gLayer


if __name__ == "__main__":
    layer = gLayer(25, 25, activation='relu')
    print("layer created successfully!")
    del layer