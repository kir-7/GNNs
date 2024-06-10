import torch
from torch import nn
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool


from models.layers import gLayer


class GNN(nn.Module):
    def __init__(self, n_layers=4, emb_dim=50, edge_dim=4, in_dim=10, out_dim=1, batch_size=4):

        super().__init__()

        self.batch_size = batch_size

        self.lin_in = Linear(in_dim, emb_dim)   

        self.convs = torch.nn.ModuleList()

        for i in range(n_layers):
            self.convs.append(gLayer(emb_dim, edge_dim))
        
        self.pool = global_mean_pool


        self.lin_pred = Linear(emb_dim, out_dim)
    

    def forward(self, data):

        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        h = self.lin_in(x)

        for conv in self.convs:
            h = h + conv(h, edge_index, edge_attr)
        
        h_graph = self.pool(h, self.batch_size)

        out = self.lin_pred(h_graph)

        return out.view(-1)
