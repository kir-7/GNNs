
import torch
from torch import nn
import torch.functional as F
from torch.nn import ReLU, SiLU, Linear, Sequential, LayerNorm, BatchNorm1d

import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import add_self_loops, degree
from torch_scatter import scatter


# GNN have 3 types of pooling(information gathering from other nodes, edges and global) - message passing, convolutions and Pooling
# https://github.com/chaitjo/geometric-gnn-dojo/blob/main/geometric_gnn_101.ipynb



# gLayer also uses the edge embeddings to gather information (the edge_attr in message is for edge embedding gathering)
class gLayer(MessagePassing):
    def __init__(self, emb_dim=25, edge_dim=25, activation='relu', norm='batch', aggr='add', device='cpu'):
        
        super().__init__(aggr=aggr)

        self.emb_dim = emb_dim
        self.edge_dim = edge_dim

        self.activation = {"silu": SiLU(), "relu": ReLU()}[activation]

        self.norm = {"layer": LayerNorm, "batch": BatchNorm1d}[norm]

        # MLP `\psi` for computing messages `m_ij`
        # Implemented as a stack of Linear->BN->ReLU->Linear->BN->ReLU
        # dims: (2d + d_e) -> d
        self.mlp_msg = Sequential(
            Linear(2*emb_dim + self.edge_dim, emb_dim), self.norm(emb_dim), self.activation,
            Linear(emb_dim, emb_dim), self.norm(emb_dim), ReLU()
          )
        
        # MLP `\phi` for computing updated node features `h_i^{l+1}`
        # Implemented as a stack of Linear->BN->ReLU->Linear->BN->ReLU
        # dims: 2d -> d
        self.mlp_upd = Sequential(
            Linear(2*emb_dim, emb_dim), self.norm(emb_dim), self.activation,
            Linear(emb_dim, emb_dim), self.norm(emb_dim), self.activation
        )

    def reset_parameters(self):
        super().reset_parameters()
        
        for layer_msg in self.mlp_msg.children():
            layer_msg.reset_parameters()

        for layer_upd in self.mlp_upd.children():
            layer_upd.reset_parameters()


    def forward(self, h, edge_index, edge_attr):
        
        out = self.propagate(edge_index, h=h, edge_attr=edge_attr)
        return out

    def message(self, h_i, h_j, edge_attr):
       
        msg = torch.cat([h_i, h_j, edge_attr], dim=-1)
        return self.mlp_msg(msg)
    
    def aggregate(self, inputs, index):
       
        return scatter(inputs, index, dim=self.node_dim, reduce=self.aggr)
    
    def update(self, aggr_out, h):

        upd_out = torch.cat([h, aggr_out], dim=-1)
        return self.mlp_upd(upd_out)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(emb_dim={self.emb_dim}, aggr={self.aggr})')
    

class gConv(MessagePassing):
    
    '''
        2 types of graph convolutions: spectral and spatial

        Spectral methods are not inherently localized; they consider the entire graph structure via the graph Laplacian.
        Spatial methods are inherently localized; they operate directly on a node’s local neighborhood.
        
        Ultimately the choice of spectral or spatial depends on the application but spatial is more widely used for its scalability, flexibility, and ease of implementation

        But spectral is more sophisticated and can be improved to maybe produce better results 

        Here we are implementing spatial convolution

        There are multiple types of Spatial convolutions, here  


    '''

    def __init__(self, in_channels, out_channels, bias=True, activation='relu', aggr='add'):
        
        super().__init__(aggr=aggr)

        
        self.activation = {"silu": SiLU(), "relu": ReLU()}[activation]

        self.lin = Linear(in_channels, out_channels, bias= False, weight_initializer='glorot', activation=self.activation)

        if bias:
            self.bias = torch.Parameter(torch.empty(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, edge_index):
        
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        x = self.lin(x)

        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        out = self.propagate(edge_index, x=x, norm=norm)

        out = out + self.bias

        return out

    def message(self, x_j, norm):

        return norm.view(-1, 1) * x_j