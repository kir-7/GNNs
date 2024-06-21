
import torch
from torch import nn
import torch.functional as F
from torch.nn import ReLU, SiLU, Linear, Sequential, LayerNorm, BatchNorm1d

import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.nn.inits import zeros
from torch_geometric.utils import add_self_loops, degree, get_laplacian
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
    
'''
    After Spatial Convolution next step is more methods of convolutions and building different models using convolutions

    after GCN next step is GIN and then GAN

    along with improvement in the architechture need to improve the functinality as well things like doing 'graph rewiring' and stuff 
'''

class ChebConv(MessagePassing):

    '''  From PyG official documentation  '''

    def __init__(self, in_channels, out_channels, k, normalization='sym', bias=True):
        
        self.in_channels = in_channels
        self,out_channels - out_channels
        self.normalization = normalization
        self.k = k

        self.lins = torch.nn.ModuleList(
            Linear(in_channels, out_channels, bias=False, weight_initializer='glorot') for _ in range(k)
        )

        if bias:
            self.bias = torch.Parameter(torch.Tensor(out_channels))
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        super().reset_parameters()
        for lin in self.lins:
            lin.resset_parameters()
        zeros(self.bias)

    def __norm__(self, edge_index, num_nodes, edge_weight, normalization, lambda_max=None, dtype=None, batch=None):
        
        edge_index, edge_weight = get_laplacian(edge_index, edge_weight, normalization, num_nodes=num_nodes)

        if lambda_max is None:
            lambda_max = 2.0 * edge_weight.max()
        else:
            lambda_max = torch.Tensor(lambda_max, dtype=dtype, device=edge_weight.device)
        

        if batch is not None and lambda_max.numel() > 1:
            lambda_max = lambda_max[batch[edge_index[0]]]
        

        edge_weight = (2.0 * edge_weight) / lambda_max
        edge_weight.masked_fill_(edge_weight == float('inf'), 0)

        loop_mask = edge_index[0] == edge_index[1]
        edge_weight[loop_mask] -= 1

        return edge_index, edge_weight

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


    def forward(self, x, edge_index, edge_weight, batch, lambda_max):

        edge_index, norm = self.__norm__(edge_index, x.size(self.node_dim), edge_weight, self.normalization, lambda_max, x.dtype, batch)
        
        Tx_0 = x
        Tx_1 = x
        out = self.lins[0](Tx_0)

        if len(self.lins) > 1:
            Tx_1 = self.propogate(edge_index, x=x, norm=norm)
        
        for lin in self.lins:
            
            Tx_2 = self.propagate(edge_index, x=Tx_1, nomr=norm)
            Tx_2 = 2.*Tx_2 - Tx_0

            out = out + lin.forward(Tx_2)

            Tx_0 , Tx_1 = Tx_1, Tx_2


        if self.bais is not None:
            out = out + self.bias
        

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, K={len(self.lins)}, '
                f'normalization={self.normalization})')


class KipfConv(MessagePassing):

    ''' 
        An extension to Cheb conv, the work of Kipf and Welling, which simplifies the ChebNet approach by utilizing only local information, setting K = 2  
        https://medium.com/@jlcastrog99/spectral-graph-convolutions-c7241af4d8e2#:~:text=the%20work%20of%20Kipf%20and%20Welling%2C%20which%20simplifies%20the%20ChebNet%20approach%20by%20utilizing%20only%20local%20information%2C%20setting%20K%20%3D%202
    '''
    
    def __init__(self):
        pass

