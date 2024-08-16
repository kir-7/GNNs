
import torch
from torch import nn
import torch.functional as F
from torch.nn import ReLU, SiLU, Sequential, LayerNorm, BatchNorm1d, Parameter


import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.nn import MessagePassing, Linear, global_mean_pool
from torch_geometric.nn.inits import zeros, glorot, reset
from torch_geometric.nn.models import MLP
from torch_geometric.utils import add_self_loops, degree, get_laplacian, remove_self_loops, spmm
from torch_scatter import scatter
import torch_sparse


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
            if hasattr(layer_msg, 'reset_parameters'):
                layer_msg.reset_parameters()
            

        for layer_upd in self.mlp_upd.children():
            if hasattr(layer_upd, 'reset_parameters'):
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

    def __init__(self, in_channels, out_channels, edge_dim=None, bias=True, activation='relu', aggr='add'):
        
        super().__init__(aggr=aggr)

        
        self.activation = {"silu": SiLU(), "relu": ReLU()}[activation]

        self.lin = Sequential(Linear(in_channels, out_channels, bias= False, weight_initializer='glorot'), self.activation)
        
        self.edge_weights = edge_dim

        if bias:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.bais = None

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        for layer in self.lin.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, edge_index, edge_attr):

        if self.edge_weights is not None:
        
            edge_index, edge_attr = add_self_loops(edge_index, edge_attr, num_nodes=x.size(0))
        else:
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

    
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        if self.edge_weights is not None:

            deg_inv_sqrt_row = deg_inv_sqrt[row].unsqueeze(-1)
            deg_inv_sqrt_col = deg_inv_sqrt[col].unsqueeze(-1)
            norm = deg_inv_sqrt_row * edge_attr * deg_inv_sqrt_col

        else:
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
            
        x = self.lin(x)

        out = self.propagate(edge_index, x=x, norm=norm)

        out = out + self.bias

        return out

    def message(self, x_j, norm):

        return torch.matmul(norm, x_j)
    
'''
    After Spatial Convolution next step is more methods of convolutions and building different models using convolutions

    after GCN next step is GIN and then GAT

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
            self.bias = Parameter(torch.Tensor(out_channels))
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

'''
Coming up GATs and random walks (very messed up ordering)
'''
class GATConv(MessagePassing):
    def __init__(
            self,
            in_channels,
            out_channels: int,
            heads: int = 1,
            concat: bool = True,
            negative_slope: float = 0.2,
            dropout: float = 0.0,
            add_self_loops: bool = True,
            edge_dim = None,
            fill_value = 'mean',
            bias: bool = True,
            update_edges = False,
            **kwargs,
        ):
            kwargs.setdefault('aggr', 'add')
            super().__init__(node_dim=0, **kwargs)

            self.in_channels = in_channels
            self.out_channels = out_channels
            self.heads = heads
            self.concat = concat
            self.negative_slope = negative_slope
            self.dropout = dropout
            self.add_self_loops = add_self_loops
            self.edge_dim = edge_dim
            self.fill_value = fill_value
            self.update_edges = update_edges

            # In case we are operating in bipartite graphs, we apply separate
            # transformations 'lin_src' and 'lin_dst' to source and target nodes:
            self.lin = self.lin_src = self.lin_dst = None
            if isinstance(in_channels, int):
                self.lin = Linear(in_channels, heads * out_channels, bias=False,
                                weight_initializer='glorot')
            else:
                self.lin_src = Linear(in_channels[0], heads * out_channels, False,
                                    weight_initializer='glorot')
                self.lin_dst = Linear(in_channels[1], heads * out_channels, False,
                                    weight_initializer='glorot')

            # The learnable parameters to compute attention coefficients:
            self.att_src = Parameter(torch.empty(1, heads, out_channels))
            self.att_dst = Parameter(torch.empty(1, heads, out_channels))

            if edge_dim is not None:
                self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False,
                                    weight_initializer='glorot')
                self.att_edge = Parameter(torch.empty(1, heads, out_channels))

                if update_edges:
                    self.phi_1 = Linear(edge_dim, edge_dim, bias=False, weight_initializer='glorot')
                    self.phi_2 = Linear(in_channels, edge_dim, bias=False, weight_initializer='glorot')
                    self.phi_3 = Linear(in_channels, edge_dim, bias=False, weight_initializer='glorot')

            else:
                self.lin_edge = None
                self.register_parameter('att_edge', None)

            if bias and concat:
                self.bias = Parameter(torch.empty(heads * out_channels))
            elif bias and not concat:
                self.bias = Parameter(torch.empty(out_channels))
            else:
                self.register_parameter('bias', None)

            self.reset_parameters()

    def reset_parameters(self):
            super().reset_parameters()
            if self.lin is not None:
                self.lin.reset_parameters()
            if self.lin_src is not None:
                self.lin_src.reset_parameters()
            if self.lin_dst is not None:
                self.lin_dst.reset_parameters()
            if self.lin_edge is not None:
                self.lin_edge.reset_parameters()
            if self.update_edges:
                self.phi_1.reset_parameters()
                self.phi_2.reset_parameters()
                self.phi_3.reset_parameters()

            glorot(self.att_src)
            glorot(self.att_dst)
            glorot(self.att_edge)
            zeros(self.bias)


    def forward(  # noqa: F811
            self,
            x,
            edge_index,
            edge_attr = None,
            size = None,
            return_attention_weights = None,
        ):


            H, C = self.heads, self.out_channels
            x_copy = x.clone()

            # We first transform the input node features. If a tuple is passed, we
            # transform source and target node features via separate weights:
            if isinstance(x, torch.Tensor):
                assert x.dim() == 2, "Static graphs not supported in 'GATConv'"

                if self.lin is not None:
                    x_src = x_dst = self.lin(x).view(-1, H, C)
                else:
                    # If the module is initialized as bipartite, transform source
                    # and destination node features separately:
                    assert self.lin_src is not None and self.lin_dst is not None
                    x_src = self.lin_src(x).view(-1, H, C)
                    x_dst = self.lin_dst(x).view(-1, H, C)

            else:  # Tuple of source and target node features:
                x_src, x_dst = x
                assert x_src.dim() == 2, "Static graphs not supported in 'GATConv'"

                if self.lin is not None:
                    # If the module is initialized as non-bipartite, we expect that
                    # source and destination node features have the same shape and
                    # that they their transformations are shared:
                    x_src = self.lin(x_src).view(-1, H, C)
                    if x_dst is not None:
                        x_dst = self.lin(x_dst).view(-1, H, C)
                else:
                    assert self.lin_src is not None and self.lin_dst is not None

                    x_src = self.lin_src(x_src).view(-1, H, C)
                    if x_dst is not None:
                        x_dst = self.lin_dst(x_dst).view(-1, H, C)

            x = (x_src, x_dst)

            # Next, we compute node-level attention coefficients, both for source
            # and target nodes (if present):
            alpha_src = (x_src * self.att_src).sum(dim=-1)
            alpha_dst = None if x_dst is None else (x_dst * self.att_dst).sum(-1)
            alpha = (alpha_src, alpha_dst)

            if self.add_self_loops:
                if isinstance(edge_index, torch.Tensor):
                    # We only want to add self-loops for nodes that appear both as
                    # source and target nodes:
                    num_nodes = x_src.size(0)
                    if x_dst is not None:
                        num_nodes = min(num_nodes, x_dst.size(0))
                    num_nodes = min(size) if size is not None else num_nodes
                    edge_index, edge_attr = remove_self_loops(
                        edge_index, edge_attr)
                    edge_index, edge_attr = add_self_loops(
                        edge_index, edge_attr, fill_value=self.fill_value,
                        num_nodes=num_nodes)
                elif isinstance(edge_index, SparseTensor):
                    if self.edge_dim is None:
                        edge_index = torch_sparse.set_diag(edge_index)
                    else:
                        raise NotImplementedError(
                            "The usage of 'edge_attr' and 'add_self_loops' "
                            "simultaneously is currently not yet supported for "
                            "'edge_index' in a 'SparseTensor' form")

            # edge_updater_type: (alpha: OptPairTensor, edge_attr: OptTensor)
            alpha, e_out = self.edge_updater(edge_index, x=x_copy, alpha=alpha, edge_attr=edge_attr,
                                    size=size)

            # propagate_type: (x: OptPairTensor, alpha: Tensor)
            out = self.propagate(edge_index, x=x, alpha=alpha, size=size)

            if self.concat:
                out = out.view(-1, self.heads * self.out_channels)
            else:
                out = out.mean(dim=1)

            if self.bias is not None:
                out = out + self.bias

            return out, e_out


    def edge_update(self, x_j, x_i, alpha_j, alpha_i,
                        edge_attr, index, ptr,
                        dim_size) :

            alpha = alpha_j if alpha_i is None else alpha_j + alpha_i

            e_out = edge_attr

            if index.numel() == 0:
                return alpha
            if edge_attr is not None and self.lin_edge is not None:
                if edge_attr.dim() == 1:
                    edge_attr = edge_attr.view(-1, 1)


                if self.update_edges:
                    edge_attr_c = edge_attr.clone()
                    e_out = self.phi_1(edge_attr_c) + self.phi_2(x_i) + self.phi_3(x_j)

                edge_attr = self.lin_edge(edge_attr)
                edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
                alpha_edge = (edge_attr * self.att_edge).sum(dim=-1)
                alpha = alpha + alpha_edge

            alpha = F.leaky_relu(alpha, self.negative_slope)
            alpha = torch_geometric.utils.softmax(alpha, index, ptr, dim_size)
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)

            return alpha, e_out

    def message(self, x_j, alpha):
        return alpha.unsqueeze(-1) * x_j

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')

'''
In all of these models we are using the edge attributes to update the node attributes but we are never updateing the edge attributes themselves:

a good convolution example:https://arxiv.org/pdf/1906.01227

'''


class GIN(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_dim, aggr = 'add', eps = 0, train_eps=False):
        
        super().__init__(aggr=aggr)
        self.aggr = aggr
        self.initial_eps = eps
        
        if train_eps:
            self.eps = torch.nn.Parameter(torch.empty(1))
        else:
            self.register_buffer('eps', torch.empty(1))

        self.mlp = MLP([in_channels, out_channels, out_channels], act=self.act, act_first=self.act_first, norm=self.norm, norm_kwargs=self.norm_kwargs)
        
        if edge_dim is not None:
            if isinstance(self.mlp, torch.nn.Sequential):
                mlp = self.mlp[0]
            if hasattr(nn, 'in_features'):
                in_channels = mlp.in_features
            elif hasattr(nn, 'in_channels'):
                in_channels = mlp.in_channels

            self.lin_in = Linear(edge_dim, in_channels)    
        else:
            self.lin_in = None

        self.reset_parameters()



    def reset_parameters(self) -> None:
        super().reset_parameters()
        reset(self.mlp)
        self.eps.data.fill_(self.initial_eps)
        if self.lin_in is not None:
            self.lin_in.reset_parameters()
    

    def forward(self, x, edge_index, edge_attr, size=None):

        x = (x, x)

        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

        x_r = x[1]

        out = out + (1 + self.eps)*x_r

        return self.mlp(out)

    def message(self, x_j):
        if self.lin is not None:
            edge_attr = self.lin(edge_attr)

        return (x_j + edge_attr).relu()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(mlp={self.mlp})'
