import torch
from torch import nn
from torch.nn import Linear, ReLU, SiLU, LayerNorm, BatchNorm1d, Sequential
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool


from models.layers import gLayer, gConv, GATConv


class GNN(nn.Module):
    def __init__(self, n_layers=4, emb_dim=50, edge_dim=4, in_dim=10, out_dim=1):

        super().__init__()
        
        self.emb_dim = emb_dim
         
        self.lin_in = Linear(in_dim, emb_dim)   

        self.convs = torch.nn.ModuleList()

        for i in range(n_layers):
            self.convs.append(gLayer(emb_dim, edge_dim))
        
        self.pool = global_mean_pool

        self.lin_pred = Linear(emb_dim, out_dim)
    
    def reset_parameters(self):
        super().reset_parameters()
        self.lin_in.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.pool.reset_parameters()
        self.lin_pred.reset_parameters()

    def forward(self, data):

        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        h = self.lin_in(x)

        for conv in self.convs:
            h = h + conv(h, edge_index, edge_attr)
        
        h_graph = self.pool(h, data.batch)
        
        out = self.lin_pred(h_graph)

        return out

    def __repr__(self):
        return f"MPNN Model({self.n_layers} layers, {self.emb_dim} emb_dim)"


class GCN(nn.Module):
    
    '''
        we have the SpatialGCN which uses the Spatial convolution layers, the filters used represent the dimentionality of the node embeddings
        so we can structure it in multiple ways

        we can use it similar to the U-Net architechture
        we can even add skip connections
        we can use an archiechture similar to that of highway networks

        But depending on the implementation, our model architechture will change so this is'nt a concrete model class just a neat way to bind all things together  

    '''

    def __init__(self, conv_layer_filters, emb_dim, in_dim, out_dim, edge_dim=4, aggr='add', egde_usage=False, activation='relu', norm='batch'):

        super().__init__()

        
        self.convs = torch.nn.ModuleList()
        self.conv_layer_filters = conv_layer_filters


        self.activation = {"relu":ReLU(), 'selu':SiLU()}[activation]
        self.norm = {"layer": LayerNorm, "batch": BatchNorm1d}[norm]

        # conv_layers_filters : List of filters that are going to be applied. 
        # eg: if  conv_layers_filters = [16, 32, 64] then 2 convolutional layers will be applied: 16->32 and 32->64

        self.lin_in = Sequential(Linear(in_dim, emb_dim), self.activation,  
                                 Linear(emb_dim, conv_layer_filters[0]), self.activation)

        for i in range(len(conv_layer_filters)-1):
            self.convs.append(gConv(conv_layer_filters[i], conv_layer_filters[i+1], edge_dim=edge_dim, aggr=aggr))


        self.pool = global_mean_pool

        self.lin_pred = Linear(conv_layer_filters[-1], out_dim)

                

    def reset_parameters(self):
        super().reset_parameters()
        for layer in self.lin_in:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.pool.reset_parameters()
        self.lin_pred.reset_parameters()

    def forward(self, data):
        
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = self.lin_in(x)

        for i in range(len(self.convs)):
            x = self.convs[i](x, edge_index, edge_attr)
        
        h_graph = self.pool(x, data.batch)
        
        out = self.lin_pred(h_graph)

        return out

    def __repr__(self):
        return f"GCN model (gconv, {self.conv_layer_filters})"

