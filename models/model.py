import torch
from torch import nn
from torch.nn import Linear, ReLU, SiLU, Sigmoid, LayerNorm, BatchNorm1d, Sequential
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing, global_mean_pool, global_add_pool


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

class GraphEncoder(MessagePassing):

    def __init__(self, filters : list, emb_dim, edge_dim, node_in_dim, edge_in_dim, latent_dim, mu=0, sigma=1.0, activation='relu', norm='batch', add_self_loops=False):

        super().__init__()
        
        self.emb_dim = emb_dim
        self.edge_dim = edge_dim
        self.node_in_dim = node_in_dim
        self.edge_in_dim = edge_in_dim
        self.latent_dim = latent_dim
        
        self.mu = mu
        self.sigma = sigma

        self.activation = self.activation = {"silu": SiLU(), "relu": ReLU(), 'sigmoid':Sigmoid()}[activation]
        self.norm = {"layer": LayerNorm, "batch": BatchNorm1d}[norm]
        
        
        self.gate_activation = Sigmoid()

        self.node_lin_in = Linear(node_in_dim, emb_dim, bias=False, weight_initializer='glorot')
        self.edge_lin_in = Linear(edge_in_dim, edge_dim, bias=False, weight_initializer='glorot')

        self.filters = [emb_dim] + filters
                
        self.convs = nn.ModuleList()
        
        for i in range(len(self.filters)-1):
            self.convs.append(GATConv(self.filters[i], self.filters[i+1], edge_dim=edge_dim, add_self_loops=add_self_loops, update_edges=True))

        self.phi_1 = Linear(self.filters[-1], self.latent_dim, bias=False, weight_initializer='glorot')
        self.phi_2 = Linear(self.filters[-1], self.latent_dim, bias=False, weight_initializer='glorot')
        self.phi_3 = Linear(self.edge_dim, self.latent_dim, bias=False, weight_initializer='glorot')
        self.phi_4 = Linear(self.edge_dim, self.latent_dim, bias=False, weight_initializer='glorot')
        
        
    def forward(self, x, edge_index, edge_attr, batch):

        x = self.node_lin_in(x)
        edge_attr = self.edge_lin_in(edge_attr)
        
        for i in range(len(self.convs)): 
            x, edge_attr = self.convs[i](x, edge_index, edge_attr)
            
            x = self.norm(self.filters[i+1])(x)
            x = self.activation(x)
            
            edge_attr = self.norm(self.edge_dim)(edge_attr)
            edge_attr = self.activation(edge_attr)

        out = self.edge_updater(edge_index, x=x, edge_attr=edge_attr)
        

        out = global_add_pool(out, batch)
        
        return out
    
    def edge_update(self, x_i, x_j, edge_attr):
        
        out = self.phi_1(x_i) + self.phi_2(x_j) + self.phi_3(edge_attr)
        out = self.gate_activation(out)
        out = out * self.phi_4(edge_attr) 

        return out



class GraphDecoder(MessagePassing):
    def __init__(self, filters : list, latent_dim, M, R, emb_dim, edge_dim, add_self_loops=False, batch_size=32):
        
        super().__init__()
        
        self.latent_dim = latent_dim
        self.M = M
        self.R = R
        self.emb_dim = emb_dim
        self.edge_dim = edge_dim
        self.filters = filters
        self.batch_size = batch_size

        self.mlp = Sequential(
            Linear(self.latent_dim, 256, bias=True, weight_initializer='glorot'),
            ReLU(),
            Linear(256, 256, bias=True, weight_initializer='glorot'),
            ReLU(),
            Linear(256, 128, bias=True, weight_initializer='glorot'),
            ReLU(),
            Linear(128, self.M * self.R, bias=True, weight_initializer='glorot')
        )

        self.convs = nn.ModuleList()
        for i in range(len(self.filters)-1):
            self.convs.append(GATConv(filters[i], filters[i+1], edge_dim=self.edge_dim, add_self_loops=add_self_loops, update_edges=True))
        
        self.edge_mlp = Sequential(
                Linear(self.edge_dim, 128),
                ReLU(),
                Linear(128, 64),
                ReLU(),
                Linear(64, 16),
                ReLU(),
                Linear(16, 4)
        )

        self.lin_in_node = Linear(5, self.filters[0], bias=False, weight_initializer='glorot')
        self.lin_in_edge = Linear(self.latent_dim, self.edge_dim)
        
    
    def forward(self, z):

        out = self.mlp(z)
        out = out.view(-1, self.M, self.R)
        out = torch.nn.functional.softmax(out, dim=2)
        z_boa = torch.max(out, dim=2).indices
        
        x, edge_index, edge_attr, batch_map = self.construct_graph(z_boa, z, self.lin_in_edge)

        
        x = self.lin_in_node(x)

        for i in range(len(self.convs)):
            x, edge_attr = self.convs[i](x, edge_index, edge_attr)

        edge_attr = self.edge_mlp(edge_attr)

        return edge_attr, batch_map[edge_index[0]]

    
    def construct_graph(self, z_boa, z, project):
        batch_size, atom_types = z_boa.shape
        device = z_boa.device
    
        # Calculate the total number of atoms
        total_atoms = z_boa.sum().int().item()
    
        # Create x: one-hot encoding of atom types
        x = torch.zeros(total_atoms, atom_types, device=device)
        start_idx = 0
        batch_map = torch.zeros(total_atoms, dtype=torch.long, device=device)
        
        for i in range(batch_size):
            num_atoms_batch = z_boa[i].sum().int().item()
            end_idx = start_idx + num_atoms_batch
            
            # One-hot encoding for atoms in this batch
            x[start_idx:end_idx] = torch.nn.functional.one_hot(
                torch.arange(atom_types, device=device).repeat_interleave(z_boa[i].int()),
                num_classes=atom_types
            )
            
            # Map these atoms to their batch
            batch_map[start_idx:end_idx] = i
            
            start_idx = end_idx
    
        # Create edge_index and prepare edge_attr
        edge_index = []
        edge_attr = []
        start_idx = 0
        
        
        for i in range(batch_size):
            num_atoms_batch = z_boa[i].sum().int().item()
            end_idx = start_idx + num_atoms_batch
            
            # Fully connect atoms within this batch
            batch_edge_index = torch.combinations(torch.arange(start_idx, end_idx, device=device), r=2).t()
            edge_index.append(batch_edge_index)
            
            # Project z for this batch to edge_attr_dim
            batch_z_proj = project(z[i].float().unsqueeze(0))  # [1, edge_attr_dim]
            
            # Repeat the projected z_boa for each edge in this batch
            num_edges_batch = batch_edge_index.shape[1]
            batch_edge_attr = batch_z_proj.repeat(num_edges_batch, 1)  # [num_edges_batch, edge_attr_dim]
            
            edge_attr.append(batch_edge_attr)
            
            start_idx = end_idx
    
        edge_index = torch.cat(edge_index, dim=1)
        edge_attr = torch.cat(edge_attr, dim=0)
    
        return x, edge_index, edge_attr, batch_map
    