import torch
from torch import nn
from torch.nn import Linear, ReLU, LeakyReLU, SiLU, Sigmoid, LayerNorm, BatchNorm1d, Sequential
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing, global_mean_pool, global_add_pool, GAE
from torch_geometric.data import Data 


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

    def __init__(self, filters : list, emb_dim, edge_dim, node_in_dim, edge_in_dim, latent_dim, mu=0, sigma=1.0, activation='leaky', norm='batch', dropout=0.3, add_self_loops=False, negative_slope=0.2, device='cpu'):

        super().__init__()

        self.emb_dim = emb_dim
        self.edge_dim = edge_dim
        self.node_in_dim = node_in_dim
        self.edge_in_dim = edge_in_dim
        self.latent_dim = latent_dim
        self.device = device

        self.mu = mu
        self.sigma = sigma

        self.gate_activation = Sigmoid()

        self.node_lin_in = Linear(node_in_dim, emb_dim, bias=False, weight_initializer='glorot')
        self.edge_lin_in = Linear(edge_in_dim, edge_dim, bias=False, weight_initializer='glorot')

        self.filters = [emb_dim] + filters

        self.convs = nn.ModuleList()
        self.activation = {"swish": SiLU(), "relu": ReLU(), 'gelu':nn.GELU(), 'leaky':LeakyReLU(negative_slope)}[activation]
        self.normalization = {"layer": torch.nn.LayerNorm, "batch": torch.nn.BatchNorm1d}[norm]
        self.edge_norm = self.normalization(edge_dim)
        self.norm = nn.ModuleList()
        self.out_norm = BatchNorm1d(self.latent_dim)
        self.dropout = dropout

        for i in range(len(self.filters)-1):
            self.convs.append(GATConv(self.filters[i], self.filters[i+1], edge_dim=edge_dim, add_self_loops=add_self_loops, update_edges=True))
            self.norm.append(self.normalization(self.filters[i+1]))

        self.phi_1 = Sequential(Linear(self.filters[-1], self.latent_dim, bias=False, weight_initializer='glorot'), self.normalization(self.latent_dim), self.activation)
        self.phi_2 = Sequential(Linear(self.filters[-1], self.latent_dim, bias=False, weight_initializer='glorot'), self.normalization(self.latent_dim), self.activation)
        self.phi_3 = Sequential(Linear(self.edge_dim, self.latent_dim, bias=False, weight_initializer='glorot'), self.normalization(self.latent_dim), self.activation)
        self.phi_4 = Sequential(Linear(self.edge_dim, self.latent_dim, bias=False, weight_initializer='glorot'), self.normalization(self.latent_dim), self.activation)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters
        self.node_lin_in.reset_parameters()
        self.edge_lin_in.reset_parameters()

        for norm_layer in self.norm:
            norm_layer.reset_parameters()

        self.phi_1[0].reset_parameters()
        self.phi_2[0].reset_parameters()
        self.phi_3[0].reset_parameters()
        self.phi_4[0].reset_parameters()
        self.out_norm.reset_parameters()

    def forward(self, x, edge_index, edge_attr, batch):

        norm = self.normalization(self.latent_dim).to(self.device)

        x = self.node_lin_in(x)
        edge_attr = self.edge_lin_in(edge_attr)


        for i in range(len(self.convs)):
            x, edge_attr = self.convs[i](x, edge_index, edge_attr)

            x = self.norm[i](x)
            x = self.activation(x)
            x = nn.functional.dropout(x, p = self.dropout, training=self.training)

            edge_attr = self.edge_norm(edge_attr)
            edge_attr = self.activation(edge_attr)
            edge_attr = nn.functional.dropout(edge_attr, p = self.dropout, training=self.training)

        out = self.edge_updater(edge_index, x=x, edge_attr=edge_attr)


        out = global_add_pool(out, batch)
        out = norm(out)
        out = self.activation(out)

        return out

    def edge_update(self, x_i, x_j, edge_attr):

        out = self.phi_1(x_i) + self.phi_2(x_j) + self.phi_3(edge_attr)
        out = self.gate_activation(out)
        out = out * self.phi_4(edge_attr)

        return out

    
class GraphDecoder(MessagePassing):
    def __init__(self, filters : list, latent_dim, M, R, emb_dim, edge_dim, activation='leaky', norm='batch', add_self_loops=False, batch_size=32, dropout=0.3, negative_slope=0.2):

        super().__init__()

        self.latent_dim = latent_dim
        self.M = M
        self.R = R
        self.emb_dim = emb_dim
        self.edge_dim = edge_dim
        self.filters = filters
        self.batch_size = batch_size

        self.activation = {"swish": SiLU(), "relu": ReLU(), 'leaky':LeakyReLU(negative_slope)}[activation]
        self.normalization = {"layer": torch.nn.LayerNorm, "batch": torch.nn.BatchNorm1d}[norm]
        self.dropout = dropout

        self.norm = nn.ModuleList()
        self.edge_norm = self.normalization(self.edge_dim)

        self.mlp = Sequential(
            Linear(self.latent_dim, 512, bias=True, weight_initializer='glorot'),
            BatchNorm1d(512),
            self.activation,
            Linear(512, 512, bias=True, weight_initializer='glorot'),
            BatchNorm1d(512),
            self.activation,
            Linear(512, 256, bias=True, weight_initializer='glorot'),
            BatchNorm1d(256),
            self.activation,
            Linear(256, self.M * self.R, bias=True, weight_initializer='glorot')
        )

        self.convs = nn.ModuleList()
        for i in range(len(self.filters)-1):
            self.convs.append(GATConv(filters[i], filters[i+1], edge_dim=self.edge_dim, add_self_loops=add_self_loops, update_edges=True))
            self.norm.append(self.normalization(filters[i+1]))

        self.edge_mlp = Sequential(
                Linear(self.edge_dim, 128),
                BatchNorm1d(128),
                self.activation,
                Linear(128, 64),
                BatchNorm1d(64),
                self.activation,
                Linear(64, 16),
                BatchNorm1d(16),
                self.activation,
                Linear(16, 4)
        )

        self.lin_in_node = Linear(5, self.filters[0], bias=False, weight_initializer='glorot')
        self.lin_in_edge = Linear(self.latent_dim, self.edge_dim)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters
        self.lin_in_node.reset_parameters()
        self.lin_in_edge.reset_parameters()
        for layer in self.edge_mlp.children():
            if isinstance(layer, Linear):
                layer.reset_parameters()
        for layer in self.mlp.children():
            if isinstance(layer, Linear):
                layer.reset_parameters()



    def forward(self, z):

        out = self.mlp(z)
        out = out.view(-1, self.M, self.R)
        out = torch.nn.functional.softmax(out, dim=2)
        z_boa = torch.max(out, dim=2).indices

        x, edge_index, edge_attr, batch_map = self.construct_graph(z_boa, z, self.lin_in_edge)


        x = self.lin_in_node(x)

        for i in range(len(self.convs)):

            x, edge_attr = self.convs[i](x, edge_index, edge_attr)
            x = self.norm[i](x)
            x = self.activation(x)
            x = nn.functional.dropout(x, p = self.dropout, training=self.training)

            edge_attr = self.edge_norm(edge_attr)
            edge_attr = self.activation(edge_attr)
            edge_attr = nn.functional.dropout(edge_attr, p = self.dropout, training=self.training)

        edge_attr = self.edge_mlp(edge_attr)
        edge_attr = torch.nn.functional.softmax(edge_attr, dim=1)

        return out.view(-1, self.R), edge_index, edge_attr, batch_map[edge_index[0]]


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


class MGVAE(GAE):
    def __init__(self, encoder, decoder, MAX_LOGSTD=10, batch_size=32, device='cpu'):
        super().__init__(encoder, decoder)
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)

        assert encoder.latent_dim == decoder.latent_dim
        self.latent_dim = encoder.latent_dim
        self.MAX_LOGSTD = MAX_LOGSTD
        self.batch_size = batch_size
        self.M, self.R = self.decoder.M, self.decoder.R

        self.mean_layer = Linear(self.latent_dim, self.latent_dim, bias=False, weight_initializer='glorot')
        self.logvar_layer = Linear(self.latent_dim, self.latent_dim, bias=False, weight_initializer='glorot')

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.mean_layer.reset_parameters()
        self.logvar_layer.reset_parameters()

    def reparametrize(self, mu, logstd):
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu

    def encode(self, x, edge_index, edge_attr, batch):

        z = self.encoder(x, edge_index, edge_attr, batch)


        mean, logvar = self.mean_layer(z), self.logvar_layer(z).clamp(max=self.MAX_LOGSTD)

        return mean, logvar

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x, edge_index, edge_attr, batch):


        mean, logvar = self.encode(x, edge_index, edge_attr, batch)


        z = self.reparametrize(mean, logvar)

        z_boa, edge_index_out, edge_attr_out, edge_batch_out = self.decode(z)

        return z_boa, edge_attr_out, mean, logvar

    def reconstruct_molecules(self, z_boa, edge_index, edge_attr, edge_batch, batch_size):

        reconstructed_molecules = []

        for i in range(batch_size):
            # Get nodes for this molecule
            molecule_atoms = z_boa[i]
            num_atoms = molecule_atoms.sum().int().item()

            # Create x (atom types)
            x = torch.zeros((num_atoms, z_boa.size(1)), dtype=torch.float)
            atom_idx = 0
            for atom_type in range(z_boa.size(1)):
                num_of_this_atom = molecule_atoms[atom_type].int().item()
                x[atom_idx:atom_idx+num_of_this_atom, atom_type] = 1
                atom_idx += num_of_this_atom

            # Get edges for this molecule
            mask = edge_batch == i
            molecule_edge_index = edge_index[:, mask]
            molecule_edge_attr = edge_attr[mask]

            # Create one-hot encoded edge_attr
            edge_attr_onehot = torch.zeros((molecule_edge_attr.size(0), 4), dtype=torch.float)
            edge_attr_onehot.scatter_(1, molecule_edge_attr.unsqueeze(1), 1)

            # Create Data object
            data = Data(
                x=x,
                edge_index=molecule_edge_index,
                edge_attr=edge_attr_onehot
            )

            reconstructed_molecules.append(data)

        return reconstructed_molecules
