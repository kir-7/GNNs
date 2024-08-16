import torch
import torch.nn.functional as F
import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.utils import remove_self_loops, to_dense_adj, dense_to_sparse
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import time
from tqdm import tqdm


def get_params(model):

    total_param = 0
    for param in model.parameters():
        total_param += np.prod(list(param.data.size()))

    return total_param


def permute_graph(data, perm):
    """Helper function for permuting PyG Data object attributes consistently.
    """
    # Permute the node attribute ordering
    data.x = data.x[perm]
    data.pos = data.pos[perm]
    data.z = data.z[perm]
    data.batch = data.batch[perm]

    # Permute the edge index
    adj = to_dense_adj(data.edge_index)
    adj = adj[:, perm, :]
    adj = adj[:, :, perm]
    data.edge_index = dense_to_sparse(adj)[0]

    # Note: 
    # (1) While we originally defined the permutation matrix P as only having 
    #     entries 0 and 1, its implementation via `perm` uses indexing into 
    #     torch tensors, instead. 
    # (2) It is cumbersome to permute the edge_attr, so we set it to constant 
    #     dummy values. For any experiments beyond unit testing, all GNN models 
    #     use the original edge_attr.

    return data

def permutation_invariance_unit_test(module, dataloader):
    """Unit test for checking whether a module (GNN model) is 
    permutation invariant.
    """
    it = iter(dataloader)
    data = next(it)

    # Set edge_attr to dummy values (for simplicity)
    data.edge_attr = torch.zeros(data.edge_attr.shape)

    # Forward pass on original example
    out_1 = module(data)

    # Create random permutation
    perm = torch.randperm(data.x.shape[0])
    data = permute_graph(data, perm)

    # Forward pass on permuted example
    out_2 = module(data)

    # Check whether output varies after applying transformations
    return torch.allclose(out_1, out_2, atol=1e-04)


def permutation_equivariance_unit_test(module, dataloader):
    """Unit test for checking whether a module (GNN layer) is 
    permutation equivariant.
    """
    it = iter(dataloader)
    data = next(it)

    # Set edge_attr to dummy values (for simplicity)
    data.edge_attr = torch.zeros(data.edge_attr.shape)

    # Forward pass on original example
    out_1 = module(data.x, data.edge_index, data.edge_attr)

    # Create random permutation
    perm = torch.randperm(data.x.shape[0])
    data = permute_graph(data, perm)

    # Forward pass on permuted example
    out_2 = module(data.x, data.edge_index, data.edge_attr)

    # Check whether output varies after applying transformations
    return torch.allclose(out_1[perm], out_2, atol=1e-04) 


def pad_edge_attr(true_edge_attr, pred_edge_attr, device):
    # Define the "none" edge
    none_edge = torch.tensor([1, 0, 0, 0], device=true_edge_attr.device, dtype=true_edge_attr.dtype)

    # Determine the number of edges
    true_edge_count = true_edge_attr.size(0)
    pred_edge_count = pred_edge_attr.size(0)

    # Pad true_edge_attr if necessary
    if true_edge_count < pred_edge_count:
        padding = none_edge.repeat(pred_edge_count - true_edge_count, 1)
        true_edge_attr = torch.cat([true_edge_attr, padding], dim=0)

    # Pad pred_edge_attr if necessary
    if pred_edge_count < true_edge_count:
        padding = none_edge.repeat(true_edge_count - pred_edge_count, 1)
        pred_edge_attr = torch.cat([pred_edge_attr, padding], dim=0)

    true_edge_attr = true_edge_attr.to(device)
    pred_edge_attr = pred_edge_attr.to(device)
    return true_edge_attr, pred_edge_attr

def create_atom_count_vectors(d, M, R, batch_size, device):

    atom_count_vectors = torch.zeros(batch_size, M, R, dtype=torch.float).to(device)

    for i in range(batch_size):

        atom_counts = torch.sum(d[0].x[:, 0:5].transpose(0, 1), dim=1)
        onehot_vectors = torch.zeros(5, R)

        # Create masks for each atom type and set the corresponding positions to 1
        for k in range(5):
            atom_count = int(atom_counts[k].item())  # Convert to integer
            if atom_count < R:
                onehot_vectors[k, atom_count] = 1
            else:
                onehot_vectors[k, R - 1] = 1

        atom_count_vectors[i, :] = onehot_vectors



    return atom_count_vectors.view(-1, R)


def molecular_loss(true_edge_attr, pred_edge_attr, true_node_prob, pred_node_prob, mu, logvar, lambda_e=1.0, lambda_a=1.0, lambda_kl=0.1, epsilon=1e-8):

    edge_loss = F.cross_entropy(pred_edge_attr, true_edge_attr)

    node_loss = F.cross_entropy(pred_node_prob, true_node_prob)

    kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu**2 - (logvar.exp() + epsilon)**2, dim=1), dim=0)

    return lambda_e*edge_loss + lambda_a*node_loss + lambda_kl*kl_loss



def train(model, train_loader, loss_function, optimizer, device='cpu'):
    
    model.train()
    loss_all = 0

    for data in train_loader:

        data = data.to(device)
        optimizer.zero_grad()
        y_pred = model(data)
        loss = loss_function(y_pred, data.y)
        loss.backward()
        loss_all += loss.item() 
        optimizer.step()

    return loss_all / len(train_loader.dataset)


def eval(model, val_loader, loss_function, device='cpu'):
    
    model.eval()
    error = 0

    for data in val_loader:
        data = data.to(device)
        
        with torch.no_grad():
            y_pred = model(data)
            # mean absolute error
            error += loss_function(y_pred, data.y).item()
    
    return error / len(val_loader.dataset)


def run_experiment(model, model_name, n_epochs, loss_function, optimizer, scheduler, train_loader, val_loader, test_loader):
    
    print(f"Running experiment for {model_name}, training on {len(train_loader.dataset)} samples for {n_epochs} epochs.")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("\nModel architecture:")
    print(model)
    print(f'Total parameters: {get_params(model)}')
    model = model.to(device)
    
    print("\nStart training:")
    best_val_error = None
    perf_per_epoch = [] # Track Test/Val MAE vs. epoch (for plotting)
    

    t = time.time()

    for epoch in tqdm(range(1, n_epochs+1)):
        # Call LR scheduler at start of each epoch
        lr = scheduler.optimizer.param_groups[0]['lr']

        # Train model for one epoch, return avg. training loss
        loss = train(model, train_loader, loss_function, optimizer, device)
        
        # Evaluate model on validation set
        val_error = eval(model, val_loader, loss_function, device)

        scheduler.step(val_error)
        
        
        if best_val_error is None or val_error <= best_val_error:
            # Evaluate model on test set if validation metric improves
            test_error = eval(model, test_loader, loss_function, device)
            best_val_error = val_error

        if epoch % 10 == 0:
            # Print and track stats every 10 epochs
            print(f'Epoch: {epoch:03d}, LR: {lr:5f}, Loss: {loss:.7f}, '
                  f'Val Loss: {val_error:.7f}, Test Loss: {test_error:.7f}')
        
        perf_per_epoch.append((loss, test_error, val_error, epoch, model_name))
    
    t = time.time() - t
    train_time = t/60
    print(f"\nDone! Training took {train_time:.2f} mins. Best validation Loss: {best_val_error:.7f}, corresponding test Loss: {test_error:.7f}.")
    
    return best_val_error, test_error, train_time, perf_per_epoch

    