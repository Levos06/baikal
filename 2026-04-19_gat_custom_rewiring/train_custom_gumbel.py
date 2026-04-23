import torch
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import scatter, to_dense_batch, softmax
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.loader import DataLoader
import os
import time
import glob
import random
import numpy as np
import sys
from sklearn.metrics import precision_score, recall_score, precision_recall_curve

# --- CONFIG ---
DATA_DIR = "/home/levos/experiments/data_processed_50k"
PROJECT_DIR = "/home/levos/experiments/2026-04-19_gat_custom_rewiring"
BATCH_SIZE = 128
NUM_WORKERS = 4
TOTAL_EPOCHS = 1000
VIRTUAL_EPOCH_SIZE = 200
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
ALPHA = 0.1

# Load stats
STATS = torch.load('/home/levos/experiments/tools/feat_stats_21.pt', weights_only=True)
MEANS, STDS = STATS['means'].to(DEVICE), STATS['stds'].to(DEVICE)

# --- CUSTOM GATv2Conv with e_ij = a^T LeakyReLU(W[h_i || h_j]) + W_ij ---
class CustomGATv2Conv(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=1,
                 concat=True, negative_slope=0.2, dropout=0.0,
                 add_self_loops=True, bias=True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops

        self.lin_l = Linear(in_channels, heads * out_channels, bias=bias, weight_initializer='glorot')
        self.lin_r = Linear(in_channels, heads * out_channels, bias=bias, weight_initializer='glorot')

        self.att = torch.nn.Parameter(torch.Tensor(1, heads, out_channels))

        if bias and concat:
            self.bias = torch.nn.Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_weight=None, return_attention_weights=False):
        H, C = self.heads, self.out_channels

        x_l = self.lin_l(x).view(-1, H, C)
        x_r = self.lin_r(x).view(-1, H, C)

        out = self.propagate(edge_index, x=(x_l, x_r), edge_weight=edge_weight)

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j, x_i, edge_weight, index, ptr, size_i):
        x = x_i + x_j
        x = F.leaky_relu(x, self.negative_slope)
        alpha = (x * self.att).sum(dim=-1)
        
        if edge_weight is not None:
            alpha = alpha + edge_weight

        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return x_j * alpha.unsqueeze(-1)


# --- HELPER FUNCTIONS ---
def add_features_21_norm(batch):
    x, ptr, b_idx = batch.x, batch.ptr, batch.batch
    sizes = ptr[1:] - ptr[:-1]
    t0, x0, y0, z0 = [torch.repeat_interleave(x[ptr[:-1], i], sizes) for i in [1,2,3,4]]
    dt, dx, dy, dz = x[:,1]-t0, x[:,2]-x0, x[:,3]-y0, x[:,4]-z0
    dr2 = dx**2 + dy**2 + dz**2; dr = torch.sqrt(dr2 + 1e-8)
    s2, tof = (0.225 * dt)**2 - dr2, dt - dr/0.225
    r_xy, phi = torch.sqrt(x[:, 2]**2 + x[:, 3]**2 + 1e-8), torch.atan2(x[:, 3], x[:, 2])
    rho = torch.sqrt(x[:, 2]**2 + x[:, 3]**2 + x[:, 4]**2 + 1e-8)
    cosTheta = x[:, 4] / (rho + 1e-8)
    
    row, col = batch.edge_index
    dist_edges = torch.sqrt(torch.sum((x[row, 2:5] - x[col, 2:5])**2, dim=1) + 1e-8)
    mean_dist_neigh = scatter(dist_edges, row, dim=0, dim_size=x.size(0), reduce='mean')
    neigh_charge = scatter(x[col, 0], row, dim=0, dim_size=x.size(0), reduce='sum')
    event_mean_q = scatter(x[:, 0], b_idx, dim=0, reduce='mean')
    q_rel_mean = x[:, 0] / (torch.gather(event_mean_q, 0, b_idx) + 1e-8)
    cos_alpha = dz / (dr + 1e-8)
    
    x_bin, y_bin = torch.round(x[:, 2] / 0.03), torch.round(x[:, 3] / 0.03)
    xy_bins = torch.stack([b_idx, x_bin, y_bin], dim=1)
    _, str_ids = torch.unique(xy_bins, dim=0, return_inverse=True)
    hits_on_string = scatter(torch.ones_like(x[:, 0]), str_ids, dim=0, reduce='sum')[str_ids]
    max_z, min_z = scatter(x[:, 4], str_ids, dim=0, reduce='max')[str_ids], scatter(x[:, 4], str_ids, dim=0, reduce='min')[str_ids]
    z_span = max_z - min_z
    
    event_n_hits = sizes.float(); n_hits = torch.gather(event_n_hits, 0, b_idx)
    duration = torch.gather(scatter(x[:, 1], b_idx, dim=0, reduce='max') - scatter(x[:, 1], b_idx, dim=0, reduce='min'), 0, b_idx)
    
    raw_extra = torch.stack([s2, dt, dr, r_xy, phi, rho, cosTheta, tof, mean_dist_neigh, neigh_charge, q_rel_mean, cos_alpha, hits_on_string, z_span, n_hits, duration], dim=1)
    batch.x = torch.cat([x, (raw_extra - MEANS) / (STDS + 1e-8)], dim=1)
    return batch

def get_complete_edge_index(batch_vector):
    device = batch_vector.device
    nodes_range = torch.arange(batch_vector.size(0), device=device)
    row = nodes_range.repeat_interleave(nodes_range.size(0))
    col = nodes_range.repeat(nodes_range.size(0))
    mask = batch_vector[row] == batch_vector[col]
    return torch.stack([row[mask], col[mask]], dim=0)

def rewire_edges_custom_gumbel(edge_index, batch_vector, Q, K, Q_dense, K_dense, mask, current_epoch, max_epochs, alpha=0.1):
    B, N_max, _ = Q_dense.shape
    device = edge_index.device
    N = batch_vector.size(0)
    
    S = torch.bmm(Q_dense, K_dense.transpose(1, 2))
    
    if Q.requires_grad:
        U = torch.rand_like(S)
        gumbel_noise = -torch.log(-torch.log(U + 1e-8) + 1e-8)
        S_noisy = S + gumbel_noise
    else:
        S_noisy = S
    
    valid_mask = mask.unsqueeze(2) & mask.unsqueeze(1)
    diag_mask = torch.eye(N_max, device=device).unsqueeze(0).bool()
    
    S_add = S_noisy.clone()
    S_add[~valid_mask | diag_mask] = -float('inf')
    
    S_remove = S_noisy.clone()
    S_remove[~valid_mask | diag_mask] = float('inf')
    
    flat_indices = torch.zeros((B, N_max), dtype=torch.long, device=device)
    flat_indices[mask] = torch.arange(mask.sum(), device=device)
    
    edges_to_add = []
    edges_to_remove = []
    
    edges_batch_idx = batch_vector[edge_index[0]]
    edge_counts = scatter(torch.ones_like(edges_batch_idx), edges_batch_idx, dim=0, dim_size=B, reduce='sum')
    k_vals = (edge_counts * alpha).long()
    
    for b in range(B):
        k = k_vals[b].item()
        if k <= 0: continue
            
        S_add_b = S_add[b].view(-1)
        top_k_vals, top_k_idx = torch.topk(S_add_b, k)
        valid_top = top_k_vals > -float('inf')
        top_k_idx = top_k_idx[valid_top]
        
        row_add = flat_indices[b, top_k_idx // N_max]
        col_add = flat_indices[b, top_k_idx % N_max]
        edges_to_add.append(torch.stack([row_add, col_add], dim=0))
        
        S_rem_b = S_remove[b].view(-1)
        bot_k_vals, bot_k_idx = torch.topk(S_rem_b, k, largest=False)
        valid_bot = bot_k_vals < float('inf')
        bot_k_idx = bot_k_idx[valid_bot]
        
        row_rem = flat_indices[b, bot_k_idx // N_max]
        col_rem = flat_indices[b, bot_k_idx % N_max]
        edges_to_remove.append(torch.stack([row_rem, col_rem], dim=0))
        
    if len(edges_to_add) > 0:
        edges_to_add = torch.cat(edges_to_add, dim=1)
    else:
        edges_to_add = torch.empty((2, 0), dtype=torch.long, device=device)
        
    if len(edges_to_remove) > 0:
        edges_to_remove = torch.cat(edges_to_remove, dim=1)
    else:
        edges_to_remove = torch.empty((2, 0), dtype=torch.long, device=device)
        
    if edges_to_remove.size(1) > 0:
        hash_mult = N
        current_hashes = edge_index[0] * hash_mult + edge_index[1]
        remove_hashes = edges_to_remove[0] * hash_mult + edges_to_remove[1]
        keep_mask = ~torch.isin(current_hashes, remove_hashes)
        edge_index = edge_index[:, keep_mask]
        
    if edges_to_add.size(1) > 0:
        edge_index = torch.cat([edge_index, edges_to_add], dim=1)
        
    edge_index = torch.unique(edge_index, dim=1)
    
    # Custom Gumbel-Softmax formulation
    tau_start = 5.0
    tau_end = 0.1
    tau = tau_start * (tau_end / tau_start) ** (current_epoch / max_epochs)
    
    row, col = edge_index
    edge_scores = (Q[row] * K[col]).sum(dim=-1) # S_ij
    
    if Q.requires_grad:
        U = torch.rand_like(edge_scores)
        gumbel_noise = -torch.log(-torch.log(U + 1e-8) + 1e-8)
        edge_scores = edge_scores + gumbel_noise
        
    # As discussed, e_ij = a^T LeakyReLU(...) + W_ij
    # Here W_ij is the temperature scaled scores. No sigmoid, as the formula adds it directly.
    # Softmax over the neighborhood is applied implicitly inside the GAT layer.
    edge_weights = (edge_scores / tau).unsqueeze(1) # [E, 1]
    
    return edge_index, edge_weights

class CustomGAT_DiffGumbel(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.current_epoch = 1
        
        self.convs = torch.nn.ModuleList()
        self.projs = torch.nn.ModuleList()
        
        # Local layers using CUSTOM GATv2Conv
        self.convs.append(CustomGATv2Conv(in_channels, 64, heads=4))
        self.projs.append(torch.nn.Linear(in_channels, 256))
        
        for _ in range(2):
            self.convs.append(CustomGATv2Conv(256, 64, heads=4))
            self.projs.append(torch.nn.Identity())
            
        self.proj_Q = torch.nn.Linear(in_channels, 64)
        self.proj_K = torch.nn.Linear(in_channels, 64)
            
        # Global Layer 
        from torch_geometric.nn import GATv2Conv as StdGATv2Conv
        self.fc_conv = StdGATv2Conv(256, 64, heads=4)
        
        self.head = torch.nn.Sequential(
            torch.nn.Linear(256, 512), torch.nn.GELU(),
            torch.nn.Linear(512, 256), torch.nn.GELU(),
            torch.nn.Linear(256, out_channels)
        )
        
    def forward(self, x, edge_index, batch_vector):
        edge_weights = None
        
        if self.current_epoch > 1:
            Q = self.proj_Q(x)
            K = self.proj_K(x)
            Q_dense, mask = to_dense_batch(Q, batch_vector)
            K_dense, _ = to_dense_batch(K, batch_vector)
            edge_index, edge_weights = rewire_edges_custom_gumbel(
                edge_index, batch_vector, Q, K, Q_dense, K_dense, mask,
                current_epoch=self.current_epoch, max_epochs=TOTAL_EPOCHS, alpha=ALPHA
            )
        else:
            edge_weights = torch.zeros((edge_index.size(1), 1), device=x.device)
            
        h = x
        for i in range(3):
            h = F.gelu(self.convs[i](h, edge_index, edge_weight=edge_weights) + self.projs[i](h))
        
        fc_edge_index = get_complete_edge_index(batch_vector)
        h = F.gelu(self.fc_conv(h, fc_edge_index) + h)
        
        return self.head(h)

class MediumDataset(torch.utils.data.IterableDataset):
    def __init__(self, split='train'):
        super().__init__()
        self.data_dir = os.path.join(DATA_DIR, split)
    def __iter__(self):
        try:
            while True:
                files = sorted(glob.glob(os.path.join(self.data_dir, "*.pt")))
                random.shuffle(files)
                for f in files:
                    try:
                        data_list = torch.load(f, weights_only=False)
                        random.shuffle(data_list)
                        for data in data_list: yield data
                    except GeneratorExit: return
                    except: continue
        except GeneratorExit: return

def calculate_metrics(labels, probs):
    p, r, _ = precision_recall_curve(labels, probs)
    p_at_r09 = np.interp(0.9, r[::-1], p[::-1])
    preds = (probs > 0.5).astype(int)
    prec = precision_score(labels, preds, zero_division=0); rec = recall_score(labels, preds, zero_division=0)
    return prec, rec, p_at_r09

def evaluate(model, loader, num_batches=50):
    model.eval()
    all_labels, all_probs, losses = [], [], []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= num_batches: break
            batch = add_features_21_norm(batch.to(DEVICE))
            out = model(batch.x, batch.edge_index, batch.batch)
            losses.append(F.cross_entropy(out, batch.y).item())
            all_probs.extend(F.softmax(out, dim=1)[:, 1].cpu().numpy()); all_labels.extend(batch.y.cpu().numpy())
    return np.mean(losses), *calculate_metrics(np.array(all_labels), np.array(all_probs))

def train():
    print(f"Starting Custom GATv2 Diff-Gumbel Rewiring Experiment on {DEVICE}")
    model = CustomGAT_DiffGumbel(21, 2).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    train_loader = DataLoader(MediumDataset('train'), batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(MediumDataset('val'), batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)
    
    log_file = os.path.join(PROJECT_DIR, "train_custom_gumbel.log")
    with open(log_file, "w") as f:
        f.write("Epoch | Time_Start | Duration | LR | T_Loss | T_Prec | T_Rec | T_P@R0.9 | V_Loss | V_Prec | V_Rec | V_P@R0.9\n")
    
    epoch = 1
    model.current_epoch = epoch
    
    start_time = time.time()
    t_labels, t_probs, t_losses = [], [], []
    
    model.train()
    for i, batch in enumerate(train_loader):
        batch = add_features_21_norm(batch.to(DEVICE, non_blocking=True))
        optimizer.zero_grad(set_to_none=True)
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = F.cross_entropy(out, batch.y); loss.backward(); optimizer.step()
        
        t_losses.append(loss.item()); t_probs.extend(F.softmax(out, dim=1)[:, 1].detach().cpu().numpy()); t_labels.extend(batch.y.cpu().numpy())
        
        if (i + 1) % VIRTUAL_EPOCH_SIZE == 0:
            duration = time.time() - start_time; start_clock = time.strftime("%H:%M:%S", time.localtime(start_time))
            t_loss, t_prec, t_rec, t_p9 = np.mean(t_losses), *calculate_metrics(np.array(t_labels), np.array(t_probs))
            v_loss, v_prec, v_rec, v_p9 = evaluate(model, val_loader)
            log_str = f"{epoch:04d} | {start_clock} | {duration:5.1f}s | 1.0e-04 | {t_loss:.4f} | {t_prec:.4f} | {t_rec:.4f} | {t_p9:.4f} | {v_loss:.4f} | {v_prec:.4f} | {v_rec:.4f} | {v_p9:.4f}"
            print(log_str)
            with open(log_file, "a") as f: f.write(log_str + "\n")
            if epoch % 100 == 0: torch.save(model.state_dict(), f"{PROJECT_DIR}/checkpoints/model_custom_gumbel_{epoch}.pt")
            epoch += 1
            model.current_epoch = epoch
            if epoch > TOTAL_EPOCHS: break
            t_labels, t_probs, t_losses = [], [], []; start_time = time.time(); model.train()

if __name__ == "__main__":
    train()
