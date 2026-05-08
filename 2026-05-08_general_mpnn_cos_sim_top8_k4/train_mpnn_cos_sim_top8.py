import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter
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
PROJECT_DIR = "/home/levos/experiments/2026-05-08_general_mpnn_cos_sim_top8_k4"
BATCH_SIZE = 32 # Lower batch size because FC layer and dynamic edges are memory intensive
NUM_WORKERS = 4
TOTAL_EPOCHS = 1000
VIRTUAL_EPOCH_SIZE = 200
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load stats
STATS = torch.load('/home/levos/experiments/tools/feat_stats_21.pt', weights_only=True)
MEANS, STDS = STATS['means'].to(DEVICE), STATS['stds'].to(DEVICE)

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
    
    # Graphs for local features use original edge_index
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
    # Returns edge_index where nodes in same batch are fully connected
    device = batch_vector.device
    counts = torch.bincount(batch_vector)
    
    edge_indices = []
    offset = 0
    for count in counts:
        if count > 0:
            nodes = torch.arange(offset, offset + count, device=device)
            row = nodes.repeat_interleave(count)
            col = nodes.repeat(count)
            edge_indices.append(torch.stack([row, col], dim=0))
            offset += count
            
    if not edge_indices:
        return torch.empty((2, 0), dtype=torch.long, device=device)
        
    return torch.cat(edge_indices, dim=1)

def get_topk_similarity_edges(h, batch_vector, k=8):
    # Calculate cosine similarity matrix per graph and return top-k edges
    device = h.device
    counts = torch.bincount(batch_vector)
    
    edge_indices = []
    offset = 0
    for count in counts:
        if count > 0:
            h_graph = h[offset:offset+count]
            h_norm = F.normalize(h_graph, p=2, dim=1)
            sim = torch.mm(h_norm, h_norm.t())
            
            # Prevent self-loops from being selected as top-k
            sim.fill_diagonal_(-float('inf'))
            
            # If the graph has fewer than k+1 nodes, we just take all possible neighbors
            actual_k = min(k, count - 1)
            if actual_k > 0:
                # Get top-k indices for each node
                _, topk_indices = torch.topk(sim, actual_k, dim=1)
                
                # Create row and col tensors
                row = torch.arange(count, device=device).view(-1, 1).repeat(1, actual_k).view(-1)
                col = topk_indices.view(-1)
                
                edge_indices.append(torch.stack([row + offset, col + offset], dim=0))
            
            offset += count
            
    if not edge_indices:
        return torch.empty((2, 0), dtype=torch.long, device=device)
        
    return torch.cat(edge_indices, dim=1)

class GeneralMPNNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='mean')
        
        # Function f: message generation
        # Input: concatenated features of sender and receiver nodes [x_i, x_j]
        hidden = max(in_channels, out_channels) * 2
        self.f = nn.Sequential(
            nn.Linear(2 * in_channels, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, out_channels),
            nn.LayerNorm(out_channels),
            nn.GELU()
        )
        
        # Function u: node update
        # Input: concatenated current node state and aggregated message [x, aggr_out]
        self.u = nn.Sequential(
            nn.Linear(in_channels + out_channels, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, out_channels),
            nn.LayerNorm(out_channels)
            # Output of update will go through GELU in the main forward pass (skip connection addition)
        )

    def forward(self, x, edge_index):
        # Start propagation
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        # Concatenate receiver and sender features
        tmp = torch.cat([x_i, x_j], dim=1)
        return self.f(tmp)

    def update(self, aggr_out, x):
        # Concatenate current node features and aggregated messages
        tmp = torch.cat([x, aggr_out], dim=1)
        return self.u(tmp)

class MPNN_CosSimTopK_FCLast(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.projs = torch.nn.ModuleList()
        
        # Layers 1-3: MPNN (Local Connectivity)
        self.convs.append(GeneralMPNNConv(in_channels, 256))
        self.projs.append(torch.nn.Linear(in_channels, 256))
        
        for _ in range(2):
            self.convs.append(GeneralMPNNConv(256, 256))
            self.projs.append(torch.nn.Identity())
            
        # Layer 4: MPNN with Complete Graph (Global Connectivity)
        self.fc_conv = GeneralMPNNConv(256, 256)
        
        self.head = torch.nn.Sequential(
            torch.nn.Linear(256, 512), torch.nn.GELU(),
            torch.nn.Linear(512, 256), torch.nn.GELU(),
            torch.nn.Linear(256, out_channels)
        )
        
    def forward(self, x, edge_index, batch_vector):
        h = x
        
        # Layer 1 (i=0) uses base graph only
        h = F.gelu(self.convs[0](h, edge_index) + self.projs[0](h))
        
        # Layers 2 and 3 (i=1, 2) use dynamic cos sim top-k graph + base graph
        for i in range(1, 3):
            sim_edges = get_topk_similarity_edges(h, batch_vector, k=8)
            # Combine base edges and new similarity edges
            curr_edge_index = torch.cat([edge_index, sim_edges], dim=1)
            # Remove duplicate edges to keep computation efficient
            curr_edge_index = torch.unique(curr_edge_index, dim=1)
            
            h = F.gelu(self.convs[i](h, curr_edge_index) + self.projs[i](h))
        
        # Last layer (Global / Fully Connected)
        fc_edge_index = get_complete_edge_index(batch_vector)
        h = F.gelu(self.fc_conv(h, fc_edge_index) + h) # Residual skip
        
        return self.head(h)

class MediumDataset(torch.utils.data.IterableDataset):
    def __init__(self, split='train', k=4):
        super().__init__()
        self.data_dir = os.path.join(DATA_DIR, split)
        self.k = k
    def __iter__(self):
        try:
            while True:
                files = sorted(glob.glob(os.path.join(self.data_dir, "*.pt")))
                random.shuffle(files)
                for f in files:
                    try:
                        data_list = torch.load(f, weights_only=False)
                        random.shuffle(data_list)
                        for data in data_list:
                            num_nodes = data.x.size(0)
                            if num_nodes <= 1:
                                data.edge_index = torch.zeros((2, 0), dtype=torch.long)
                            else:
                                indices = np.arange(num_nodes)
                                mask = (np.abs(indices[:, None] - indices) <= self.k) & (indices[:, None] != indices)
                                data.edge_index = torch.from_numpy(np.array(np.where(mask))).long()
                            yield data
                    except GeneratorExit: return
                    except: continue
        except GeneratorExit: return

def calculate_metrics(labels, probs):
    p, r, _ = precision_recall_curve(labels, probs)
    p_at_r095 = np.interp(0.95, r[::-1], p[::-1])
    preds = (probs > 0.5).astype(int)
    prec = precision_score(labels, preds, zero_division=0); rec = recall_score(labels, preds, zero_division=0)
    return prec, rec, p_at_r095

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
    print(f"Starting General MPNN + CosSim (Top-8) + k=4 Experiment on {DEVICE}")
    model = MPNN_CosSimTopK_FCLast(21, 2).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    train_loader = DataLoader(MediumDataset('train', k=4), batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(MediumDataset('val', k=4), batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)
    
    os.makedirs(f"{PROJECT_DIR}/checkpoints", exist_ok=True)
    
    log_file = os.path.join(PROJECT_DIR, "train_mpnn_cossim_top8_k4.log")
    with open(log_file, "w") as f:
        f.write("Epoch | Time_Start | Duration | LR | T_Loss | T_Prec | T_Rec | T_P@R0.95 | V_Loss | V_Prec | V_Rec | V_P@R0.95\n")
    
    epoch = 1
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
            t_loss, t_prec, t_rec, t_p95 = np.mean(t_losses), *calculate_metrics(np.array(t_labels), np.array(t_probs))
            v_loss, v_prec, v_rec, v_p95 = evaluate(model, val_loader)
            log_str = f"{epoch:04d} | {start_clock} | {duration:5.1f}s | 1.0e-04 | {t_loss:.4f} | {t_prec:.4f} | {t_rec:.4f} | {t_p95:.4f} | {v_loss:.4f} | {v_prec:.4f} | {v_rec:.4f} | {v_p95:.4f}"
            print(log_str)
            with open(log_file, "a") as f: f.write(log_str + "\n")
            if epoch % 100 == 0: torch.save(model.state_dict(), f"{PROJECT_DIR}/checkpoints/model_mpnn_cossim_top8_k4_{epoch}.pt")
            epoch += 1
            if epoch > TOTAL_EPOCHS: break
            t_labels, t_probs, t_losses = [], [], []; start_time = time.time(); model.train()

if __name__ == "__main__":
    train()