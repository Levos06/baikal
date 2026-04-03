import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.loader import DataLoader
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.utils import scatter
import re

# --- CONFIG ---
DATA_DIR = "/home/levos/experiments/data_processed_50k"
PROJECT_DIR = "/home/levos/experiments/2026-04-01_gat_baselines_res_study"
STATS_PATH = "/home/levos/experiments/tools/feat_stats_21.pt"
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
NUM_EVENTS_TARGET = 1000

# Load stats
STATS = torch.load(STATS_PATH, weights_only=True)
MEANS, STDS = STATS['means'].to(DEVICE), STATS['stds'].to(DEVICE)

def add_features_21_norm(batch):
    x, edge_index, ptr, b_idx = batch.x, batch.edge_index, batch.ptr, batch.batch
    sizes = ptr[1:] - ptr[:-1]
    t0, x0, y0, z0 = [torch.repeat_interleave(x[ptr[:-1], i], sizes) for i in [1,2,3,4]]
    dt, dx, dy, dz = x[:,1]-t0, x[:,2]-x0, x[:,3]-y0, x[:,4]-z0
    dr2 = dx**2 + dy**2 + dz**2; dr = torch.sqrt(dr2 + 1e-8)
    s2, tof = (0.225 * dt)**2 - dr2, dt - dr/0.225
    r_xy, phi = torch.sqrt(x[:, 2]**2 + x[:, 3]**2 + 1e-8), torch.atan2(x[:, 3], x[:, 2])
    rho = torch.sqrt(x[:, 2]**2 + x[:, 3]**2 + x[:, 4]**2 + 1e-8)
    cosTheta = x[:, 4] / (rho + 1e-8)
    row, col = edge_index
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

class DiagnosticGAT(torch.nn.Module):
    def __init__(self, res=True):
        super().__init__()
        self.res = res
        self.convs = torch.nn.ModuleList([GATv2Conv(21 if i==0 else 256, 64, heads=4) for i in range(4)])
        self.projs = torch.nn.ModuleList([torch.nn.Linear(21 if i==0 else 256, 256) for i in range(4)])
    def forward(self, x, edge_index):
        states = [x]
        h = x
        for i in range(4):
            h_new = self.convs[i](h, edge_index)
            if self.res: h = F.gelu(h_new + self.projs[i](h))
            else: h = F.gelu(h_new)
            states.append(h)
        return states

def calculate_diversity(batch_states, batch_idx):
    batch_event_dists = [[] for _ in range(len(batch_states))]
    event_ids = torch.unique(batch_idx)
    for eid in event_ids:
        mask = (batch_idx == eid)
        for layer_idx, h in enumerate(batch_states):
            h_event = F.normalize(h[mask], p=2, dim=1)
            if h_event.size(0) < 2: continue
            sim = torch.mm(h_event, h_event.t())
            dist = 1.0 - sim
            n = dist.size(0)
            avg_dist = (dist.sum() - dist.diag().sum()) / (n * (n - 1))
            batch_event_dists[layer_idx].append(avg_dist.item())
    return batch_event_dists

class MediumDataset(torch.utils.data.IterableDataset):
    def __init__(self, split='val'):
        super().__init__()
        self.data_dir = os.path.join(DATA_DIR, split)
    def __iter__(self):
        files = sorted(glob.glob(os.path.join(self.data_dir, "*.pt")))
        for f in files:
            data_list = torch.load(f, weights_only=False)
            for data in data_list: yield data

def get_checkpoint_stats(path, res_mode):
    model = DiagnosticGAT(res=res_mode).to(DEVICE)
    model.load_state_dict(torch.load(path, weights_only=True), strict=False)
    model.eval()
    
    loader = DataLoader(MediumDataset(), batch_size=64)
    all_dists = [[] for _ in range(5)]
    events_count = 0
    
    with torch.no_grad():
        for batch in loader:
            batch = add_features_21_norm(batch.to(DEVICE))
            states = model(batch.x, batch.edge_index)
            batch_dists = calculate_diversity(states, batch.batch)
            for i in range(5): all_dists[i].extend(batch_dists[i])
            events_count += len(torch.unique(batch.batch))
            if events_count >= NUM_EVENTS_TARGET: break
            
    return [np.mean(d) for d in all_dists]

def run_evolution_test():
    # 1. Collect checkpoints
    res_files = sorted(glob.glob(f"{PROJECT_DIR}/checkpoints_res/model_res_*.pt"), key=lambda x: int(re.search(r'_(\d+).pt', x).group(1)))
    nores_files = sorted(glob.glob(f"{PROJECT_DIR}/checkpoints_nores/model_nores_*.pt"), key=lambda x: int(re.search(r'_(\d+).pt', x).group(1)))
    
    results_res = {}
    results_nores = {}
    
    print("Testing Residual checkpoints...")
    for f in res_files:
        epoch = int(re.search(r'_(\d+).pt', f).group(1))
        print(f"  Epoch {epoch}...")
        results_res[epoch] = get_checkpoint_stats(f, True)
        
    print("Testing No-Residual checkpoints...")
    for f in nores_files:
        epoch = int(re.search(r'_(\d+).pt', f).group(1))
        print(f"  Epoch {epoch}...")
        results_nores[epoch] = get_checkpoint_stats(f, False)
        
    # 2. Plotting
    os.makedirs(f"{PROJECT_DIR}/plots", exist_ok=True)
    layers = range(5)
    
    # --- Combined Plot ---
    plt.figure(figsize=(12, 8))
    for epoch, dists in results_res.items():
        alpha = 0.2 + 0.8 * (epoch / max(results_res.keys()))
        plt.plot(layers, dists, color='blue', alpha=alpha, label=f'Res Ep {epoch}' if epoch % 200 == 0 else "")
    for epoch, dists in results_nores.items():
        alpha = 0.2 + 0.8 * (epoch / max(results_nores.keys()))
        plt.plot(layers, dists, color='red', alpha=alpha, label=f'NoRes Ep {epoch}' if epoch % 200 == 0 else "")
    plt.title('Evolution of Oversmoothing (1000 events, Res vs No-Res)')
    plt.xlabel('Layer')
    plt.ylabel('Mean Pairwise Cosine Distance')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(ncol=2, fontsize='small')
    plt.savefig(f"{PROJECT_DIR}/plots/oversmoothing_evolution_combined.png")
    
    # --- Res Only ---
    plt.figure(figsize=(10, 6))
    for epoch, dists in results_res.items():
        color = plt.cm.Blues(0.3 + 0.7 * (epoch / max(results_res.keys())))
        plt.plot(layers, dists, color=color, label=f'Epoch {epoch}')
    plt.title('Evolution of Diversity: GAT with Residual')
    plt.xlabel('Layer')
    plt.ylabel('Mean Pairwise Cosine Distance')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='x-small')
    plt.tight_layout()
    plt.savefig(f"{PROJECT_DIR}/plots/oversmoothing_evolution_res.png")
    
    # --- No-Res Only ---
    plt.figure(figsize=(10, 6))
    for epoch, dists in results_nores.items():
        color = plt.cm.Reds(0.3 + 0.7 * (epoch / max(results_nores.keys())))
        plt.plot(layers, dists, color=color, label=f'Epoch {epoch}')
    plt.title('Evolution of Diversity: GAT No-Residual')
    plt.xlabel('Layer')
    plt.ylabel('Mean Pairwise Cosine Distance')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='x-small')
    plt.tight_layout()
    plt.savefig(f"{PROJECT_DIR}/plots/oversmoothing_evolution_nores.png")
    
    print("\nAll plots saved to plots/ directory.")

if __name__ == "__main__":
    run_evolution_test()
