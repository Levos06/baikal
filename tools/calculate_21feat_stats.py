import torch
import os
import glob
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.utils import scatter
from tqdm import tqdm

DATA_DIR = "data_processed_50k/val"
C_WATER = 0.225

def add_features_raw(batch):
    x, edge_index, ptr, b_idx = batch.x, batch.edge_index, batch.ptr, batch.batch
    num_nodes = x.size(0)
    
    first = ptr[:-1]; sizes = ptr[1:] - ptr[:-1]
    t0, x0, y0, z0 = [torch.repeat_interleave(x[first, i], sizes) for i in [1,2,3,4]]
    dt, dx, dy, dz = x[:,1]-t0, x[:,2]-x0, x[:,3]-y0, x[:,4]-z0
    dr2 = dx**2 + dy**2 + dz**2; dr = torch.sqrt(dr2 + 1e-8)
    
    s2 = (C_WATER * dt)**2 - dr2
    tof = dt - dr/C_WATER
    r_xy = torch.sqrt(x[:, 2]**2 + x[:, 3]**2 + 1e-8)
    phi = torch.atan2(x[:, 3], x[:, 2])
    rho = torch.sqrt(x[:, 2]**2 + x[:, 3]**2 + x[:, 4]**2 + 1e-8)
    cosTheta = x[:, 4] / (rho + 1e-8)
    
    row, col = edge_index
    dist_edges = torch.sqrt(torch.sum((x[row, 2:5] - x[col, 2:5])**2, dim=1) + 1e-8)
    mean_dist_neigh = scatter(dist_edges, row, dim=0, dim_size=num_nodes, reduce='mean')
    neigh_charge = scatter(x[col, 0], row, dim=0, dim_size=num_nodes, reduce='sum')
    
    event_mean_q = scatter(x[:, 0], b_idx, dim=0, reduce='mean')
    q_rel_mean = x[:, 0] / (torch.gather(event_mean_q, 0, b_idx) + 1e-8)
    cos_alpha = dz / (dr + 1e-8)
    
    # Corrected String Logic
    x_bin, y_bin = torch.round(x[:, 2] / 0.03), torch.round(x[:, 3] / 0.03)
    xy_bins = torch.stack([b_idx, x_bin, y_bin], dim=1)
    _, str_ids = torch.unique(xy_bins, dim=0, return_inverse=True)
    
    hits_on_string = scatter(torch.ones_like(x[:, 0]), str_ids, dim=0, reduce='sum')[str_ids]
    max_z = scatter(x[:, 4], str_ids, dim=0, reduce='max')[str_ids]
    min_z = scatter(x[:, 4], str_ids, dim=0, reduce='min')[str_ids]
    z_span = max_z - min_z
    
    event_n_hits = sizes.float()
    n_hits = torch.gather(event_n_hits, 0, b_idx)
    event_t_min = scatter(x[:, 1], b_idx, dim=0, reduce='min')
    event_t_max = scatter(x[:, 1], b_idx, dim=0, reduce='max')
    duration = torch.gather(event_t_max - event_t_min, 0, b_idx)

    return torch.stack([
        s2, dt, dr, r_xy, phi, rho, cosTheta, tof,
        mean_dist_neigh, neigh_charge, q_rel_mean, cos_alpha,
        hits_on_string, z_span, n_hits, duration
    ], dim=1)

def run_stats():
    device = torch.device('cuda')
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.pt")))
    all_features = []
    
    print("Collecting statistics for 21 features...")
    for f in tqdm(files):
        data_list = torch.load(f, weights_only=False)
        loader = DataLoader(data_list, batch_size=512)
        for batch in loader:
            f_batch = add_features_raw(batch.to(device))
            all_features.append(f_batch.cpu())
            
    all_f = torch.cat(all_features, dim=0)
    means = torch.mean(all_f, dim=0)
    stds = torch.std(all_f, dim=0)
    
    print("\n--- CALCULATED STATS (Extra 5 to 20) ---")
    for i in range(means.size(0)):
        print(f"F{i+5}: mean={means[i]:.6f}, std={stds[i]:.6f}")
    
    # Save for training script
    torch.save({'means': means, 'stds': stds}, 'tools/feat_stats_21.pt')

if __name__ == "__main__":
    run_stats()
