import h5py
import numpy as np
import torch
from torch_geometric.utils import scatter
import os
from tqdm import tqdm

# --- CONFIG ---
IN_FILE = "/home3/ivkhar/Baikal/data/normed/baikal_mc2020_multi_split_0924mid_eq_norm.h5"
OUT_FILE = "/home/levos/experiments/data_k4_full_18M.h5"
STATS_PATH = "/home/levos/experiments/tools/feat_stats_21.pt"
K = 4
DEVICE = torch.device('cuda:0')

def preprocess():
    if os.path.exists(OUT_FILE):
        print(f"Removing existing file {OUT_FILE}...")
        os.remove(OUT_FILE)

    stats = torch.load(STATS_PATH, weights_only=True)
    means, stds = stats['means'].to(DEVICE), stats['stds'].to(DEVICE)

    with h5py.File(IN_FILE, 'r') as fin:
        total_events = len(fin['train/ev_starts/data']) - 1
        print(f"Starting TOTAL precalculation for {total_events} events...")
        starts = fin['train/ev_starts/data']
        
        with h5py.File(OUT_FILE, 'w') as fout:
            # Create datasets with chunking enabled for better performance
            dset_x = fout.create_dataset('x', (0, 21), maxshape=(None, 21), dtype='float32', chunks=(10000, 21))
            dset_y = fout.create_dataset('y', (0,), maxshape=(None,), dtype='uint8', chunks=(10000,))
            dset_edges = fout.create_dataset('edges', (2, 0), maxshape=(2, None), dtype='int32', chunks=(2, 50000))
            dset_ev_starts = fout.create_dataset('ev_starts', (total_events + 1,), dtype='int64')
            dset_edge_starts = fout.create_dataset('edge_starts', (total_events + 1,), dtype='int64')

            curr_node_global = 0
            curr_edge_global = 0
            dset_ev_starts[0] = 0
            dset_edge_starts[0] = 0

            batch_size_events = 5000 # Process 5k events per chunk
            for i in tqdm(range(0, total_events, batch_size_events)):
                end_idx = min(i + batch_size_events, total_events)
                ev_range = starts[i : end_idx + 1]
                
                raw_x = torch.from_numpy(fin['train/data/data'][ev_range[0] : ev_range[-1]]).float().to(DEVICE)
                raw_y = torch.from_numpy(fin['train/labels/data'][ev_range[0] : ev_range[-1]]).to(DEVICE)
                rel_starts = ev_range - ev_range[0]
                
                x_list, y_list, edge_list = [], [], []
                
                for j in range(len(rel_starts) - 1):
                    s, e = rel_starts[j], rel_starts[j+1]
                    x_ev = raw_x[s:e]; y_ev = (raw_y[s:e] != 0).to(torch.uint8)
                    n = x_ev.size(0)
                    
                    if n <= 1:
                        edge_index = torch.zeros((2, 0), dtype=torch.long, device=DEVICE)
                    else:
                        idx = torch.arange(n, device=DEVICE)
                        mask = (torch.abs(idx.view(-1, 1) - idx.view(1, -1)) <= K) & (idx.view(-1, 1) != idx.view(1, -1))
                        edge_index = torch.stack(torch.where(mask))
                    
                    # 21 Features
                    t0, x0, y0, z0 = x_ev[0, 1], x_ev[0, 2], x_ev[0, 3], x_ev[0, 4]
                    dt, dx, dy, dz = x_ev[:,1]-t0, x_ev[:,2]-x0, x_ev[:,3]-y0, x_ev[:,4]-z0
                    dr2 = dx**2 + dy**2 + dz**2; dr = torch.sqrt(dr2 + 1e-8)
                    s2, tof = (0.225 * dt)**2 - dr2, dt - dr/0.225
                    r_xy, phi = torch.sqrt(x_ev[:, 2]**2 + x_ev[:, 3]**2 + 1e-8), torch.atan2(x_ev[:, 3], x_ev[:, 2])
                    rho = torch.sqrt(x_ev[:, 2]**2 + x_ev[:, 3]**2 + x_ev[:, 4]**2 + 1e-8)
                    cosTheta = x_ev[:, 4] / (rho + 1e-8)
                    row, col = edge_index
                    dist_edges = torch.sqrt(torch.sum((x_ev[row, 2:5] - x_ev[col, 2:5])**2, dim=1) + 1e-8)
                    mean_dist_neigh = scatter(dist_edges, row, dim=0, dim_size=n, reduce='mean')
                    neigh_charge = scatter(x_ev[col, 0], row, dim=0, dim_size=n, reduce='sum')
                    q_rel_mean = x_ev[:, 0] / (x_ev[:, 0].mean() + 1e-8)
                    cos_alpha = dz / (dr + 1e-8)
                    x_bin, y_bin = torch.round(x_ev[:, 2] / 0.03), torch.round(x_ev[:, 3] / 0.03)
                    xy_bins = torch.stack([x_bin, y_bin], dim=1)
                    _, str_ids = torch.unique(xy_bins, dim=0, return_inverse=True)
                    hits_on_string = scatter(torch.ones_like(x_ev[:, 0]), str_ids, dim=0, reduce='sum')[str_ids]
                    max_z, min_z = scatter(x_ev[:, 4], str_ids, dim=0, reduce='max')[str_ids], scatter(x_ev[:, 4], str_ids, dim=0, reduce='min')[str_ids]
                    z_span = max_z - min_z
                    extra = torch.stack([s2, dt, dr, r_xy, phi, rho, cosTheta, tof, mean_dist_neigh, neigh_charge, q_rel_mean, cos_alpha, hits_on_string, z_span, torch.full((n,), n, device=DEVICE), torch.full((n,), x_ev[:, 1].max() - x_ev[:, 1].min(), device=DEVICE)], dim=1)
                    
                    x_final = torch.cat([x_ev, (extra - means) / (stds + 1e-8)], dim=1)
                    
                    edge_list.append((edge_index + curr_node_global).cpu().numpy().astype('int32'))
                    x_list.append(x_final.cpu().numpy())
                    y_list.append(y_ev.cpu().numpy())
                    
                    curr_node_global += n
                    curr_edge_global += edge_index.shape[1]
                    dset_ev_starts[i + j + 1] = curr_node_global
                    dset_edge_starts[i + j + 1] = curr_edge_global

                # Efficient batch write
                x_c = np.concatenate(x_list, axis=0); y_c = np.concatenate(y_list, axis=0); e_c = np.concatenate(edge_list, axis=1)
                dset_x.resize((dset_x.shape[0] + x_c.shape[0], 21)); dset_x[-x_c.shape[0]:] = x_c
                dset_y.resize((dset_y.shape[0] + y_c.shape[0],)); dset_y[-y_c.shape[0]:] = y_c
                dset_edges.resize((2, dset_edges.shape[1] + e_c.shape[1])); dset_edges[:, -e_c.shape[1]:] = e_c

    print(f"Preprocessing finished! Massive file saved at: {OUT_FILE}")

if __name__ == "__main__":
    preprocess()
