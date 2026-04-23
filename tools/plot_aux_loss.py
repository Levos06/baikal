import os
import sys
import glob
import torch
import importlib.util
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch
import torch.nn.functional as F

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load stats
STATS = torch.load('/home/levos/experiments/tools/feat_stats_21.pt', map_location=DEVICE, weights_only=True)
MEANS, STDS = STATS['means'].to(DEVICE), STATS['stds'].to(DEVICE)

def add_features_21_norm(batch):
    from torch_geometric.utils import scatter
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
    from torch_scatter import scatter_max, scatter_min
    max_val, _ = scatter_max(x[:, 1], b_idx, dim=0)
    min_val, _ = scatter_min(x[:, 1], b_idx, dim=0)
    duration = torch.gather(max_val - min_val, 0, b_idx)
    
    raw_extra = torch.stack([s2, dt, dr, r_xy, phi, rho, cosTheta, tof, mean_dist_neigh, neigh_charge, q_rel_mean, cos_alpha, hits_on_string, z_span, n_hits, duration], dim=1)
    batch.x = torch.cat([x, (raw_extra - MEANS) / (STDS + 1e-8)], dim=1)
    return batch

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

mod1 = load_module("aux1", "/home/levos/experiments/2026-04-21_gat_auxiliary_task/train_auxiliary_task.py")
mod2 = load_module("aux2", "/home/levos/experiments/2026-04-21_gat_auxiliary_task/train_auxiliary_task_v2.py")

val_loader = DataLoader(mod1.MediumDataset('val'), batch_size=64, num_workers=4)

epochs = list(range(100, 1001, 100))
losses_v1 = []
losses_v2 = []

def eval_model(model, loader, target_type):
    model.eval()
    aux_losses = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= 10: # Just take 10 batches for quick estimation
                break
            batch = add_features_21_norm(batch.to(DEVICE))
            out, S, mask = model(batch.x, batch.edge_index, batch.batch)
            if S is not None and mask is not None:
                y_dense, _ = to_dense_batch(batch.y.float(), batch.batch)
                
                if target_type == 'v1':
                    y_target = torch.bmm(y_dense.unsqueeze(2), y_dense.unsqueeze(1))
                else:
                    y_target = (y_dense.unsqueeze(2) == y_dense.unsqueeze(1)).float()
                    
                valid_mask = mask.unsqueeze(2) & mask.unsqueeze(1)
                diag_mask = torch.eye(S.size(1), device=DEVICE).unsqueeze(0).bool()
                calc_mask = valid_mask & ~diag_mask
                
                S_flat = S[calc_mask]
                y_target_flat = y_target[calc_mask]
                loss_aux = F.binary_cross_entropy_with_logits(S_flat, y_target_flat).item()
                aux_losses.append(loss_aux)
    return np.mean(aux_losses)

for ep in epochs:
    print(f"Evaluating Epoch {ep}...")
    
    # V1
    m1 = mod1.GATv2_DynamicInput(21, 2).to(DEVICE)
    m1.load_state_dict(torch.load(f"/home/levos/experiments/2026-04-21_gat_auxiliary_task/checkpoints/model_aux_{ep}.pt", map_location=DEVICE, weights_only=True))
    m1.current_epoch = ep
    loss1 = eval_model(m1, val_loader, 'v1')
    losses_v1.append(loss1)
    
    # V2
    m2 = mod2.GATv2_DynamicInput(21, 2).to(DEVICE)
    m2.load_state_dict(torch.load(f"/home/levos/experiments/2026-04-21_gat_auxiliary_task/checkpoints/model_aux_v2_{ep}.pt", map_location=DEVICE, weights_only=True))
    m2.current_epoch = ep
    loss2 = eval_model(m2, val_loader, 'v2')
    losses_v2.append(loss2)

print("\nEpoch | Aux Loss V1 | Aux Loss V2")
print("-" * 35)
for ep, l1, l2 in zip(epochs, losses_v1, losses_v2):
    print(f"{ep:04d}  | {l1:.5f}       | {l2:.5f}")

plt.figure(figsize=(10, 6))
plt.plot(epochs, losses_v1, marker='o', label='Auxiliary Task V1 (Target = $y_i \cdot y_j$)')
plt.plot(epochs, losses_v2, marker='s', label='Auxiliary Task V2 (Target = $1$ if $y_i==y_j$ else $0$)')
plt.xlabel('Epoch')
plt.ylabel('Auxiliary BCE Loss')
plt.title('Graph Construction Error (Auxiliary Loss) Evolution')
plt.legend()
plt.grid(True)
plt.savefig('/home/levos/experiments/plots/auxiliary_loss_evolution.png')
plt.close()
print("Plot saved to /home/levos/experiments/plots/auxiliary_loss_evolution.png")
