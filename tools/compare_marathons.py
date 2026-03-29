import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader
from torch_geometric.utils import scatter
import glob
import os
import numpy as np
from sklearn.metrics import precision_recall_curve

# --- CONFIG ---
DATA_DIR = "data_processed_50k/val"
CKPT_13 = "2026-03-27_full_mlp_optimized_1500ep/checkpoints/model_final_1500.pt"
CKPT_21 = "2026-03-28_extended_21features/checkpoints/model_21feat_norm_800.pt"
STATS_21 = torch.load('tools/feat_stats_21.pt', weights_only=True)
C_WATER = 0.225

# --- MODELS ---
class BaseResGCN(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 512); self.conv2 = GCNConv(512, 512); self.conv3 = GCNConv(512, 512)
        self.conv4 = GCNConv(512, 512); self.conv5 = GCNConv(512, 512)
        self.proj1 = torch.nn.Linear(in_channels, 512); self.proj2 = torch.nn.Linear(512, 512)
        self.proj3 = torch.nn.Linear(512, 512); self.proj4 = torch.nn.Linear(512, 512)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(512 * 5, 1024), torch.nn.GELU(), torch.nn.Dropout(0.1),
            torch.nn.Linear(1024, 512), torch.nn.GELU(), torch.nn.Dropout(0.1),
            torch.nn.Linear(512, 2)
        )
    def forward(self, x, edge_index):
        h1 = F.gelu(self.conv1(x, edge_index) + self.proj1(x))
        h2 = F.gelu(self.conv2(h1, edge_index) + self.proj2(h1))
        h3 = F.gelu(self.conv3(h2, edge_index) + self.proj3(h2))
        h4 = F.gelu(self.conv4(h3, edge_index) + self.proj4(h3))
        h5 = F.gelu(self.conv5(h4, edge_index))
        combined = torch.cat([h1, h2, h3, h4, h5], dim=1)
        return self.head(combined)

# --- EXACT COPIES FROM TRAINING SCRIPTS ---
def add_features_13(batch):
    x, ptr = batch.x, batch.ptr
    first = ptr[:-1]; sizes = ptr[1:] - ptr[:-1]
    t0, x0, y0, z0 = [torch.repeat_interleave(x[first, i], sizes) for i in [1,2,3,4]]
    dt, dx, dy, dz = x[:,1]-t0, x[:,2]-x0, x[:,3]-y0, x[:,4]-z0
    dr2 = dx**2 + dy**2 + dz**2; dr = torch.sqrt(dr2 + 1e-8)
    s2 = (C_WATER * dt)**2 - dr2
    r = torch.sqrt(x[:, 2]**2 + x[:, 3]**2 + 1e-8)
    phi = torch.atan2(x[:, 3], x[:, 2])
    rho = torch.sqrt(x[:, 2]**2 + x[:, 3]**2 + x[:, 4]**2 + 1e-8)
    cosT = x[:, 4] / (rho + 1e-8)
    tof = dt - dr/C_WATER
    ext = torch.stack([s2, dt, dr, r, phi, rho, cosT, tof], dim=1)
    batch.x = torch.cat([x, ext], dim=1)
    return batch

def add_features_21(batch):
    device = batch.x.device
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
    x_bin, y_bin = torch.round(x[:, 2] / 0.03), torch.round(x[:, 3] / 0.03)
    xy_bins = torch.stack([b_idx, x_bin, y_bin], dim=1)
    _, str_ids = torch.unique(xy_bins, dim=0, return_inverse=True)
    hits_on_string = scatter(torch.ones_like(x[:, 0]), str_ids, dim=0, reduce='sum')[str_ids]
    max_z = scatter(x[:, 4], str_ids, dim=0, reduce='max')[str_ids]
    min_z = scatter(x[:, 4], str_ids, dim=0, reduce='min')[str_ids]
    z_span = max_z - min_z
    event_n_hits = sizes.float(); n_hits = torch.gather(event_n_hits, 0, b_idx)
    event_t_min = scatter(x[:, 1], b_idx, dim=0, reduce='min'); event_t_max = scatter(x[:, 1], b_idx, dim=0, reduce='max')
    duration = torch.gather(event_t_max - event_t_min, 0, b_idx)

    raw_extra = torch.stack([s2, dt, dr, r_xy, phi, rho, cosTheta, tof, mean_dist_neigh, neigh_charge, q_rel_mean, cos_alpha, hits_on_string, z_span, n_hits, duration], dim=1)
    norm_extra = (raw_extra - STATS_21['means'].to(device)) / (STATS_21['stds'].to(device) + 1e-8)
    batch.x = torch.cat([x, norm_extra], dim=1)
    return batch

def evaluate(model, mode='13'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.pt")))
    all_labels, all_probs = [], []
    with torch.no_grad():
        for f in files:
            data_list = torch.load(f, weights_only=False)
            loader = DataLoader(data_list, batch_size=512)
            for batch in loader:
                batch = add_features_13(batch.to(device)) if mode=='13' else add_features_21(batch.to(device))
                out = model(batch.x, batch.edge_index)
                all_probs.extend(F.softmax(out, dim=1)[:, 1].cpu().numpy())
                all_labels.extend(batch.y.cpu().numpy())
    p, r, _ = precision_recall_curve(all_labels, all_probs)
    return np.interp(0.9, r[::-1], p[::-1])

if __name__ == "__main__":
    print("Comparing models on FULL validation set...")
    m13 = BaseResGCN(13); m13.load_state_dict(torch.load(CKPT_13, weights_only=True))
    p9_13 = evaluate(m13, '13')
    print(f"Model 13-feat (1500 ep): P@R0.9 = {p9_13:.4f}")
    
    m21 = BaseResGCN(21); m21.load_state_dict(torch.load(CKPT_21, weights_only=True))
    p9_21 = evaluate(m21, '21')
    print(f"Model 21-feat ( 800 ep): P@R0.9 = {p9_21:.4f}")
