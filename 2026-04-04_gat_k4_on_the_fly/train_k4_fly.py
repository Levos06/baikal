import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import scatter
from sklearn.metrics import precision_score, recall_score, precision_recall_curve
import os
import time
import sys

# --- CONFIG ---
FILE_PATH = "/home3/ivkhar/Baikal/data/normed/baikal_mc2020_multi_split_0924mid_eq_norm.h5"
PROJECT_DIR = "/home/levos/experiments/2026-04-04_gat_k4_on_the_fly"
BATCH_SIZE = 128
NUM_WORKERS = 0
TOTAL_EPOCHS = 2000
VIRTUAL_EPOCH_SIZE = 400
DEVICE = torch.device('cuda:0') # Using the 4090

# Load stats
STATS = torch.load('/home/levos/experiments/tools/feat_stats_21.pt', weights_only=True)
MEANS, STDS = STATS['means'].to(DEVICE), STATS['stds'].to(DEVICE)

class BaikalH5Dataset(Dataset):
    def __init__(self, file_path, start_ev=0, num_events=1000, k=4):
        super().__init__()
        self.file_path = file_path
        self.start_ev = start_ev
        self.num_events = num_events
        self.k = k
        print(f"Loading {num_events} event starts from HDF5...", flush=True)
        with h5py.File(self.file_path, 'r') as f:
            self.starts = f['train/ev_starts/data'][start_ev : start_ev + num_events + 1]
        print("Starts loaded.", flush=True)
            
    def len(self): return self.num_events

    def get(self, idx):
        with h5py.File(self.file_path, 'r') as f:
            start, end = self.starts[idx], self.starts[idx+1]
            x_raw = torch.from_numpy(f['train/data/data'][start:end]).float()
            y = torch.from_numpy((f['train/labels/data'][start:end] != 0).astype(np.int64))
            
            num_nodes = x_raw.size(0)
            if num_nodes <= 1:
                edge_index = torch.zeros((2, 0), dtype=torch.long)
            else:
                indices = np.arange(num_nodes)
                mask = (np.abs(indices[:, None] - indices) <= self.k) & (indices[:, None] != indices)
                edge_index = torch.from_numpy(np.array(np.where(mask))).long()
            
            return Data(x=x_raw, edge_index=edge_index, y=y)

def add_features_21_norm(batch):
    x, edge_index, ptr, b_idx = batch.x, batch.edge_index, batch.ptr, batch.batch
    num_nodes = x.size(0)
    sizes = ptr[1:] - ptr[:-1]
    first = ptr[:-1]
    t0, x0, y0, z0 = [torch.repeat_interleave(x[first, i], sizes) for i in [1,2,3,4]]
    dt, dx, dy, dz = x[:,1]-t0, x[:,2]-x0, x[:,3]-y0, x[:,4]-z0
    dr2 = dx**2 + dy**2 + dz**2; dr = torch.sqrt(dr2 + 1e-8)
    s2, tof = (0.225 * dt)**2 - dr2, dt - dr/0.225
    r_xy, phi = torch.sqrt(x[:, 2]**2 + x[:, 3]**2 + 1e-8), torch.atan2(x[:, 3], x[:, 2])
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
    max_z, min_z = scatter(x[:, 4], str_ids, dim=0, reduce='max')[str_ids], scatter(x[:, 4], str_ids, dim=0, reduce='min')[str_ids]
    z_span = max_z - min_z
    event_n_hits = sizes.float(); n_hits = torch.gather(event_n_hits, 0, b_idx)
    duration = torch.gather(scatter(x[:, 1], b_idx, dim=0, reduce='max') - scatter(x[:, 1], b_idx, dim=0, reduce='min'), 0, b_idx)
    raw_extra = torch.stack([s2, dt, dr, r_xy, phi, rho, cosTheta, tof, mean_dist_neigh, neigh_charge, q_rel_mean, cos_alpha, hits_on_string, z_span, n_hits, duration], dim=1)
    batch.x = torch.cat([x, (raw_extra - MEANS) / (STDS + 1e-8)], dim=1)
    return batch

class GATv2_Standard(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.projs = torch.nn.ModuleList()
        self.convs.append(GATv2Conv(in_channels, 64, heads=4)); self.projs.append(torch.nn.Linear(in_channels, 256))
        for _ in range(3):
            self.convs.append(GATv2Conv(256, 64, heads=4)); self.projs.append(torch.nn.Identity())
        self.head = torch.nn.Sequential(torch.nn.Linear(256, 512), torch.nn.GELU(), torch.nn.Linear(512, 256), torch.nn.GELU(), torch.nn.Linear(256, out_channels))
    def forward(self, x, edge_index):
        h = x
        for i in range(4): h = F.gelu(self.convs[i](h, edge_index) + self.projs[i](h))
        return self.head(h)

def calculate_metrics(labels, probs):
    p, r, _ = precision_recall_curve(labels, probs)
    p_at_r09 = np.interp(0.9, r[::-1], p[::-1])
    preds = (probs > 0.5).astype(int)
    return precision_score(labels, preds, zero_division=0), recall_score(labels, preds, zero_division=0), p_at_r09

def evaluate(model, loader, num_batches=50):
    model.eval()
    all_labels, all_probs, losses = [], [], []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= num_batches: break
            batch = add_features_21_norm(batch.to(DEVICE))
            out = model(batch.x, batch.edge_index)
            losses.append(F.cross_entropy(out, batch.y).item())
            all_probs.extend(F.softmax(out, dim=1)[:, 1].cpu().numpy()); all_labels.extend(batch.y.cpu().numpy())
    return np.mean(losses), *calculate_metrics(np.array(all_labels), np.array(all_probs))

def train():
    print(f"Starting GATv2 k=4 On-The-Fly Experiment on {DEVICE}", flush=True)
    model = GATv2_Standard(21, 2).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    print("Initializing datasets...", flush=True)
    # Reducing train size to 500k for stability test
    train_dataset = BaikalH5Dataset(FILE_PATH, 0, 500000, k=4)
    val_dataset = BaikalH5Dataset(FILE_PATH, 1000000, 50000, k=4)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    
    log_file = os.path.join(PROJECT_DIR, "train_k4_fly.log")
    with open(log_file, "w") as f:
        f.write("Epoch | Time_Start | Duration | LR | T_Loss | T_Prec | T_Rec | T_P@R0.9 | V_Loss | V_Prec | V_Rec | V_P@R0.9\n")
    
    print("Starting training loop...", flush=True)
    epoch = 1
    start_time = time.time()
    t_labels, t_probs, t_losses = [], [], []
    
    model.train()
    for i, batch in enumerate(train_loader):
        batch = add_features_21_norm(batch.to(DEVICE, non_blocking=True))
        optimizer.zero_grad(set_to_none=True)
        out = model(batch.x, batch.edge_index)
        loss = F.cross_entropy(out, batch.y); loss.backward(); optimizer.step()
        t_losses.append(loss.item()); t_probs.extend(F.softmax(out, dim=1)[:, 1].detach().cpu().numpy()); t_labels.extend(batch.y.cpu().numpy())
        
        if (i + 1) % VIRTUAL_EPOCH_SIZE == 0:
            duration = time.time() - start_time; start_clock = time.strftime("%H:%M:%S", time.localtime(start_time))
            t_loss, t_prec, t_rec, t_p9 = np.mean(t_losses), *calculate_metrics(np.array(t_labels), np.array(t_probs))
            v_loss, v_prec, v_rec, v_p9 = evaluate(model, val_loader)
            log_str = f"{epoch:04d} | {start_clock} | {duration:5.1f}s | 1.0e-04 | {t_loss:.4f} | {t_prec:.4f} | {t_rec:.4f} | {t_p9:.4f} | {v_loss:.4f} | {v_prec:.4f} | {v_rec:.4f} | {v_p9:.4f}"
            print(log_str, flush=True)
            with open(log_file, "a") as f: f.write(log_str + "\n")
            if epoch % 100 == 0: torch.save(model.state_dict(), f"{PROJECT_DIR}/checkpoints/model_k4_{epoch}.pt")
            epoch += 1
            if epoch > TOTAL_EPOCHS: break
            t_labels, t_probs, t_losses = [], [], []; start_time = time.time(); model.train()

if __name__ == "__main__":
    train()
