import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.utils import scatter
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import os
import time
import glob
import random
import numpy as np
import sys
import re
from sklearn.metrics import precision_score, recall_score, precision_recall_curve

# --- CONFIG ---
DATA_DIR = "data_processed_50k"
PROJECT_DIR = "2026-03-31_gat_marathon"
BATCH_SIZE = 512
NUM_WORKERS = 4
TOTAL_EPOCHS = 1000
VIRTUAL_EPOCH_SIZE = 200
C_WATER = 0.225

# Load normalization stats
STATS = torch.load('tools/feat_stats_21.pt', weights_only=True)
MEANS = STATS['means']
STDS = STATS['stds']

def add_features_21_norm(batch):
    device = batch.x.device
    x, edge_index, ptr, b_idx = batch.x, batch.edge_index, batch.ptr, batch.batch
    num_nodes = x.size(0)
    first = ptr[:-1]; sizes = ptr[1:] - ptr[:-1]
    t0, x0, y0, z0 = [torch.repeat_interleave(x[first, i], sizes) for i in [1,2,3,4]]
    dt, dx, dy, dz = x[:,1]-t0, x[:,2]-x0, x[:,3]-y0, x[:,4]-z0
    dr2 = dx**2 + dy**2 + dz**2; dr = torch.sqrt(dr2 + 1e-8)
    s2, tof = (C_WATER * dt)**2 - dr2, dt - dr/C_WATER
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
    norm_extra = (raw_extra - MEANS.to(device)) / (STDS.to(device) + 1e-8)
    batch.x = torch.cat([x, norm_extra], dim=1)
    return batch

class JKResGATv2(torch.nn.Module):
    def __init__(self, in_channels, out_channels, heads=4):
        super().__init__()
        self.conv1 = GATv2Conv(in_channels, 128, heads=heads)
        self.conv2 = GATv2Conv(128 * heads, 128, heads=heads)
        self.conv3 = GATv2Conv(128 * heads, 128, heads=heads)
        self.conv4 = GATv2Conv(128 * heads, 128, heads=heads)
        self.proj1 = torch.nn.Linear(in_channels, 128 * heads)
        self.proj2 = torch.nn.Linear(128 * heads, 128 * heads)
        self.proj3 = torch.nn.Linear(128 * heads, 128 * heads)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(128 * heads * 4, 1024), torch.nn.GELU(), torch.nn.Dropout(0.1),
            torch.nn.Linear(1024, 512), torch.nn.GELU(), torch.nn.Dropout(0.1),
            torch.nn.Linear(512, out_channels)
        )
    def forward(self, x, edge_index):
        h1 = F.gelu(self.conv1(x, edge_index) + self.proj1(x))
        h2 = F.gelu(self.conv2(h1, edge_index) + self.proj2(h1))
        h3 = F.gelu(self.conv3(h2, edge_index) + self.proj3(h2))
        h4 = F.gelu(self.conv4(h3, edge_index))
        combined = torch.cat([h1, h2, h3, h4], dim=1)
        return self.head(combined)

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

def evaluate(model, loader, device, criterion, num_batches=100):
    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= num_batches: break
            batch = add_features_21_norm(batch.to(device))
            out = model(batch.x, batch.edge_index)
            all_probs.extend(F.softmax(out, dim=1)[:, 1].cpu().numpy()); all_labels.extend(batch.y.cpu().numpy())
    p, r, _ = precision_recall_curve(np.array(all_labels), np.array(all_probs))
    return np.interp(0.9, r[::-1], p[::-1])

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting GATv2 REVOLUTION on {device} (GeneratorExit Fix)")
    model = JKResGATv2(21, 2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    train_loader = DataLoader(MediumDataset('train'), batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(MediumDataset('val'), batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    
    epoch = 1
    start_time = time.time()
    model.train()
    for i, batch in enumerate(train_loader):
        batch = add_features_21_norm(batch.to(device, non_blocking=True))
        optimizer.zero_grad(set_to_none=True); out = model(batch.x, batch.edge_index)
        loss = criterion(out, batch.y); loss.backward(); optimizer.step()
        
        if (i + 1) % VIRTUAL_EPOCH_SIZE == 0:
            v_p9 = evaluate(model, val_loader, device, criterion)
            print(f"Epoch {epoch:04d} | Time: {time.time()-start_time:.1f}s | Val P@R0.9: {v_p9:.4f}")
            if epoch % 50 == 0:
                torch.save(model.state_dict(), f"{PROJECT_DIR}/checkpoints/model_gat_{epoch}.pt")
            epoch += 1
            if epoch > TOTAL_EPOCHS: break
            start_time = time.time(); model.train()

if __name__ == "__main__":
    train()
