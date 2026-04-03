import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import scatter
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import os
import time
import glob
import random
import numpy as np
import sys
from sklearn.metrics import precision_recall_curve

# --- SILENCE SYSTEM NOISE ---
class StderrFilter:
    def __init__(self, stream):
        self.stream = stream
    def write(self, data):
        if "GeneratorExit" not in data and "RuntimeError: generator ignored" not in data:
            self.stream.write(data)
            self.stream.flush()
    def flush(self): self.stream.flush()
sys.stderr = StderrFilter(sys.stderr)

# --- CONFIG ---
DATA_DIR = "data_processed_50k"
PROJECT_DIR = "2026-03-30_refined_13features"
BATCH_SIZE = 512
NUM_WORKERS = 4
TOTAL_EPOCHS = 1000
VIRTUAL_EPOCH_SIZE = 200
C_WATER = 0.225
Z_STD_FACTOR = 0.974272

# Load stats for 21 feat set (we will select our 13 from here)
STATS = torch.load('tools/feat_stats_21.pt', weights_only=True)
M21 = STATS['means']
S21 = STATS['stds']

# Indices of the 13 chosen features in the 21-vector (Relative to extra part)
# We need: dt(1), dr(2), r_xy(3), rho(5), cosTheta(6), ToF_Res(7), NeighDist(8), NeighQ(9)
# Plus the 5 base features.
EXTRA_INDICES = [1, 2, 3, 5, 6, 7, 8, 9]

def add_features_refined(batch):
    device = batch.x.device
    x, edge_index, ptr, b_idx = batch.x, batch.edge_index, batch.ptr, batch.batch
    num_nodes = x.size(0)
    
    # 1. UPGRADE Z (independent event centering)
    z_raw = x[:, 4]
    z_means = scatter(z_raw, b_idx, dim=0, reduce='mean')
    z_centered = z_raw - torch.gather(z_means, 0, b_idx)
    z_final = z_centered / Z_STD_FACTOR
    
    # Update base X with refined Z
    x_new = x.clone()
    x_new[:, 4] = z_final
    
    # 2. Physics Calculations
    first = ptr[:-1]; sizes = ptr[1:] - ptr[:-1]
    t0, x0, y0, z0 = [torch.repeat_interleave(x[first, i], sizes) for i in [1,2,3,4]]
    dt, dx, dy, dz = x[:,1]-t0, x[:,2]-x0, x[:,3]-y0, x[:,4]-z0
    dr2 = dx**2 + dy**2 + dz**2; dr = torch.sqrt(dr2 + 1e-8)
    
    # Selected Extra Features
    tof = dt - dr/C_WATER
    r_xy = torch.sqrt(x[:, 2]**2 + x[:, 3]**2 + 1e-8)
    rho = torch.sqrt(x[:, 2]**2 + x[:, 3]**2 + x[:, 4]**2 + 1e-8)
    cosTheta = x[:, 4] / (rho + 1e-8)
    
    row, col = edge_index
    dist_edges = torch.sqrt(torch.sum((x[row, 2:5] - x[col, 2:5])**2, dim=1) + 1e-8)
    mean_dist_neigh = scatter(dist_edges, row, dim=0, dim_size=num_nodes, reduce='mean')
    neigh_charge = scatter(x[col, 0], row, dim=0, dim_size=num_nodes, reduce='sum')

    # Raw extra set (8 selected)
    raw_extra = torch.stack([
        dt, dr, r_xy, rho, cosTheta, tof, mean_dist_neigh, neigh_charge
    ], dim=1)
    
    # Normalize using stored 21-feat stats
    # Map our selected indices to the global stats
    idx_map = [6, 7, 8, 10, 11, 12, 13, 14] # Indices in 21-vector (0-based)
    m_sub = M21[np.array(idx_map)-5].to(device) # Stats are for F5-F20
    s_sub = S21[np.array(idx_map)-5].to(device)
    
    norm_extra = (raw_extra - m_sub) / (s_sub + 1e-8)
    
    batch.x = torch.cat([x_new, norm_extra], dim=1) # 5 + 8 = 13 features
    return batch

class JKResGCN_Refined(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 512); self.conv2 = GCNConv(512, 512); self.conv3 = GCNConv(512, 512)
        self.conv4 = GCNConv(512, 512); self.conv5 = GCNConv(512, 512)
        self.proj1 = torch.nn.Linear(in_channels, 512); self.proj2 = torch.nn.Linear(512, 512)
        self.proj3 = torch.nn.Linear(512, 512); self.proj4 = torch.nn.Linear(512, 512)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(512 * 5, 1024), torch.nn.GELU(), torch.nn.Dropout(0.1),
            torch.nn.Linear(1024, 512), torch.nn.GELU(), torch.nn.Dropout(0.1),
            torch.nn.Linear(512, out_channels)
        )
    def forward(self, x, edge_index):
        h1 = F.gelu(self.conv1(x, edge_index) + self.proj1(x))
        h2 = F.gelu(self.conv2(h1, edge_index) + self.proj2(h1))
        h3 = F.gelu(self.conv3(h2, edge_index) + self.proj3(h2))
        h4 = F.gelu(self.conv4(h3, edge_index) + self.proj4(h3))
        h5 = F.gelu(self.conv5(h4, edge_index))
        combined = torch.cat([h1, h2, h3, h4, h5], dim=1)
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
                        for data in data_list: yield data
                    except: continue
        except GeneratorExit: return

def evaluate(model, loader, device):
    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= 100: break
            batch = add_features_refined(batch.to(device))
            out = model(batch.x, batch.edge_index)
            all_probs.extend(F.softmax(out, dim=1)[:, 1].cpu().numpy()); all_labels.extend(batch.y.cpu().numpy())
    if not all_labels: return 0
    p, r, _ = precision_recall_curve(np.array(all_labels), np.array(all_probs))
    return np.interp(0.9, r[::-1], p[::-1])

def train():
    device = torch.device('cuda')
    print(f"Starting REFINED 13-FEATURE Training on {device}")
    model = JKResGCN_Refined(13, 2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    train_loader = DataLoader(MediumDataset('train'), batch_size=BATCH_SIZE, num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(MediumDataset('val'), batch_size=BATCH_SIZE, num_workers=4, pin_memory=True, persistent_workers=True)
    
    epoch = 1
    start_time = time.time()
    for i, batch in enumerate(train_loader):
        batch = add_features_refined(batch.to(device, non_blocking=True))
        optimizer.zero_grad(set_to_none=True); out = model(batch.x, batch.edge_index)
        loss = criterion(out, batch.y); loss.backward(); optimizer.step()
        
        if (i + 1) % VIRTUAL_EPOCH_SIZE == 0:
            v_p9 = evaluate(model, val_loader, device)
            print(f"Epoch {epoch:04d} | Time: {time.time()-start_time:.1f}s | Val P@R0.9: {v_p9:.4f}")
            if epoch % 50 == 0:
                torch.save(model.state_dict(), f"{PROJECT_DIR}/checkpoints/model_refined_{epoch}.pt")
            epoch += 1
            if epoch > TOTAL_EPOCHS: break
            start_time = time.time()

if __name__ == "__main__":
    train()
