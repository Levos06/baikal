import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import os
import time
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, precision_recall_curve
import numpy as np
import random
import glob

DATA_DIR = "data_processed"
PROJECT_DIR = "2026-03-24_extended_minkowski_jk"
BATCH_SIZE = 128
VIRTUAL_EPOCH_SIZE = 800
TOTAL_VIRTUAL_EPOCHS = 400
C_WATER = 0.225

class ChunkedDataset(torch.utils.data.IterableDataset):
    def __init__(self, split='train', shuffle=True):
        super().__init__()
        self.split = split
        self.files = sorted(glob.glob(os.path.join(DATA_DIR, split, "chunk_*.pt")))
        self.shuffle = shuffle
    def __iter__(self):
        try:
            while True:
                file_list = self.files.copy()
                if self.shuffle: random.shuffle(file_list)
                for file in file_list:
                    data_list = torch.load(file, weights_only=False)
                    if self.shuffle: random.shuffle(data_list)
                    for data in data_list: yield data
        except GeneratorExit:
            return # Properly exit the generator when broken

def add_extended_features_vectorized(batch):
    # Vectorized computation instead of per-graph loop for faster CPU/GPU processing
    # Note: We need per-graph reference (first hit). 
    # To keep it simple but fast, we'll use batch.ptr to find first hits.
    
    x = batch.x # [Q, T, X, Y, Z]
    ptr = batch.ptr
    
    # Indices of first hits in each graph in the batch
    first_hit_indices = ptr[:-1]
    
    # Broadcast t0, x0, y0, z0 to match x dimensions
    # torch.repeat_interleave is perfect for this (it repeats based on graph sizes)
    graph_sizes = ptr[1:] - ptr[:-1]
    
    t0 = torch.repeat_interleave(x[first_hit_indices, 1], graph_sizes)
    x0 = torch.repeat_interleave(x[first_hit_indices, 2], graph_sizes)
    y0 = torch.repeat_interleave(x[first_hit_indices, 3], graph_sizes)
    z0 = torch.repeat_interleave(x[first_hit_indices, 4], graph_sizes)
    
    dt = x[:, 1] - t0
    dx, dy, dz = x[:, 2]-x0, x[:, 3]-y0, x[:, 4]-z0
    dr2 = dx**2 + dy**2 + dz**2
    dr = torch.sqrt(dr2 + 1e-8)
    s2 = (C_WATER * dt)**2 - dr2
    
    # Geometry (global)
    r = torch.sqrt(x[:, 2]**2 + x[:, 3]**2 + 1e-8)
    phi = torch.atan2(x[:, 3], x[:, 2])
    rho = torch.sqrt(x[:, 2]**2 + x[:, 3]**2 + x[:, 4]**2 + 1e-8)
    cosTheta = x[:, 4] / (rho + 1e-8)
    
    # T-ToF (causal residual relative to first hit)
    tof_res = dt - dr/C_WATER
    
    # Extended 13 Features: [Q, T, X, Y, Z] + [s2, dt, dr, r, phi, rho, cosT, tof_res]
    ext = torch.stack([s2, dt, dr, r, phi, rho, cosTheta, tof_res], dim=1)
    batch.x = torch.cat([x, ext], dim=1)
    return batch

class JKResGCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 512)
        self.conv2 = GCNConv(512, 512)
        self.conv3 = GCNConv(512, 512)
        self.conv4 = GCNConv(512, 512)
        self.conv5 = GCNConv(512, 512)
        self.proj1 = torch.nn.Linear(in_channels, 512)
        self.proj2 = torch.nn.Linear(512, 512)
        self.proj3 = torch.nn.Linear(512, 512)
        self.proj4 = torch.nn.Linear(512, 512)
        self.fc = torch.nn.Linear(512 * 5, out_channels)

    def forward(self, x, edge_index):
        h1 = F.gelu(self.conv1(x, edge_index) + self.proj1(x))
        h2 = F.gelu(self.conv2(h1, edge_index) + self.proj2(h1))
        h3 = F.gelu(self.conv3(h2, edge_index) + self.proj3(h2))
        h4 = F.gelu(self.conv4(h3, edge_index) + self.proj4(h3))
        h5 = F.gelu(self.conv5(h4, edge_index))
        combined = torch.cat([h1, h2, h3, h4, h5], dim=1)
        return self.fc(combined)

def get_precision_at_recall(labels, probs, target_recall=0.9):
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    if not len(recall) or np.max(recall) < target_recall: return 0.0
    return np.interp(target_recall, recall[::-1], precision[::-1])

def evaluate(model, loader, device, criterion, num_batches=100):
    model.eval()
    total_loss, all_labels, all_probs = 0, [], []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= num_batches: break
            batch = add_extended_features_vectorized(batch.to(device))
            out = model(batch.x, batch.edge_index)
            loss = criterion(out, batch.y)
            total_loss += loss.item()
            probs = F.softmax(out, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy()); all_labels.extend(batch.y.cpu().numpy())
    if not all_labels: return 0,0,0,0
    all_labels, all_probs = np.array(all_labels), np.array(all_probs)
    p_at_r9 = get_precision_at_recall(all_labels, all_probs, 0.9)
    preds = (all_probs > 0.5).astype(int)
    return total_loss/(i+1), precision_score(all_labels, preds, zero_division=0), recall_score(all_labels, preds, zero_division=0), p_at_r9

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available(): torch.cuda.set_device(0)
    print(f"Device: {device} | EXTENDED MINKOWSKI JK | 13 Features (Resumed from Ep 70)")
    
    model = JKResGCN(13, 2).to(device)
    
    # Load checkpoint 70
    checkpoint_path = f"{PROJECT_DIR}/checkpoints/model_ext_jk_70.pt"
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
        start_epoch = 71
    else:
        start_epoch = 1

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    train_loader = DataLoader(ChunkedDataset('train'), batch_size=BATCH_SIZE)
    train_eval_loader = DataLoader(ChunkedDataset('train', shuffle=True), batch_size=BATCH_SIZE)
    val_loader = DataLoader(ChunkedDataset('val', shuffle=False), batch_size=BATCH_SIZE)
    
    history = {k: [] for k in ['t_loss', 't_prec', 't_rec', 't_p9', 'v_loss', 'v_prec', 'v_rec', 'v_p9']}
    batch_count = 0
    epoch = start_epoch
    start_time = time.time()
    
    model.train()
    for batch in train_loader:
        batch = add_extended_features_vectorized(batch.to(device))
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index); loss = criterion(out, batch.y)
        loss.backward(); optimizer.step()
        
        batch_count += 1
        if batch_count % VIRTUAL_EPOCH_SIZE == 0:
            epoch_time = time.time() - start_time
            t_met = evaluate(model, train_eval_loader, device, criterion)
            v_met = evaluate(model, val_loader, device, criterion)
            for i, m in enumerate(['loss', 'prec', 'rec', 'p9']):
                history[f't_{m}'].append(t_met[i]); history[f'v_{m}'].append(v_met[i])
            print(f"Epoch {epoch:04d} | Time: {epoch_time:.1f}s | T-P@R0.9: {t_met[3]:.4f} | V-P@R0.9: {v_met[3]:.4f}")
            if epoch % 10 == 0:
                torch.save(model.state_dict(), f"{PROJECT_DIR}/checkpoints/model_ext_jk_{epoch}.pt")
            epoch += 1
            if epoch > TOTAL_VIRTUAL_EPOCHS: break
            start_time = time.time()
            model.train()

if __name__ == "__main__":
    train()
