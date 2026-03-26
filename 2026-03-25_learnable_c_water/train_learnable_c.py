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
import math

DATA_DIR = "data_processed"
PROJECT_DIR = "2026-03-25_learnable_c_water"
BATCH_SIZE = 128
ACCUMULATION_STEPS = 2 
VIRTUAL_EPOCH_SIZE = 800 
TOTAL_VIRTUAL_EPOCHS = 1000
NUM_WORKERS = 4

os.makedirs(f"{PROJECT_DIR}/checkpoints", exist_ok=True)
os.makedirs(f"{PROJECT_DIR}/plots", exist_ok=True)

class ChunkedDataset(torch.utils.data.IterableDataset):
    def __init__(self, split='train', shuffle=True):
        super().__init__()
        self.split = split
        self.files = sorted(glob.glob(os.path.join(DATA_DIR, split, "chunk_*.pt")))
        self.shuffle = shuffle
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            iter_files = self.files
        else:
            per_worker = int(math.ceil(len(self.files) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_files = self.files[worker_id * per_worker : (worker_id + 1) * per_worker]
        
        try:
            while True:
                file_list = iter_files.copy()
                if self.shuffle: random.shuffle(file_list)
                for file in file_list:
                    data_list = torch.load(file, weights_only=False)
                    if self.shuffle: random.shuffle(data_list)
                    for data in data_list: yield data
        except GeneratorExit:
            return

def add_extended_features_vectorized(batch, c_water):
    x = batch.x
    ptr = batch.ptr
    first_hit_indices = ptr[:-1]
    graph_sizes = ptr[1:] - ptr[:-1]
    t0 = torch.repeat_interleave(x[first_hit_indices, 1], graph_sizes)
    x0 = torch.repeat_interleave(x[first_hit_indices, 2], graph_sizes)
    y0 = torch.repeat_interleave(x[first_hit_indices, 3], graph_sizes)
    z0 = torch.repeat_interleave(x[first_hit_indices, 4], graph_sizes)
    dt = x[:, 1] - t0
    dx, dy, dz = x[:, 2]-x0, x[:, 3]-y0, x[:, 4]-z0
    dr2 = dx**2 + dy**2 + dz**2
    dr = torch.sqrt(dr2 + 1e-8)
    s2 = (c_water * dt)**2 - dr2
    r = torch.sqrt(x[:, 2]**2 + x[:, 3]**2 + 1e-8)
    phi = torch.atan2(x[:, 3], x[:, 2])
    rho = torch.sqrt(x[:, 2]**2 + x[:, 3]**2 + x[:, 4]**2 + 1e-8)
    cosTheta = x[:, 4] / (rho + 1e-8)
    tof_res = dt - dr / (c_water + 1e-8)
    ext = torch.stack([s2, dt, dr, r, phi, rho, cosTheta, tof_res], dim=1)
    batch.x = torch.cat([x, ext], dim=1)
    return batch

class JKResGCN_LearnableC(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.tensor(0.0)) 
        self.conv1 = GCNConv(in_channels, 512)
        self.conv2 = GCNConv(512, 512)
        self.conv3 = GCNConv(512, 512)
        self.conv4 = GCNConv(512, 512)
        self.conv5 = GCNConv(512, 512)
        self.proj1 = torch.nn.Linear(in_channels, 512)
        self.proj2 = torch.nn.Linear(512, 512)
        self.proj3 = torch.nn.Linear(512, 512)
        self.proj4 = torch.nn.Linear(512, 512)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(512 * 5, 1024),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(1024, 512),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(512, out_channels)
        )

    def get_c_water(self):
        return 0.220 + (0.230 - 0.220) * torch.sigmoid(self.alpha)

    def forward(self, batch):
        c_water = self.get_c_water()
        batch = add_extended_features_vectorized(batch, c_water)
        h1 = F.gelu(self.conv1(batch.x, batch.edge_index) + self.proj1(batch.x))
        h2 = F.gelu(self.conv2(h1, batch.edge_index) + self.proj2(h1))
        h3 = F.gelu(self.conv3(h2, batch.edge_index) + self.proj3(h2))
        h4 = F.gelu(self.conv4(h3, batch.edge_index) + self.proj4(h3))
        h5 = F.gelu(self.conv5(h4, batch.edge_index))
        combined = torch.cat([h1, h2, h3, h4, h5], dim=1)
        return self.head(combined)

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
            batch = batch.to(device)
            out = model(batch)
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
    print(f"Device: {device} | LEARNABLE C_WATER | OPTIMIZED DATA LOADING")
    
    model = JKResGCN_LearnableC(13, 2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) 
    criterion = torch.nn.CrossEntropyLoss()
    
    # Resume Logic
    start_epoch = 1
    ckpts = glob.glob(f"{PROJECT_DIR}/checkpoints/model_learn_c_*.pt")
    if ckpts:
        latest_ckpt = max(ckpts, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        start_epoch = int(latest_ckpt.split('_')[-1].split('.')[0]) + 1
        print(f"Resuming from {latest_ckpt} (Epoch {start_epoch})")
        model.load_state_dict(torch.load(latest_ckpt, weights_only=True))
    
    train_loader = DataLoader(ChunkedDataset('train'), batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)
    train_eval_loader = DataLoader(ChunkedDataset('train', shuffle=True), batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(ChunkedDataset('val', shuffle=False), batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)
    
    epoch = start_epoch
    start_time = time.time()
    
    model.train()
    optimizer.zero_grad()
    for i, batch in enumerate(train_loader):
        batch = batch.to(device)
        out = model(batch)
        loss = criterion(out, batch.y) / ACCUMULATION_STEPS
        loss.backward()
        
        if (i + 1) % ACCUMULATION_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()
            
            if (i // ACCUMULATION_STEPS + 1) % VIRTUAL_EPOCH_SIZE == 0:
                epoch_time = time.time() - start_time
                current_c = model.get_c_water().item()
                t_loss, t_prec, t_rec, t_p9 = evaluate(model, train_eval_loader, device, criterion)
                v_loss, v_prec, v_rec, v_p9 = evaluate(model, val_loader, device, criterion)
                
                print(f"Epoch {epoch:04d} | Time: {epoch_time:.1f}s | C_Water: {current_c:.6f} | Opt: Multiprocessing")
                print(f"  Train: Loss {t_loss:.4f} | Prec {t_prec:.4f} | Rec {t_rec:.4f} | P@R0.9 {t_p9:.4f}")
                print(f"  Val  : Loss {v_loss:.4f} | Prec {v_prec:.4f} | Rec {v_rec:.4f} | P@R0.9 {v_p9:.4f}")
                
                if epoch % 50 == 0:
                    torch.save(model.state_dict(), f"{PROJECT_DIR}/checkpoints/model_learn_c_{epoch}.pt")
                
                epoch += 1
                if epoch > TOTAL_VIRTUAL_EPOCHS: break
                start_time = time.time()
                model.train()

if __name__ == "__main__":
    train()
