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
PROJECT_DIR = "2026-03-24_jk_minkowski_v2_085_chase"
CHECKPOINT_PREV = "2026-03-24_extended_minkowski_jk/checkpoints/model_ext_jk_400.pt"
BATCH_SIZE = 128
ACCUMULATION_STEPS = 2 # Effective batch size = 256
VIRTUAL_EPOCH_SIZE = 800 # 100k events
TOTAL_VIRTUAL_EPOCHS = 400
C_WATER = 0.225

os.makedirs(f"{PROJECT_DIR}/checkpoints", exist_ok=True)
os.makedirs(f"{PROJECT_DIR}/plots", exist_ok=True)

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
            return

def add_extended_features_vectorized(batch):
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
    s2 = (C_WATER * dt)**2 - dr2
    r = torch.sqrt(x[:, 2]**2 + x[:, 3]**2 + 1e-8)
    phi = torch.atan2(x[:, 3], x[:, 2])
    rho = torch.sqrt(x[:, 2]**2 + x[:, 3]**2 + x[:, 4]**2 + 1e-8)
    cosTheta = x[:, 4] / (rho + 1e-8)
    tof_res = dt - dr/C_WATER
    ext = torch.stack([s2, dt, dr, r, phi, rho, cosTheta, tof_res], dim=1)
    batch.x = torch.cat([x, ext], dim=1)
    return batch

class JKResGCN_v2(torch.nn.Module):
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
        
        # New MLP Head with Dropout
        self.head = torch.nn.Sequential(
            torch.nn.Linear(512 * 5, 1024),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(1024, 512),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
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
    print(f"Device: {device} | JK-MINKOWSKI V2 | CHASE FOR 0.85")
    
    model = JKResGCN_v2(13, 2).to(device)
    
    # Load Pretrained Body
    if os.path.exists(CHECKPOINT_PREV):
        print(f"Loading pretrained body from: {CHECKPOINT_PREV}")
        pretrained_dict = torch.load(CHECKPOINT_PREV, weights_only=True)
        model_dict = model.state_dict()
        # Filter out the old 'fc' layer weights
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5) # Slightly lower LR for fine-tuning
    criterion = torch.nn.CrossEntropyLoss()
    
    train_loader = DataLoader(ChunkedDataset('train'), batch_size=BATCH_SIZE)
    train_eval_loader = DataLoader(ChunkedDataset('train', shuffle=True), batch_size=BATCH_SIZE)
    val_loader = DataLoader(ChunkedDataset('val', shuffle=False), batch_size=BATCH_SIZE)
    
    history = {k: [] for k in ['t_loss', 't_prec', 't_rec', 't_p9', 'v_loss', 'v_prec', 'v_rec', 'v_p9']}
    batch_count, epoch = 0, 1
    start_time = time.time()
    
    model.train()
    optimizer.zero_grad()
    for i, batch in enumerate(train_loader):
        batch = add_extended_features_vectorized(batch.to(device))
        out = model(batch.x, batch.edge_index)
        loss = criterion(out, batch.y) / ACCUMULATION_STEPS
        loss.backward()
        
        if (i + 1) % ACCUMULATION_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()
            batch_count += 1
            
            if batch_count % VIRTUAL_EPOCH_SIZE == 0:
                epoch_time = time.time() - start_time
                t_met = evaluate(model, train_eval_loader, device, criterion)
                v_met = evaluate(model, val_loader, device, criterion)
                for j, m in enumerate(['loss', 'prec', 'rec', 'p9']):
                    history[f't_{m}'].append(t_met[j]); history[f'v_{m}'].append(v_met[j])
                print(f"Epoch {epoch:04d} | V-Loss: {v_met[0]:.4f} | V-P@R0.9: {v_met[3]:.4f}")
                if epoch % 10 == 0:
                    torch.save(model.state_dict(), f"{PROJECT_DIR}/checkpoints/model_v2_jk_{epoch}.pt")
                epoch += 1
                if epoch > TOTAL_VIRTUAL_EPOCHS: break
                start_time = time.time()
                model.train()

if __name__ == "__main__":
    train()
