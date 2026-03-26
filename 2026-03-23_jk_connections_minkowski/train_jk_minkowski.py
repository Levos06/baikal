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
PROJECT_DIR = "2026-03-23_jk_connections_minkowski"
BATCH_SIZE = 128
VIRTUAL_EPOCH_SIZE = 800
TOTAL_VIRTUAL_EPOCHS = 400
C_WATER = 0.225 # Speed of light in water, m/ns

os.makedirs(f"{PROJECT_DIR}/checkpoints", exist_ok=True)
os.makedirs(f"{PROJECT_DIR}/plots", exist_ok=True)

class ChunkedDataset(torch.utils.data.IterableDataset):
    def __init__(self, split='train', shuffle=True):
        super().__init__()
        self.split = split
        self.files = sorted(glob.glob(os.path.join(DATA_DIR, split, "chunk_*.pt")))
        self.shuffle = shuffle

    def __iter__(self):
        while True:
            file_list = self.files.copy()
            if self.shuffle: random.shuffle(file_list)
            for file in file_list:
                try:
                    data_list = torch.load(file, weights_only=False)
                    if self.shuffle: random.shuffle(data_list)
                    for data in data_list: yield data
                except Exception as e:
                    continue

def add_minkowski_features(batch):
    # batch.x has [Q, T, X, Y, Z]
    # We assume hits are sorted by time (they are in our dataset)
    # Let's compute features relative to the first hit in EACH graph in the batch
    
    new_x_list = []
    # ptr tells us where each graph starts in the batch
    ptr = batch.ptr
    for i in range(len(ptr) - 1):
        start, end = ptr[i], ptr[i+1]
        x_graph = batch.x[start:end]
        
        # Reference: first hit
        t0, x0, y0, z0 = x_graph[0, 1], x_graph[0, 2], x_graph[0, 3], x_graph[0, 4]
        
        dt = x_graph[:, 1] - t0
        dx = x_graph[:, 2] - x0
        dy = x_graph[:, 3] - y0
        dz = x_graph[:, 4] - z0
        dr2 = dx**2 + dy**2 + dz**2
        dr = torch.sqrt(dr2)
        
        s2 = (C_WATER * dt)**2 - dr2
        
        # New features: [Q, T, X, Y, Z, s2, dt, dr]
        mink_feats = torch.stack([s2, dt, dr], dim=1)
        new_x_graph = torch.cat([x_graph, mink_feats], dim=1)
        new_x_list.append(new_x_graph)
    
    batch.x = torch.cat(new_x_list, dim=0)
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
        
        # Jumping Knowledge Head (Concat all layers)
        self.fc = torch.nn.Linear(512 * 5, out_channels)

    def forward(self, x, edge_index):
        h1 = F.gelu(self.conv1(x, edge_index) + self.proj1(x))
        h2 = F.gelu(self.conv2(h1, edge_index) + self.proj2(h1))
        h3 = F.gelu(self.conv3(h2, edge_index) + self.proj3(h2))
        h4 = F.gelu(self.conv4(h3, edge_index) + self.proj4(h3))
        h5 = F.gelu(self.conv5(h4, edge_index)) # Output of last conv
        
        # JK Concatenation
        combined = torch.cat([h1, h2, h3, h4, h5], dim=1)
        return self.fc(combined)

def get_precision_at_recall(labels, probs, target_recall=0.9):
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    if len(recall) == 0 or np.max(recall) < target_recall: return 0.0
    return np.interp(target_recall, recall[::-1], precision[::-1])

def evaluate(model, loader, device, criterion, num_batches=100):
    model.eval()
    total_loss, all_labels, all_probs = 0, [], []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= num_batches: break
            batch = add_minkowski_features(batch.to(device))
            out = model(batch.x, batch.edge_index)
            loss = criterion(out, batch.y)
            total_loss += loss.item()
            probs = F.softmax(out, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
    if not all_labels: return 0,0,0,0
    all_labels, all_probs = np.array(all_labels), np.array(all_probs)
    p_at_r9 = get_precision_at_recall(all_labels, all_probs, 0.9)
    preds = (all_probs > 0.5).astype(int)
    return total_loss/(i+1), precision_score(all_labels, preds, zero_division=0), recall_score(all_labels, preds, zero_division=0), p_at_r9

def plot_metrics(history, path):
    epochs = range(1, len(history['t_loss']) + 1)
    plt.figure(figsize=(16, 10))
    for i, m in enumerate(['loss', 'prec', 'rec', 'p9']):
        plt.subplot(2, 2, i+1)
        plt.plot(epochs, history[f't_{m}'], label='Train')
        plt.plot(epochs, history[f'v_{m}'], label='Val')
        plt.title(m.upper()); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.savefig(path); plt.close()

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available(): torch.cuda.set_device(0) 
    print(f"Device: {device} | JK-MINKOWSKI | GCN 5Layer (8 Features)")
    
    model = JKResGCN(8, 2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    train_loader = DataLoader(ChunkedDataset('train'), batch_size=BATCH_SIZE)
    train_eval_loader = DataLoader(ChunkedDataset('train', shuffle=True), batch_size=BATCH_SIZE)
    val_loader = DataLoader(ChunkedDataset('val', shuffle=False), batch_size=BATCH_SIZE)
    
    history = {k: [] for k in ['t_loss', 't_prec', 't_rec', 't_p9', 'v_loss', 'v_prec', 'v_rec', 'v_p9']}
    batch_count, epoch = 0, 1
    start_time = time.time()
    
    model.train()
    for batch in train_loader:
        batch = add_minkowski_features(batch.to(device))
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)
        loss = criterion(out, batch.y)
        loss.backward(); optimizer.step()
        
        batch_count += 1
        if batch_count % VIRTUAL_EPOCH_SIZE == 0:
            t_met = evaluate(model, train_eval_loader, device, criterion)
            v_met = evaluate(model, val_loader, device, criterion)
            
            for i, m in enumerate(['loss', 'prec', 'rec', 'p9']):
                history[f't_{m}'].append(t_met[i])
                history[f'v_{m}'].append(v_met[i])
            
            print(f"Epoch {epoch:04d} | T-P@R0.9: {t_met[3]:.4f} | V-P@R0.9: {v_met[3]:.4f}")
            if epoch % 10 == 0:
                torch.save(model.state_dict(), f"{PROJECT_DIR}/checkpoints/model_jk_{epoch}.pt")
                plot_metrics(history, f"{PROJECT_DIR}/plots/metrics_jk.png")
            epoch += 1
            if epoch > TOTAL_VIRTUAL_EPOCHS: break
            model.train()

if __name__ == "__main__":
    train()
