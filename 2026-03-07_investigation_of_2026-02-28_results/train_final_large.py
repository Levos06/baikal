import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from sklearn.metrics import precision_score, recall_score, precision_recall_curve
import os
import matplotlib.pyplot as plt
import time

FILE_PATH = "/home3/ivkhar/Baikal/data/normed/baikal_mc2020_multi_split_0924mid_eq_norm.h5"
THRESHOLD = 0.5

class BaikalTemporalGraphDataset(Dataset):
    def __init__(self, file_path, start_ev=0, num_events=1000, k=2):
        super().__init__()
        self.file_path = file_path
        self.start_ev = start_ev
        self.num_events = num_events
        self.k = k
        with h5py.File(self.file_path, 'r') as f:
            self.starts = f['train/ev_starts/data'][start_ev : start_ev + num_events + 1]
            
    def len(self): return self.num_events

    def get(self, idx):
        with h5py.File(self.file_path, 'r') as f:
            start, end = self.starts[idx], self.starts[idx + 1]
            x = torch.from_numpy(f['train/data/data'][start:end]).float()
            y = torch.from_numpy((f['train/labels/data'][start:end] != 0).astype(np.int64))
            
            num_nodes = x.size(0)
            if num_nodes <= 1:
                edge_index = torch.zeros((2, 0), dtype=torch.long)
            else:
                indices = np.arange(num_nodes)
                mask = (np.abs(indices[:, None] - indices) <= self.k) & (indices[:, None] != indices)
                edge_index = torch.from_numpy(np.array(np.where(mask))).long()
            
            return Data(x=x, edge_index=edge_index, y=y)

class GCN_Final(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.gelu(self.conv1(x, edge_index))
        x = F.gelu(self.conv2(x, edge_index))
        return self.conv3(x, edge_index)

def get_precision_at_recall(labels, probs, target_recall=0.9):
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    if np.max(recall) < target_recall: return 0.0
    return np.interp(target_recall, recall[::-1], precision[::-1])

def evaluate(model, loader, device, criterion):
    model.eval()
    all_probs, all_labels = [], []
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index)
            total_loss += criterion(out, batch.y).item()
            probs = F.softmax(out, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
            
    avg_loss = total_loss / len(loader)
    preds = (np.array(all_probs) > THRESHOLD).astype(int)
    prec = precision_score(all_labels, preds, zero_division=0)
    rec = recall_score(all_labels, preds, zero_division=0)
    p_at_r9 = get_precision_at_recall(all_labels, all_probs, 0.9)
    return avg_loss, prec, rec, p_at_r9

def train(num_train=500000, num_val=50000, epochs=80, batch_size=512):
    out_dir = "2026-03-07_investigation_of_2026-02-28_results"
    os.makedirs(os.path.join(out_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'checkpoints'), exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader = DataLoader(BaikalTemporalGraphDataset(FILE_PATH, 0, num_train, k=2), batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(BaikalTemporalGraphDataset(FILE_PATH, num_train, num_val, k=2), batch_size=batch_size, num_workers=4)
    
    model = GCN_Final(5, 128, 2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    weights = torch.tensor([1.0, 1.0]).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    
    history = {'t_loss': [], 'v_loss': [], 'v_prec': [], 'v_rec': [], 'v_p9': []}
    
    print(f"Starting LARGE training: {num_train} events, {epochs} epochs, k=2", flush=True)
    for epoch in range(1, epochs + 1):
        model.train()
        t_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch.x, batch.edge_index), batch.y)
            loss.backward(); optimizer.step()
            t_loss += loss.item()
            
        v_loss, v_prec, v_rec, v_p9 = evaluate(model, val_loader, device, criterion)
        history['t_loss'].append(t_loss/len(train_loader))
        history['v_loss'].append(v_loss); history['v_prec'].append(v_prec)
        history['v_rec'].append(v_rec); history['v_p9'].append(v_p9)
        
        print(f"Epoch {epoch:03d} | T-Loss: {t_loss/len(train_loader):.4f} | V-Loss: {v_loss:.4f} | Prec: {v_prec:.4f} | Rec: {v_rec:.4f} | P@R0.9: {v_p9:.4f}", flush=True)
        torch.save(model.state_dict(), os.path.join(out_dir, 'checkpoints', 'model_large_v1.pt'))

    plt.figure(figsize=(15, 5))
    plt.subplot(1,3,1); plt.plot(history['t_loss'], label='T'); plt.plot(history['v_loss'], label='V'); plt.title('Loss'); plt.legend()
    plt.subplot(1,3,2); plt.plot(history['v_prec'], label='P'); plt.plot(history['v_rec'], label='R'); plt.title('P/R'); plt.legend()
    plt.subplot(1,3,3); plt.plot(history['v_p9']); plt.title('P@R0.9')
    plt.savefig(os.path.join(out_dir, 'plots', 'metrics_large_v1.png'))

if __name__ == "__main__":
    train()
