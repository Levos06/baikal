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

class BaikalFullDataset(Dataset):
    def __init__(self, file_path, start_ev=0, num_events=1000, k=2):
        super().__init__()
        self.file_path = file_path
        self.k = k
        with h5py.File(self.file_path, 'r') as f:
            self.starts = f['train/ev_starts/data'][start_ev : start_ev + num_events + 1]
            
    def len(self): return len(self.starts) - 1

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

class GCN_5Layer(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 512)
        self.conv2 = GCNConv(512, 1024)
        self.conv3 = GCNConv(1024, 1024)
        self.conv4 = GCNConv(1024, 512)
        self.conv5 = GCNConv(512, out_channels)

    def forward(self, x, edge_index):
        x = F.gelu(self.conv1(x, edge_index))
        x = F.gelu(self.conv2(x, edge_index))
        x = F.gelu(self.conv3(x, edge_index))
        x = F.gelu(self.conv4(x, edge_index))
        return self.conv5(x, edge_index)

def get_precision_at_recall(labels, probs, target_recall=0.9):
    precision, recall, _ = precision_recall_curve(labels, probs)
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
            all_probs.extend(probs.cpu().numpy()); all_labels.extend(batch.y.cpu().numpy())
    p_at_r9 = get_precision_at_recall(all_labels, all_probs, 0.9)
    return total_loss/len(loader), precision_score(all_labels, (np.array(all_probs)>THRESHOLD).astype(int), zero_division=0), recall_score(all_labels, (np.array(all_probs)>THRESHOLD).astype(int), zero_division=0), p_at_r9

def train():
    num_train, num_val = 18000000, 100000 # ОГРОМНЫЙ масштаб
    epochs = 150
    batch_size = 256
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} | FULL DATA COMPARISON | NO RES")
    
    train_loader = DataLoader(BaikalFullDataset(FILE_PATH, 0, num_train, k=2), batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(BaikalFullDataset(FILE_PATH, num_train, num_val, k=2), batch_size=batch_size, num_workers=8)
    train_eval_loader = DataLoader(BaikalFullDataset(FILE_PATH, 0, num_val, k=2), batch_size=batch_size, num_workers=8)

    model = GCN_5Layer(5, 2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            criterion(model(batch.x, batch.edge_index), batch.y).backward(); optimizer.step()
            
        t_loss, t_prec, t_rec, t_p9 = evaluate(model, train_eval_loader, device, criterion)
        v_loss, v_prec, v_rec, v_p9 = evaluate(model, val_loader, device, criterion)
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch:03d} | Time: {epoch_time:.1f}s | T-Loss: {t_loss:.4f} | V-Loss: {v_loss:.4f} | T-Prec: {t_prec:.4f} | V-Prec: {v_prec:.4f} | T-Rec: {t_rec:.4f} | V-Rec: {v_rec:.4f} | T-P@R0.9: {t_p9:.4f} | V-P@R0.9: {v_p9:.4f}", flush=True)
        if epoch % 5 == 0: torch.save(model.state_dict(), f"2026-03-16_res_comparison/checkpoints/model_without_res.pt")

if __name__ == "__main__":
    train()
