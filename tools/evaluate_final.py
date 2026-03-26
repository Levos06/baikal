import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from sklearn.metrics import precision_score, recall_score, precision_recall_curve
import os

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

class GCN_DeepWide(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 512)
        self.conv2 = GCNConv(512, 2048)
        self.conv3 = GCNConv(2048, 512)
        self.conv4 = GCNConv(512, out_channels)
    def forward(self, x, edge_index):
        x = F.gelu(self.conv1(x, edge_index))
        x = F.gelu(self.conv2(x, edge_index))
        x = F.gelu(self.conv3(x, edge_index))
        return self.conv4(x, edge_index)

class GCN_Capacity(torch.nn.Module):
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

def evaluate(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index)
            probs = F.softmax(out, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
    preds = (np.array(all_probs) > THRESHOLD).astype(int)
    prec = precision_score(all_labels, preds, zero_division=0)
    rec = recall_score(all_labels, preds, zero_division=0)
    p_at_r9 = get_precision_at_recall(all_labels, all_probs, 0.9)
    return prec, rec, p_at_r9

def run_eval():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_train = 150000
    num_val = 20000
    
    print("Evaluating Deep & Wide Model...")
    model_deep = GCN_DeepWide(5, 2).to(device)
    model_deep.load_state_dict(torch.load("2026-03-08_deep_wide_model/checkpoints/model_deep_wide.pt"))
    val_loader_k2 = DataLoader(BaikalTemporalGraphDataset(FILE_PATH, num_train, num_val, k=2), batch_size=512, num_workers=4)
    p, r, p9 = evaluate(model_deep, val_loader_k2, device)
    print(f"Deep & Wide | Prec: {p:.4f} | Rec: {r:.4f} | P@R0.9: {p9:.4f}")

    print("\nEvaluating k=4 Long Model...")
    model_k4 = GCN_Capacity(5, 512, 2).to(device)
    model_k4.load_state_dict(torch.load("2026-03-08_k4_long_training/checkpoints/model_k4_long.pt"))
    val_loader_k4 = DataLoader(BaikalTemporalGraphDataset(FILE_PATH, num_train, num_val, k=4), batch_size=512, num_workers=4)
    p, r, p9 = evaluate(model_k4, val_loader_k4, device)
    print(f"k=4 Long | Prec: {p:.4f} | Rec: {r:.4f} | P@R0.9: {p9:.4f}")

if __name__ == "__main__":
    run_eval()
