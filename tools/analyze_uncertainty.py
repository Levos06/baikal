import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt
import os

FILE_PATH = "/home3/ivkhar/Baikal/data/normed/baikal_mc2020_multi_split_0924mid_eq_norm.h5"
CHECKPOINT_PATH = "2026-03-08_deep_wide_model/checkpoints/model_deep_wide_v3.pt"

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

class GCN_V3(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 512)
        self.conv2 = GCNConv(512, 768)
        self.conv3 = GCNConv(768, 512)
        self.conv4 = GCNConv(512, out_channels)
    def forward(self, x, edge_index):
        x = F.gelu(self.conv1(x, edge_index))
        x = F.gelu(self.conv2(x, edge_index))
        x = F.gelu(self.conv3(x, edge_index))
        return self.conv4(x, edge_index)

def get_probs(model, loader, device):
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index)
            probs = F.softmax(out, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(batch.y.cpu().numpy())
    return np.array(all_probs), np.array(all_labels)

def analyze():
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(f"Analyzing on {device}...")
    
    model = GCN_V3(5, 2).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    
    # Берем небольшие выборки для быстроты
    train_loader = DataLoader(BaikalTemporalGraphDataset(FILE_PATH, 0, 5000, k=2), batch_size=512, num_workers=4)
    val_loader = DataLoader(BaikalTemporalGraphDataset(FILE_PATH, 250000, 5000, k=2), batch_size=512, num_workers=4)
    
    print("Collecting probabilities...")
    train_probs, train_labels = get_probs(model, train_loader, device)
    val_probs, val_labels = get_probs(model, val_loader, device)
    
    plt.figure(figsize=(12, 6))
    
    # Гистограмма для Train
    plt.subplot(1, 2, 1)
    plt.hist(train_probs[train_labels == 0], bins=50, alpha=0.5, label='Background (0)', color='blue', log=True)
    plt.hist(train_probs[train_labels == 1], bins=50, alpha=0.5, label='Signal (1)', color='red', log=True)
    plt.axvline(0.5, color='black', linestyle='--')
    plt.title("Train Probability Distribution")
    plt.xlabel("Probability of Signal")
    plt.legend()
    
    # Гистограмма для Val
    plt.subplot(1, 2, 2)
    plt.hist(val_probs[val_labels == 0], bins=50, alpha=0.5, label='Background (0)', color='blue', log=True)
    plt.hist(val_probs[val_labels == 1], bins=50, alpha=0.5, label='Signal (1)', color='red', log=True)
    plt.axvline(0.5, color='black', linestyle='--')
    plt.title("Val Probability Distribution")
    plt.xlabel("Probability of Signal")
    plt.legend()
    
    plt.tight_layout()
    output_path = "2026-03-08_deep_wide_model/plots/uncertainty_analysis.png"
    plt.savefig(output_path)
    print(f"Analysis plot saved to {output_path}")

if __name__ == "__main__":
    analyze()
