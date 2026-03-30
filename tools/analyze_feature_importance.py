import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
import copy

# --- CONFIG ---
DATA_DIR = "data_processed"
CHECKPOINT = "2026-03-25_full_mlp_training_1000ep/checkpoints/model_full_mlp_1000.pt"
BATCH_SIZE = 512
C_WATER = 0.225

FEATURE_NAMES = [
    "0: Charge", "1: Time", "2: X", "3: Y", "4: Z",
    "5: Minkowski s^2", "6: dt", "7: dr", "8: r", "9: phi",
    "10: rho", "11: cosTheta", "12: ToF Residual"
]

# --- MODEL & UTILS ---
class JKResGCN_v2(torch.nn.Module):
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

def add_features(batch):
    x, ptr = batch.x, batch.ptr
    first = ptr[:-1]; sizes = ptr[1:] - ptr[:-1]
    t0, x0, y0, z0 = [torch.repeat_interleave(x[first, i], sizes) for i in [1,2,3,4]]
    dt, dx, dy, dz = x[:,1]-t0, x[:,2]-x0, x[:,3]-y0, x[:,4]-z0
    dr2 = dx**2 + dy**2 + dz**2; dr = torch.sqrt(dr2 + 1e-8)
    s2 = (C_WATER * dt)**2 - dr2
    r = torch.sqrt(x[:, 2]**2 + x[:, 3]**2 + 1e-8)
    phi = torch.atan2(x[:, 3], x[:, 2])
    rho = torch.sqrt(x[:, 2]**2 + x[:, 3]**2 + x[:, 4]**2 + 1e-8)
    cosT = x[:, 4] / (rho + 1e-8)
    tof = dt - dr/C_WATER
    ext = torch.stack([s2, dt, dr, r, phi, rho, cosT, tof], dim=1)
    batch.x = torch.cat([x, ext], dim=1)
    return batch

def get_p_at_r(labels, probs, target=0.9):
    p, r, _ = precision_recall_curve(labels, probs)
    return np.interp(target, r[::-1], p[::-1])

def run_evaluation(model, device, val_loader, shuffle_idx=None):
    all_labels, all_probs = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = add_features(batch.to(device))
            
            if shuffle_idx is not None:
                # Permute ONLY the target feature across all nodes in the batch
                x = batch.x.clone()
                perm = torch.randperm(x.size(0))
                x[:, shuffle_idx] = x[perm, shuffle_idx]
                out = model(x, batch.edge_index)
            else:
                out = model(batch.x, batch.edge_index)
                
            probs = F.softmax(out, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
            
    return get_p_at_r(np.array(all_labels), np.array(all_probs), 0.9)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = JKResGCN_v2(13, 2).to(device)
    model.load_state_dict(torch.load(CHECKPOINT, weights_only=True))
    model.eval()
    
    val_files = sorted(glob.glob(os.path.join(DATA_DIR, 'val', "chunk_*.pt")))
    # Use first 5 chunks for speed (50k events)
    data_list = []
    for f in val_files[:5]: data_list.extend(torch.load(f, weights_only=False))
    val_loader = DataLoader(data_list, batch_size=BATCH_SIZE)
    
    print(f"Analyzing feature importance for {CHECKPOINT}...")
    
    # 1. Baseline
    baseline_p9 = run_evaluation(model, device, val_loader)
    print(f"Baseline P@R0.9: {baseline_p9:.4f}")
    
    importances = []
    for i in range(13):
        p9 = run_evaluation(model, device, val_loader, shuffle_idx=i)
        drop = baseline_p9 - p9
        importances.append(drop)
        print(f"Feature {FEATURE_NAMES[i]:15s} | P@R0.9: {p9:.4f} | Drop: {drop:.4f}")
        
    # 2. Plotting
    plt.figure(figsize=(12, 8))
    indices = np.argsort(importances)
    plt.barh(range(len(indices)), [importances[i] for i in indices], color='skyblue')
    plt.yticks(range(len(indices)), [FEATURE_NAMES[i] for i in indices])
    plt.xlabel('Importance (Drop in P@R0.9)')
    plt.title(f'Feature Importance: {os.path.basename(CHECKPOINT)}')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('2026-03-25_full_mlp_training_1000ep/plots/feature_importance.png')
    print(f"\nAnalysis complete. Plot saved to 2026-03-25_full_mlp_training_1000ep/plots/feature_importance.png")

if __name__ == "__main__":
    main()
