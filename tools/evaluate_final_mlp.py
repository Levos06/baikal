import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import os
import glob
import numpy as np
from sklearn.metrics import precision_score, recall_score, precision_recall_curve

DATA_DIR = "data_processed"
CHECKPOINT = "2026-03-25_full_mlp_training_1000ep/checkpoints/model_full_mlp_1000.pt"
BATCH_SIZE = 256
C_WATER = 0.225

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

# EXACT COPY FROM train_full_mlp.py
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

def get_p_at_r(labels, probs, target=0.9):
    p, r, _ = precision_recall_curve(labels, probs)
    return np.interp(target, r[::-1], p[::-1])

def run_evaluation():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = JKResGCN_v2(13, 2).to(device)
    model.load_state_dict(torch.load(CHECKPOINT, weights_only=True))
    model.eval()
    
    val_files = sorted(glob.glob(os.path.join(DATA_DIR, 'val', "chunk_*.pt")))
    all_labels, all_probs = [], []
    
    print(f"Evaluating {CHECKPOINT} on ALL validation data...")
    with torch.no_grad():
        for f in val_files:
            data_list = torch.load(f, weights_only=False)
            loader = DataLoader(data_list, batch_size=BATCH_SIZE)
            for batch in loader:
                batch = add_extended_features_vectorized(batch.to(device))
                out = model(batch.x, batch.edge_index)
                probs = F.softmax(out, dim=1)[:, 1]
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(batch.y.cpu().numpy())
    
    all_labels, all_probs = np.array(all_labels), np.array(all_probs)
    p9 = get_p_at_r(all_labels, all_probs, 0.9)
    preds = (all_probs > 0.5).astype(int)
    
    print("\n" + "="*30)
    print("FINAL CORRECTED EVALUATION")
    print("="*30)
    print(f"Precision: {precision_score(all_labels, preds):.4f}")
    print(f"Recall:    {recall_score(all_labels, preds):.4f}")
    print(f"P@R0.9:    {p9:.4f}")
    print("="*30)

if __name__ == "__main__":
    run_evaluation()
