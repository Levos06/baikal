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
# Final learned C_Water from logs
LEARNED_C = 0.226418 
CHECKPOINT = "2026-03-25_learnable_c_water/checkpoints/model_learn_c_1000.pt"
BATCH_SIZE = 256

class JKResGCN_LearnableC(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.tensor(0.0)) 
        self.conv1 = GCNConv(in_channels, 512); self.conv2 = GCNConv(512, 512); self.conv3 = GCNConv(512, 512)
        self.conv4 = GCNConv(512, 512); self.conv5 = GCNConv(512, 512)
        self.proj1 = torch.nn.Linear(in_channels, 512); self.proj2 = torch.nn.Linear(512, 512)
        self.proj3 = torch.nn.Linear(512, 512); self.proj4 = torch.nn.Linear(512, 512)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(512 * 5, 1024), torch.nn.GELU(), torch.nn.Dropout(0.1),
            torch.nn.Linear(1024, 512), torch.nn.GELU(), torch.nn.Dropout(0.1),
            torch.nn.Linear(512, out_channels)
        )
    def forward(self, batch):
        # Use the fixed learned C for evaluation
        c_water = LEARNED_C
        x, edge_index, ptr = batch.x, batch.edge_index, batch.ptr
        
        # Internal preprocessing (must match training exactly)
        first = ptr[:-1]; sizes = ptr[1:] - ptr[:-1]
        t0, x0, y0, z0 = [torch.repeat_interleave(x[first, i], sizes) for i in [1,2,3,4]]
        dt, dx, dy, dz = x[:,1]-t0, x[:,2]-x0, x[:,3]-y0, x[:,4]-z0
        dr2 = dx**2 + dy**2 + dz**2; dr = torch.sqrt(dr2 + 1e-8)
        s2 = (c_water * dt)**2 - dr2
        r = torch.sqrt(x[:, 2]**2 + x[:, 3]**2 + 1e-8)
        phi = torch.atan2(x[:, 3], x[:, 2])
        rho = torch.sqrt(x[:, 2]**2 + x[:, 3]**2 + x[:, 4]**2 + 1e-8)
        cosT = x[:, 4] / (rho + 1e-8)
        tof = dt - dr / (c_water + 1e-8)
        ext = torch.stack([s2, dt, dr, r, phi, rho, cosT, tof], dim=1)
        x_ext = torch.cat([x, ext], dim=1)
        
        h1 = F.gelu(self.conv1(x_ext, edge_index) + self.proj1(x_ext))
        h2 = F.gelu(self.conv2(h1, edge_index) + self.proj2(h1))
        h3 = F.gelu(self.conv3(h2, edge_index) + self.proj3(h2))
        h4 = F.gelu(self.conv4(h3, edge_index) + self.proj4(h3))
        h5 = F.gelu(self.conv5(h4, edge_index))
        combined = torch.cat([h1, h2, h3, h4, h5], dim=1)
        return self.head(combined)

def get_p_at_r(labels, probs, target=0.9):
    p, r, _ = precision_recall_curve(labels, probs)
    return np.interp(target, r[::-1], p[::-1])

def run_evaluation():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = JKResGCN_LearnableC(13, 2).to(device)
    model.load_state_dict(torch.load(CHECKPOINT, weights_only=True))
    model.eval()
    
    val_files = sorted(glob.glob(os.path.join(DATA_DIR, 'val', "chunk_*.pt")))
    all_labels, all_probs = [], []
    
    print(f"Evaluating Learnable C model ({CHECKPOINT})")
    print(f"Using fixed learned C_Water: {LEARNED_C}")
    
    with torch.no_grad():
        for f in val_files:
            data_list = torch.load(f, weights_only=False)
            loader = DataLoader(data_list, batch_size=BATCH_SIZE)
            for batch in loader:
                batch = batch.to(device)
                out = model(batch)
                probs = F.softmax(out, dim=1)[:, 1]
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(batch.y.cpu().numpy())
    
    all_labels, all_probs = np.array(all_labels), np.array(all_probs)
    p9 = get_p_at_r(all_labels, all_probs, 0.9)
    preds = (all_probs > 0.5).astype(int)
    
    print("\n" + "="*30)
    print("LEARNABLE C FINAL EVALUATION")
    print("="*30)
    print(f"Precision: {precision_score(all_labels, preds):.4f}")
    print(f"Recall:    {recall_score(all_labels, preds):.4f}")
    print(f"P@R0.9:    {p9:.4f}")
    print("="*30)

if __name__ == "__main__":
    run_evaluation()
