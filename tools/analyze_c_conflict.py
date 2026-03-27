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

DATA_DIR = "data_processed"
CHECKPOINT = "2026-03-25_learnable_c_water/checkpoints/model_learn_c_1000.pt"
BATCH_SIZE = 512

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
    def forward(self, batch, c_water_val):
        x, edge_index, ptr = batch.x, batch.edge_index, batch.ptr
        first = ptr[:-1]; sizes = ptr[1:] - ptr[:-1]
        t0, x0, y0, z0 = [torch.repeat_interleave(x[first, i], sizes) for i in [1,2,3,4]]
        dt, dx, dy, dz = x[:,1]-t0, x[:,2]-x0, x[:,3]-y0, x[:,4]-z0
        dr2 = dx**2 + dy**2 + dz**2; dr = torch.sqrt(dr2 + 1e-8)
        s2 = (c_water_val * dt)**2 - dr2
        r = torch.sqrt(x[:, 2]**2 + x[:, 3]**2 + 1e-8)
        phi = torch.atan2(x[:, 3], x[:, 2])
        rho = torch.sqrt(x[:, 2]**2 + x[:, 3]**2 + x[:, 4]**2 + 1e-8)
        cosT = x[:, 4] / (rho + 1e-8)
        tof = dt - dr / (c_water_val + 1e-8)
        ext = torch.stack([s2, dt, dr, r, phi, rho, cosT, tof], dim=1)
        x_ext = torch.cat([x, ext], dim=1)
        h1 = F.gelu(self.conv1(x_ext, edge_index) + self.proj1(x_ext))
        h2 = F.gelu(self.conv2(h1, edge_index) + self.proj2(h1))
        h3 = F.gelu(self.conv3(h2, edge_index) + self.proj3(h2))
        h4 = F.gelu(self.conv4(h3, edge_index) + self.proj4(h3))
        h5 = F.gelu(self.conv5(h4, edge_index))
        combined = torch.cat([h1, h2, h3, h4, h5], dim=1)
        return self.head(combined)

def run_extended_sweep():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = JKResGCN_LearnableC(13, 2).to(device)
    model.load_state_dict(torch.load(CHECKPOINT, weights_only=True))
    model.eval()
    
    val_files = sorted(glob.glob(os.path.join(DATA_DIR, 'val', "chunk_*.pt")))
    c_range = np.linspace(0.2200, 0.2300, 21) # Wide range
    p9_results, loss_results = [], []
    
    print(f"Starting Wide Loss/Metric Sweep...")
    criterion = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for c in c_range:
            total_loss = 0
            all_labels, all_probs = [], []
            for f in val_files:
                data_list = torch.load(f, weights_only=False)
                loader = DataLoader(data_list, batch_size=BATCH_SIZE)
                for batch in loader:
                    batch = batch.to(device)
                    out = model(batch, c)
                    loss = criterion(out, batch.y)
                    total_loss += loss.item()
                    probs = F.softmax(out, dim=1)[:, 1]
                    all_probs.extend(probs.cpu().numpy()); all_labels.extend(batch.y.cpu().numpy())
            
            p, r, _ = precision_recall_curve(all_labels, all_probs)
            p9 = np.interp(0.9, r[::-1], p[::-1])
            avg_loss = total_loss / (len(val_files) * (10000 // BATCH_SIZE))
            
            p9_results.append(p9); loss_results.append(avg_loss)
            print(f"C={c:.5f} | P@R0.9: {p9:.4f} | Loss: {avg_loss:.6f}")
    
    # Dual axis plot
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    color = 'tab:green'
    ax1.set_xlabel('C_Water (m/ns)')
    ax1.set_ylabel('P@R0.9', color=color)
    ax1.plot(c_range, p9_results, color=color, marker='o', label='P@R0.9')
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Validation Loss', color=color)
    ax2.plot(c_range, loss_results, color=color, marker='x', linestyle='--', label='Loss')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.axvline(x=0.226418, color='black', alpha=0.5, label='Learned C')
    plt.title('Physics Conflict: Quality Peak vs Loss Minimum')
    fig.tight_layout()
    plt.savefig('2026-03-25_learnable_c_water/plots/c_water_conflict_analysis.png')
    print("\nConflict analysis plot saved.")

if __name__ == "__main__":
    run_extended_sweep()
