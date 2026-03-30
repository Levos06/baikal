import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.utils import scatter
from torch_geometric.loader import DataLoader
import glob
import os

DATA_DIR = "data_processed_50k/val"
C_WATER = 0.225
# Load our global stats
STATS = torch.load('tools/feat_stats_21.pt', weights_only=True)
M_MINK = STATS['means'][0] # Minkowski is first in the extra set (F5)
S_MINK = STATS['stds'][0]

def get_raw_minkowski(batch):
    x, ptr = batch.x, batch.ptr
    first = ptr[:-1]; sizes = ptr[1:] - ptr[:-1]
    t0, x0, y0, z0 = [torch.repeat_interleave(x[first, i], sizes) for i in [1,2,3,4]]
    dt, dx, dy, dz = x[:,1]-t0, x[:,2]-x0, x[:,3]-y0, x[:,4]-z0
    dr2 = dx**2 + dy**2 + dz**2
    s2 = (C_WATER * dt)**2 - dr2
    return s2

def analyze_minkowski():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.pt")))[:5]
    raw_vals = []
    
    print(f"Global Stats from file: Mean={M_MINK:.4f}, Std={S_MINK:.4f}")
    
    for f in files:
        data_list = torch.load(f, weights_only=False)
        loader = DataLoader(data_list, batch_size=512)
        for batch in loader:
            raw_vals.append(get_raw_minkowski(batch).numpy())
            
    raw = np.concatenate(raw_vals)
    norm = (raw - M_MINK.item()) / (S_MINK.item() + 1e-8)
    
    print("\n--- Current Batch Distribution ---")
    print(f"Raw:  Mean={np.mean(raw):.4f}, Std={np.std(raw):.4f}, Range=[{np.min(raw):.2f}, {np.max(raw):.2f}]")
    print(f"Norm: Mean={np.mean(norm):.4f}, Std={np.std(norm):.4f}, Range=[{np.min(norm):.2f}, {np.max(norm):.2f}]")
    
    # Plotting to see if distribution is "killed"
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(raw, bins=100, color='blue', alpha=0.7)
    plt.title("Raw Minkowski Distribution")
    plt.xlabel("s^2")
    
    plt.subplot(1, 2, 2)
    plt.hist(norm, bins=100, color='green', alpha=0.7)
    plt.title("Normalized Minkowski Distribution")
    plt.xlabel("z-score")
    
    plt.tight_layout()
    plt.savefig('2026-03-28_extended_21features/plots/minkowski_audit.png')
    print("\nAudit plot saved to 2026-03-28_extended_21features/plots/minkowski_audit.png")

if __name__ == "__main__":
    analyze_minkowski()
