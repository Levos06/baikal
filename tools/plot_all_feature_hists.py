import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
import glob
import os
import sys

# Add experiment folder to path for import
sys.path.append('2026-03-28_extended_21features')
from train_21feat import add_features_21_norm

DATA_DIR = "data_processed_50k/val"
OUT_DIR = "2026-03-28_extended_21features/plots/feature_distributions"
os.makedirs(OUT_DIR, exist_ok=True)

FEATURE_NAMES = [
    "0_Charge", "1_Time", "2_X", "3_Y", "4_Z",
    "5_Minkowski_s2", "6_dt", "7_dr", "8_r_xy", "9_phi",
    "10_rho", "11_cosTheta", "12_ToF_Res", "13_NeighDist",
    "14_NeighQ", "15_Q_Mean", "16_cosAlpha", "17_StrHits",
    "18_StrZSpan", "19_EventNhits", "20_Duration"
]

def generate_histograms():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.pt")))[:10]
    all_features = []
    
    print("Collecting data for histograms...")
    for f in files:
        data_list = torch.load(f, weights_only=False)
        loader = DataLoader(data_list, batch_size=512)
        for batch in loader:
            batch = add_features_21_norm(batch.to(device))
            all_features.append(batch.x.cpu().numpy())
            
    data = np.concatenate(all_features, axis=0)
    
    print(f"Plotting {len(FEATURE_NAMES)} histograms...")
    for i in range(len(FEATURE_NAMES)):
        plt.figure(figsize=(10, 6))
        plt.hist(data[:, i], bins=100, color='skyblue', edgecolor='black', alpha=0.7)
        plt.title(f"Distribution of Feature {FEATURE_NAMES[i]}")
        plt.xlabel("Normalized Value (Z-score)")
        plt.ylabel("Frequency")
        plt.grid(True, linestyle='--', alpha=0.6)
        
        plt.savefig(os.path.join(OUT_DIR, f"{FEATURE_NAMES[i]}.png"))
        plt.close()
        if (i+1) % 5 == 0: print(f"  Processed {i+1}/21")

if __name__ == "__main__":
    generate_histograms()
