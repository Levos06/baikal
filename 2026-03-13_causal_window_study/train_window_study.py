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
V_WATER = 0.225
D_MAX = 100.0

# Stats for normalization
T_RES_MEAN, T_RES_STD = 518.25, 1560.43
R_MEAN, R_STD = 1.34, 0.45
PHI_MEAN, PHI_STD = -0.31, 1.79
DEG_MEAN, DEG_STD = 3.0, 1.0
RHO_MEAN, RHO_STD = 1.69, 0.40
COS_MEAN, COS_STD = 0.03, 0.57
TOF_MEAN, TOF_STD = 9.82, 2.40
NEIGHQ_MEAN, NEIGHQ_STD = 0.09, 7.85

class BaikalStudyDataset(Dataset):
    def __init__(self, file_path, start_ev=0, num_events=1000, t_cut=3.0):
        super().__init__()
        self.file_path = file_path
        self.start_ev = start_ev
        self.num_events = num_events
        self.t_cut = t_cut
        with h5py.File(self.file_path, 'r') as f:
            self.starts = f['train/ev_starts/data'][start_ev : start_ev + num_events + 1]
            
    def len(self): return self.num_events

    def get(self, idx):
        with h5py.File(self.file_path, 'r') as f:
            start, end = self.starts[idx], self.starts[idx + 1]
            x_main = f['train/data/data'][start:end]
            q, t, x_pos, y_pos, z_pos = x_main[:,0], x_main[:,1], x_main[:,2], x_main[:,3], x_main[:,4]
            x_tres = (f['train/t_res/data'][start:end] - T_RES_MEAN) / T_RES_STD
            r_cyl = np.sqrt(x_pos**2 + y_pos**2)
            x_r = (r_cyl - R_MEAN) / R_STD
            x_phi = (np.arctan2(y_pos, x_pos) - PHI_MEAN) / PHI_STD
            rho = np.sqrt(x_pos**2 + y_pos**2 + z_pos**2)
            x_rho = (rho - RHO_MEAN) / RHO_STD
            x_cos = (np.divide(z_pos, rho, out=np.zeros_like(z_pos), where=rho!=0) - COS_MEAN) / COS_STD
            
            num_nodes = x_main.shape[0]
            if num_nodes <= 1:
                edge_index = torch.zeros((2, 0), dtype=torch.long)
                x_deg, x_tof, x_nq = np.zeros(num_nodes), np.zeros(num_nodes), np.zeros(num_nodes)
            else:
                pos = x_main[:, 2:5]
                dist = np.sqrt(np.sum((pos[:, None, :] - pos[None, :, :])**2, axis=2))
                dt = np.abs(t[:, None] - t[None, :])
                causal_mask = (np.abs(dt - dist/V_WATER) < self.t_cut) & (dist < D_MAX) & (dist > 0)
                rows, cols = np.where(causal_mask)
                edge_index = torch.from_numpy(np.array([rows, cols])).long()
                counts = np.bincount(rows, minlength=num_nodes)
                x_deg = (counts - DEG_MEAN) / DEG_STD
                x_tof, x_nq = np.zeros(num_nodes), np.zeros(num_nodes)
                if len(rows) > 0:
                    tres_causal = np.abs(dt[rows, cols] - dist[rows, cols]/V_WATER)
                    for node in range(num_nodes):
                        node_mask = rows == node
                        if np.any(node_mask):
                            x_tof[node] = np.mean(tres_causal[node_mask])
                            x_nq[node] = np.sum(q[cols[node_mask]])
                x_tof = (x_tof - TOF_MEAN) / TOF_STD
                x_nq = (x_nq - NEIGHQ_MEAN) / NEIGHQ_STD

            x_combined = np.column_stack([x_main, x_tres, x_r, x_phi, x_deg, x_rho, x_cos, x_tof, x_nq])
            x_combined = torch.from_numpy(x_combined).float()
            y = torch.from_numpy((f['train/labels/data'][start:end] != 0).astype(np.int64))
            return Data(x=x_combined, edge_index=edge_index, y=y)

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

def evaluate(model, loader, device, criterion):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index)
            probs = F.softmax(out, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy()); all_labels.extend(batch.y.cpu().numpy())
    
    precision, recall, thresholds = precision_recall_curve(all_labels, all_probs)
    if np.max(recall) < 0.9: p_at_r9 = 0.0
    else: p_at_r9 = np.interp(0.9, recall[::-1], precision[::-1])
    return p_at_r9

def run_study():
    t_cuts = [3, 6, 9, 12, 15, 18, 21, 24]
    results = []
    num_train, num_val = 100000, 15000
    epochs = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for t_cut in t_cuts:
        print(f"\n--- Testing T_CUT = {t_cut} ns ---")
        train_loader = DataLoader(BaikalStudyDataset(FILE_PATH, 0, num_train, t_cut=t_cut), batch_size=256, shuffle=True, num_workers=4)
        val_loader = DataLoader(BaikalStudyDataset(FILE_PATH, num_train, num_val, t_cut=t_cut), batch_size=256, num_workers=4)
        
        model = GCN_V3(13, 2).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
        criterion = torch.nn.CrossEntropyLoss()
        
        for epoch in range(1, epochs + 1):
            model.train()
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                loss = criterion(model(batch.x, batch.edge_index), batch.y)
                loss.backward(); optimizer.step()
            print(f"Epoch {epoch} completed", flush=True)
            
        final_p9 = evaluate(model, val_loader, device, criterion)
        print(f"Final V-P@R0.9 for T_CUT={t_cut}: {final_p9:.4f}")
        results.append(final_p9)
        
        # Save intermediate plot
        plt.figure(figsize=(10, 6))
        plt.plot(t_cuts[:len(results)], results, marker='o')
        plt.title("P@R0.9 vs Causal Window T_CUT")
        plt.xlabel("T_CUT (ns)")
        plt.ylabel("Validation P@R0.9")
        plt.grid(True)
        plt.savefig("2026-03-13_causal_window_study/plots/study_progress.png")
        plt.close()

    print("\nStudy complete!")
    print("T_CUTs:", t_cuts)
    print("Results:", results)

if __name__ == "__main__":
    run_study()
