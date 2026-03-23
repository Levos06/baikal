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
T_CUT = 1.0     # Asymmetric half-window
D_MAX = 100.0

# Stats for normalization
R_MEAN, R_STD = 1.34, 0.45
PHI_MEAN, PHI_STD = -0.31, 1.79
DEG_MEAN, DEG_STD = 3.0, 1.0
RHO_MEAN, RHO_STD = 1.69, 0.40
COS_MEAN, COS_STD = 0.03, 0.57
TOF_MEAN, TOF_STD = 9.82, 2.40
NEIGHQ_MEAN, NEIGHQ_STD = 0.09, 7.85

class BaikalNoTresDataset(Dataset):
    def __init__(self, file_path, start_ev=0, num_events=1000, t_cut=1.0):
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
            
            # Features (12 total, NO t_res)
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
                dt = t[None, :] - t[:, None]
                
                # Asymmetric Causality
                causal_mask = (dt - dist/V_WATER > 0) & (dt - dist/V_WATER < 2*self.t_cut) & (dist < D_MAX)
                rows, cols = np.where(causal_mask)
                edge_index = torch.from_numpy(np.array([rows, cols])).long()
                
                counts = np.bincount(rows, minlength=num_nodes)
                x_deg = (counts - DEG_MEAN) / DEG_STD
                x_tof, x_nq = np.zeros(num_nodes), np.zeros(num_nodes)
                if len(rows) > 0:
                    tres_causal = dt[rows, cols] - dist[rows, cols]/V_WATER
                    for node in range(num_nodes):
                        node_mask = rows == node
                        if np.any(node_mask):
                            x_tof[node] = np.mean(tres_causal[node_mask])
                            x_nq[node] = np.sum(q[cols[node_mask]])
                x_tof = (x_tof - TOF_MEAN) / TOF_STD
                x_nq = (x_nq - NEIGHQ_MEAN) / NEIGHQ_STD

            # Combined 12 features
            x_combined = np.column_stack([x_main, x_r, x_phi, x_deg, x_rho, x_cos, x_tof, x_nq])
            x_combined = torch.from_numpy(x_combined).float()
            y = torch.from_numpy((f['train/labels/data'][start:end] != 0).astype(np.int64))
            return Data(x=x_combined, edge_index=edge_index, y=y)

class GCN_12(torch.nn.Module):
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

def get_precision_at_recall(labels, probs, target_recall=0.9):
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    if np.max(recall) < 0.9: return 0.0
    return np.interp(0.9, recall[::-1], precision[::-1])

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
    avg_loss = total_loss / len(loader)
    preds = (np.array(all_probs) > THRESHOLD).astype(int)
    prec = precision_score(all_labels, preds, zero_division=0)
    rec = recall_score(all_labels, preds, zero_division=0)
    p_at_r9 = get_precision_at_recall(all_labels, all_probs, 0.9)
    return avg_loss, prec, rec, p_at_r9

def train(num_train=100000, num_val=15000, epochs=100, batch_size=256):
    project_dir = "2026-03-14_asymmetric_no_tres"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} | Experiment: NO T_RES (12 feat) | Window: {T_CUT}ns")
    
    train_loader = DataLoader(BaikalNoTresDataset(FILE_PATH, 0, num_train, t_cut=T_CUT), batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(BaikalNoTresDataset(FILE_PATH, num_train, num_val, t_cut=T_CUT), batch_size=batch_size, num_workers=4)
    train_eval_loader = DataLoader(BaikalNoTresDataset(FILE_PATH, 0, num_val, t_cut=T_CUT), batch_size=batch_size, num_workers=4)

    model = GCN_12(12, 2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch.x, batch.edge_index), batch.y)
            loss.backward(); optimizer.step()
            
        t_loss, t_prec, t_rec, t_p9 = evaluate(model, train_eval_loader, device, criterion)
        v_loss, v_prec, v_rec, v_p9 = evaluate(model, val_loader, device, criterion)
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch:03d} | Time: {epoch_time:.1f}s | T-Loss: {t_loss:.4f} | V-Loss: {v_loss:.4f} | T-P@R0.9: {t_p9:.4f} | V-P@R0.9: {v_p9:.4f}", flush=True)
        
    torch.save(model.state_dict(), os.path.join(project_dir, "checkpoints/model_no_tres.pt"))

if __name__ == "__main__":
    train()
