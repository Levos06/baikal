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
V_WATER = 0.225 # Скорость света в воде, м/нс
T_CUT = 50.0    # Временное окно для причинности, нс
D_MAX = 100.0   # Максимальное расстояние между связанными хитами, м

# Статистики для нормализации (из предыдущего этапа)
T_RES_MEAN, T_RES_STD = 518.25, 1560.43
R_MEAN, R_STD = 1.34, 0.45
PHI_MEAN, PHI_STD = -0.31, 1.79
DEG_MEAN, DEG_STD = 3.0, 1.0
RHO_MEAN, RHO_STD = 1.69, 0.40
COS_MEAN, COS_STD = 0.03, 0.57
TOF_MEAN, TOF_STD = 9.82, 2.40
NEIGHQ_MEAN, NEIGHQ_STD = 0.09, 7.85

class BaikalCausalDataset(Dataset):
    def __init__(self, file_path, start_ev=0, num_events=1000):
        super().__init__()
        self.file_path = file_path
        self.start_ev = start_ev
        self.num_events = num_events
        with h5py.File(self.file_path, 'r') as f:
            self.starts = f['train/ev_starts/data'][start_ev : start_ev + num_events + 1]
            
    def len(self): return self.num_events

    def get(self, idx):
        with h5py.File(self.file_path, 'r') as f:
            start, end = self.starts[idx], self.starts[idx + 1]
            
            x_main = f['train/data/data'][start:end]
            q, t, x_pos, y_pos, z_pos = x_main[:,0], x_main[:,1], x_main[:,2], x_main[:,3], x_main[:,4]
            
            # Признаки (13 штук)
            x_tres = (f['train/t_res/data'][start:end] - T_RES_MEAN) / T_RES_STD
            r_cyl = np.sqrt(x_pos**2 + y_pos**2)
            x_r = (r_cyl - R_MEAN) / R_STD
            x_phi = (np.arctan2(y_pos, x_pos) - PHI_MEAN) / PHI_STD
            rho = np.sqrt(x_pos**2 + y_pos**2 + z_pos**2)
            x_rho = (rho - RHO_MEAN) / RHO_STD
            x_cos = (np.divide(z_pos, rho, out=np.zeros_like(z_pos), where=rho!=0) - COS_MEAN) / COS_STD
            
            # Построение ПРИЧИННОГО графа
            num_nodes = x_main.shape[0]
            if num_nodes <= 1:
                edge_index = torch.zeros((2, 0), dtype=torch.long)
                x_deg, x_tof, x_nq = np.zeros(num_nodes), np.zeros(num_nodes), np.zeros(num_nodes)
            else:
                pos = x_main[:, 2:5]
                # Матрица расстояний и временных разностей
                dist = np.sqrt(np.sum((pos[:, None, :] - pos[None, :, :])**2, axis=2))
                dt = np.abs(t[:, None] - t[None, :])
                
                # Условие причинности: |dt - dr/v| < T_CUT
                causal_mask = (np.abs(dt - dist/V_WATER) < T_CUT) & (dist < D_MAX) & (dist > 0)
                rows, cols = np.where(causal_mask)
                edge_index = torch.from_numpy(np.array([rows, cols])).long()
                
                # Графовые признаки на основе новой топологии
                counts = np.bincount(rows, minlength=num_nodes)
                x_deg = (counts - DEG_MEAN) / DEG_STD
                
                x_tof = np.zeros(num_nodes)
                x_nq = np.zeros(num_nodes)
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

class GCN_Causal(torch.nn.Module):
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
    if np.max(recall) < target_recall: return 0.0
    return np.interp(target_recall, recall[::-1], precision[::-1])

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
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
    avg_loss = total_loss / len(loader)
    preds = (np.array(all_probs) > THRESHOLD).astype(int)
    prec = precision_score(all_labels, preds, zero_division=0)
    rec = recall_score(all_labels, preds, zero_division=0)
    p_at_r9 = get_precision_at_recall(all_labels, all_probs, 0.9)
    return avg_loss, prec, rec, p_at_r9

def train(num_train=100000, num_val=15000, epochs=300, batch_size=256):
    project_dir = "2026-03-12_causal_graph"
    plots_dir, checkpoints_dir = os.path.join(project_dir, 'plots'), os.path.join(project_dir, 'checkpoints')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} | Experiment: Causal Graph | Events: {num_train}", flush=True)
    
    # Снижаем batch_size, так как причинный граф может быть плотнее
    train_loader = DataLoader(BaikalCausalDataset(FILE_PATH, 0, num_train), batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(BaikalCausalDataset(FILE_PATH, num_train, num_val), batch_size=batch_size, num_workers=4)
    train_eval_loader = DataLoader(BaikalCausalDataset(FILE_PATH, 0, num_val), batch_size=batch_size, num_workers=4)

    model = GCN_Causal(13, 2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=30)
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0]).to(device))
    
    history = {'t_loss': [], 'v_loss': [], 't_prec': [], 'v_prec': [], 't_rec': [], 'v_rec': [], 't_p9': [], 'v_p9': []}
    
    print("Starting training with Causal Graph...", flush=True)
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
        
        history['t_loss'].append(t_loss); history['v_loss'].append(v_loss)
        history['t_prec'].append(t_prec); history['v_prec'].append(v_prec)
        history['t_rec'].append(t_rec); history['v_rec'].append(v_rec)
        history['t_p9'].append(t_p9); history['v_p9'].append(v_p9)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:03d} | Time: {epoch_time:.1f}s | LR: {current_lr:.2e} | T-Loss: {t_loss:.4f} | V-Loss: {v_loss:.4f} | T-Prec: {t_prec:.4f} | V-Prec: {v_prec:.4f} | T-Rec: {t_rec:.4f} | V-Rec: {v_rec:.4f} | T-P@R0.9: {t_p9:.4f} | V-P@R0.9: {v_p9:.4f}", flush=True)
        scheduler.step(v_loss)
        
        if epoch % 20 == 0:
            torch.save(model.state_dict(), os.path.join(checkpoints_dir, 'model_causal.pt'))
            plt.figure(figsize=(18, 5))
            plt.subplot(1,4,1); plt.plot(history['t_loss'], label='T'); plt.plot(history['v_loss'], label='V'); plt.title('Loss'); plt.legend()
            plt.subplot(1,4,2); plt.plot(history['t_prec'], label='T'); plt.plot(history['v_prec'], label='V'); plt.title('Precision'); plt.legend()
            plt.subplot(1,4,3); plt.plot(history['t_rec'], label='T'); plt.plot(history['v_rec'], label='V'); plt.title('Recall'); plt.legend()
            plt.subplot(1,4,4); plt.plot(history['t_p9'], label='T'); plt.plot(history['v_p9'], label='V'); plt.title('P@R0.9'); plt.legend()
            plt.tight_layout(); plt.savefig(os.path.join(plots_dir, 'metrics_causal.png')); plt.close()

if __name__ == "__main__":
    train()
