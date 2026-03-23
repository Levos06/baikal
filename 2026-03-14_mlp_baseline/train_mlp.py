import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_score, recall_score, precision_recall_curve
import os
import matplotlib.pyplot as plt
import time

FILE_PATH = "/home3/ivkhar/Baikal/data/normed/baikal_mc2020_multi_split_0924mid_eq_norm.h5"
THRESHOLD = 0.5

# Stats for normalization
T_RES_MEAN, T_RES_STD = 518.25, 1560.43
R_MEAN, R_STD = 1.34, 0.45
PHI_MEAN, PHI_STD = -0.31, 1.79
DEG_MEAN, DEG_STD = 3.0, 1.0 # For temporal k=2
RHO_MEAN, RHO_STD = 1.69, 0.40
COS_MEAN, COS_STD = 0.03, 0.57
TOF_MEAN, TOF_STD = 9.82, 2.40
NEIGHQ_MEAN, NEIGHQ_STD = 0.09, 7.85

FEATURE_NAMES = [
    "Charge", "Time", "X", "Y", "Z", 
    "t_res", "R", "Phi", "Degree", 
    "Rho", "CosTheta", "ToF_Res", "NeighQ"
]

class BaikalMLPDataset(Dataset):
    def __init__(self, file_path, start_ev=0, num_events=1000):
        super().__init__()
        with h5py.File(file_path, 'r') as f:
            starts = f['train/ev_starts/data'][start_ev : start_ev + num_events + 1]
            start_idx, end_idx = starts[0], starts[-1]
            
            # Load all data at once for MLP (no graph building needed)
            x_main = f['train/data/data'][start_idx:end_idx]
            q, t, x_pos, y_pos, z_pos = x_main[:,0], x_main[:,1], x_main[:,2], x_main[:,3], x_main[:,4]
            
            x_tres = (f['train/t_res/data'][start_idx:end_idx] - T_RES_MEAN) / T_RES_STD
            r_cyl = np.sqrt(x_pos**2 + y_pos**2)
            x_r = (r_cyl - R_MEAN) / R_STD
            x_phi = (np.arctan2(y_pos, x_pos) - PHI_MEAN) / PHI_STD
            rho = np.sqrt(x_pos**2 + y_pos**2 + z_pos**2)
            x_rho = (rho - RHO_MEAN) / RHO_STD
            x_cos = (np.divide(z_pos, rho, out=np.zeros_like(z_pos), where=rho!=0) - COS_MEAN) / COS_STD
            
            # For Degree, ToF and NeighQ in MLP, we'll use precalculated temporal k=2 stats 
            # to keep it fair with the 13-feat GCN experiment
            # (In a real MLP these would be 0 or pre-calculated)
            x_deg = np.zeros_like(q) # Placeholder or precalc
            x_tof = np.zeros_like(q)
            x_nq = np.zeros_like(q)
            
            self.x = np.column_stack([x_main, x_tres, x_r, x_phi, x_deg, x_rho, x_cos, x_tof, x_nq])
            self.y = (f['train/labels/data'][start_idx:end_idx] != 0).astype(np.int64)
            
    def __len__(self): return len(self.y)
    def __getitem__(self, idx):
        return torch.from_numpy(self.x[idx]).float(), torch.tensor(self.y[idx]).long()

class MLP(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_channels, 512)
        self.fc2 = torch.nn.Linear(512, 768)
        self.fc3 = torch.nn.Linear(768, 512)
        self.fc4 = torch.nn.Linear(512, out_channels)
    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        x = F.gelu(self.fc3(x))
        return self.fc4(x)

def get_precision_at_recall(labels, probs, target_recall=0.9):
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    if np.max(recall) < 0.9: return 0.0
    return np.interp(0.9, recall[::-1], precision[::-1])

def evaluate(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    total_loss = 0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            total_loss += criterion(out, y).item()
            probs = F.softmax(out, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy()); all_labels.extend(y.cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    preds = (all_probs > THRESHOLD).astype(int)
    prec = precision_score(all_labels, preds, zero_division=0)
    rec = recall_score(all_labels, preds, zero_division=0)
    p_at_r9 = get_precision_at_recall(all_labels, all_probs, 0.9)
    return avg_loss, prec, rec, p_at_r9

def train(num_train_events=100000, num_val_events=20000, epochs=20):
    project_dir = "2026-03-14_mlp_baseline"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} | MLP Baseline | Training on {num_train_events} events")
    
    train_dataset = BaikalMLPDataset(FILE_PATH, 0, num_train_events)
    val_dataset = BaikalMLPDataset(FILE_PATH, num_train_events, num_val_events)
    
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1024)
    # Subset of train for metrics
    train_eval_loader = DataLoader(train_dataset, batch_size=1024, sampler=torch.utils.data.SubsetRandomSampler(range(min(len(train_dataset), 500000))))

    model = MLP(13, 2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward(); optimizer.step()
        
        t_loss, t_prec, t_rec, t_p9 = evaluate(model, train_eval_loader, device)
        v_loss, v_prec, v_rec, v_p9 = evaluate(model, val_loader, device)
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch:02d} | Time: {epoch_time:.1f}s | T-Loss: {t_loss:.4f} | V-Loss: {v_loss:.4f} | T-P@R0.9: {t_p9:.4f} | V-P@R0.9: {v_p9:.4f}")

    # Feature Importance (Permutation)
    print("\nCalculating Feature Importance...")
    baseline_p9 = v_p9
    importances = []
    val_x = val_dataset.x
    val_y = val_dataset.y
    
    for i in range(13):
        save_col = val_x[:, i].copy()
        np.random.shuffle(val_x[:, i]) # Shuffle one feature
        
        # Eval
        temp_loader = DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(val_x).float(), torch.from_numpy(val_y).long()), batch_size=1024)
        _, _, _, shuffled_p9 = evaluate(model, temp_loader, device)
        
        importance = baseline_p9 - shuffled_p9
        importances.append(importance)
        val_x[:, i] = save_col # Restore
        print(f"Feature {FEATURE_NAMES[i]}: Importance {importance:.4f}")

    # Plot Importance
    plt.figure(figsize=(12, 6))
    plt.bar(FEATURE_NAMES, importances)
    plt.title("Feature Importance (Permutation on P@R0.9)")
    plt.xticks(rotation=45)
    plt.ylabel("Drop in P@R0.9")
    plt.tight_layout()
    plt.savefig(os.path.join(project_dir, "plots/feature_importance.png"))
    
    torch.save(model.state_dict(), os.path.join(project_dir, "checkpoints/model_mlp.pt"))

if __name__ == "__main__":
    train()
