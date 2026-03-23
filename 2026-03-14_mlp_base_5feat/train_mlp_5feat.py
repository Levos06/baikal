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

class BaikalMLP5Dataset(Dataset):
    def __init__(self, file_path, start_ev=0, num_events=1000):
        super().__init__()
        with h5py.File(file_path, 'r') as f:
            starts = f['train/ev_starts/data'][start_ev : start_ev + num_events + 1]
            start_idx, end_idx = starts[0], starts[-1]
            # Load only first 5 features
            self.x = f['train/data/data'][start_idx:end_idx]
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
    project_dir = "2026-03-14_mlp_base_5feat"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} | MLP BASE 5 FEAT | Training on {num_train_events} events")
    
    train_dataset = BaikalMLP5Dataset(FILE_PATH, 0, num_train_events)
    val_dataset = BaikalMLP5Dataset(FILE_PATH, num_train_events, num_val_events)
    
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1024)
    train_eval_loader = DataLoader(train_dataset, batch_size=1024, sampler=torch.utils.data.SubsetRandomSampler(range(min(len(train_dataset), 500000))))

    model = MLP(5, 2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    history = {'t_loss': [], 'v_loss': [], 't_p9': [], 'v_p9': []}
    
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
        
        history['t_loss'].append(t_loss); history['v_loss'].append(v_loss)
        history['t_p9'].append(t_p9); history['v_p9'].append(v_p9)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1); plt.plot(history['t_loss'], label='T'); plt.plot(history['v_loss'], label='V'); plt.title('Loss'); plt.legend()
    plt.subplot(1, 2, 2); plt.plot(history['t_p9'], label='T'); plt.plot(history['v_p9'], label='V'); plt.title('P@R0.9'); plt.legend()
    plt.savefig(os.path.join(project_dir, "plots/metrics_mlp_5feat.png"))
    
    torch.save(model.state_dict(), os.path.join(project_dir, "checkpoints/model_mlp_5feat.pt"))

if __name__ == "__main__":
    train()
