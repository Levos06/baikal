import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv
from sklearn.metrics import precision_score, recall_score, precision_recall_curve
import os
import time

# --- CONFIG ---
FAST_FILE = "/home/levos/experiments/data_k4_precalculated.h5"
PROJECT_DIR = "/home/levos/experiments/2026-04-04_gat_k4_on_the_fly"
BATCH_SIZE = 256
NUM_WORKERS = 0  # CRITICAL: Disable multiprocessing to prevent silent crashes
TOTAL_EPOCHS = 2000
VIRTUAL_EPOCH_SIZE = 200
DEVICE = torch.device('cuda:0')

class BaikalFastH5Dataset(Dataset):
    def __init__(self, file_path, start_ev=0, num_events=1000):
        super().__init__()
        self.file_path = file_path
        self.start_ev = start_ev
        self.num_events = num_events
        with h5py.File(self.file_path, 'r') as f:
            self.ev_starts = f['ev_starts'][start_ev : start_ev + num_events + 1]
            self.edge_starts = f['edge_starts'][start_ev : start_ev + num_events + 1]
        self.f = None

    def len(self): return self.num_events

    def get(self, idx):
        if self.f is None: self.f = h5py.File(self.file_path, 'r', swmr=True)
        f = self.f
        ns, ne = self.ev_starts[idx], self.ev_starts[idx+1]
        es, ee = self.edge_starts[idx], self.edge_starts[idx+1]
        x = torch.from_numpy(f['x'][ns:ne]).float()
        y = torch.from_numpy(f['y'][ns:ne]).long()
        edge_index = torch.from_numpy(f['edges'][:, es:ee]).long() - ns
        return Data(x=x, edge_index=edge_index, y=y)

class GATv2_Standard(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.projs = torch.nn.ModuleList()
        self.convs.append(GATv2Conv(in_channels, 64, heads=4)); self.projs.append(torch.nn.Linear(in_channels, 256))
        for _ in range(3):
            self.convs.append(GATv2Conv(256, 64, heads=4)); self.projs.append(torch.nn.Identity())
        self.head = torch.nn.Sequential(torch.nn.Linear(256, 512), torch.nn.GELU(), torch.nn.Linear(512, 256), torch.nn.GELU(), torch.nn.Linear(256, out_channels))
    def forward(self, x, edge_index):
        h = x
        for i in range(4): h = F.gelu(self.convs[i](h, edge_index) + self.projs[i](h))
        return self.head(h)

def calculate_metrics(labels, probs):
    p, r, _ = precision_recall_curve(labels, probs)
    p_at_r09 = np.interp(0.9, r[::-1], p[::-1])
    preds = (probs > 0.5).astype(int)
    return precision_score(labels, preds, zero_division=0), recall_score(labels, preds, zero_division=0), p_at_r09

def evaluate(model, loader, num_batches=50):
    model.eval()
    all_labels, all_probs, losses = [], [], []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= num_batches: break
            batch = batch.to(DEVICE)
            out = model(batch.x, batch.edge_index)
            losses.append(F.cross_entropy(out, batch.y).item())
            all_probs.extend(F.softmax(out, dim=1)[:, 1].cpu().numpy()); all_labels.extend(batch.y.cpu().numpy())
    return np.mean(losses), *calculate_metrics(np.array(all_labels), np.array(all_probs))

def train():
    print(f"Starting SUPER-FAST GATv2 k=4 on {DEVICE}", flush=True)
    model = GATv2_Standard(21, 2).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Infinite iterable dataset via repeating indices
    train_loader = DataLoader(BaikalFastH5Dataset(FAST_FILE, 0, 80000), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(BaikalFastH5Dataset(FAST_FILE, 80000, 20000), batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)
    
    log_file = os.path.join(PROJECT_DIR, "train_k4_fast.log")
    if not os.path.exists(log_file):
        with open(log_file, "w") as f:
            f.write("Epoch | Time_Start | Duration | LR | T_Loss | T_Prec | T_Rec | T_P@R0.9 | V_Loss | V_Prec | V_Rec | V_P@R0.9\n")
    
    epoch = 1
    start_time = time.time()
    t_losses, t_probs, t_labels = [], [], []
    
    while epoch <= TOTAL_EPOCHS:
        for batch in train_loader:
            model.train()
            batch = batch.to(DEVICE)
            optimizer.zero_grad(set_to_none=True); out = model(batch.x, batch.edge_index)
            loss = F.cross_entropy(out, batch.y); loss.backward(); optimizer.step()
            
            t_losses.append(loss.item()); t_probs.extend(F.softmax(out, dim=1)[:, 1].detach().cpu().numpy()); t_labels.extend(batch.y.cpu().numpy())
            
            if len(t_losses) >= VIRTUAL_EPOCH_SIZE:
                duration = time.time() - start_time; start_clock = time.strftime("%H:%M:%S", time.localtime(start_time))
                t_loss, t_prec, t_rec, t_p9 = np.mean(t_losses), *calculate_metrics(np.array(t_labels), np.array(t_probs))
                v_loss, v_prec, v_rec, v_p9 = evaluate(model, val_loader)
                
                log_str = f"{epoch:04d} | {start_clock} | {duration:5.1f}s | 1.0e-04 | {t_loss:.4f} | {t_prec:.4f} | {t_rec:.4f} | {t_p9:.4f} | {v_loss:.4f} | {v_prec:.4f} | {v_rec:.4f} | {v_p9:.4f}"
                print(log_str, flush=True)
                with open(log_file, "a") as f: 
                    f.write(log_str + "\n")
                    f.flush()
                
                if epoch % 100 == 0: torch.save(model.state_dict(), f"{PROJECT_DIR}/checkpoints/model_k4_{epoch}.pt")
                epoch += 1
                if epoch > TOTAL_EPOCHS: break
                t_losses, t_probs, t_labels = [], [], []; start_time = time.time()

if __name__ == "__main__":
    train()
