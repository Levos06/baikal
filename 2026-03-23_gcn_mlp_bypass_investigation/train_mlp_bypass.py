import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import os
import time
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, precision_recall_curve
import numpy as np
import random
import glob

DATA_DIR = "data_processed"
PROJECT_DIR = "2026-03-23_gcn_mlp_bypass_investigation"
BATCH_SIZE = 256
THRESHOLD = 0.5
VIRTUAL_EPOCH_SIZE = 400 # ~100k events
TOTAL_VIRTUAL_EPOCHS = 400

# Создаем директории, если их нет (на всякий случай)
os.makedirs(f"{PROJECT_DIR}/checkpoints", exist_ok=True)
os.makedirs(f"{PROJECT_DIR}/plots", exist_ok=True)

class ChunkedDataset(torch.utils.data.IterableDataset):
    def __init__(self, split='train', shuffle=True):
        super().__init__()
        self.split = split
        self.files = sorted(glob.glob(os.path.join(DATA_DIR, split, "chunk_*.pt")))
        self.shuffle = shuffle

    def __iter__(self):
        while True:
            file_list = self.files.copy()
            if self.shuffle: random.shuffle(file_list)
            for file in file_list:
                try:
                    data_list = torch.load(file, weights_only=False)
                    if self.shuffle: random.shuffle(data_list)
                    for data in data_list: yield data
                except Exception as e:
                    print(f"Error loading {file}: {e}")
                    continue

class MLP_Bypass(torch.nn.Module):
    def __init__(self, h1, h2, h_mid):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(h1, h_mid),
            torch.nn.GELU(),
            torch.nn.Linear(h_mid, h2)
        )
    def forward(self, x): return self.net(x)

class GCN_MLPBypass_5Layer(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Graph Convolutions
        self.conv1 = GCNConv(in_channels, 512)
        self.conv2 = GCNConv(512, 1024)
        self.conv3 = GCNConv(1024, 1024)
        self.conv4 = GCNConv(1024, 512)
        self.conv5 = GCNConv(512, out_channels)
        
        # MLP Bypasses
        self.mlp1 = MLP_Bypass(in_channels, 512, 512)
        self.mlp2 = MLP_Bypass(512, 1024, 1536)
        self.mlp3 = MLP_Bypass(1024, 1024, 2048)
        self.mlp4 = MLP_Bypass(1024, 512, 1536)

    def forward(self, x, edge_index):
        x = F.gelu(self.conv1(x, edge_index) + self.mlp1(x))
        x = F.gelu(self.conv2(x, edge_index) + self.mlp2(x))
        x = F.gelu(self.conv3(x, edge_index) + self.mlp3(x))
        x = F.gelu(self.conv4(x, edge_index) + self.mlp4(x))
        return self.conv5(x, edge_index)

def get_precision_at_recall(labels, probs, target_recall=0.9):
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    if len(recall) == 0 or np.max(recall) < target_recall: return 0.0
    return np.interp(target_recall, recall[::-1], precision[::-1])

def evaluate(model, loader, device, criterion, num_batches=100):
    model.eval()
    total_loss, all_labels, all_probs = 0, [], []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= num_batches: break
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index)
            loss = criterion(out, batch.y)
            total_loss += loss.item()
            probs = F.softmax(out, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
            
    if len(all_labels) == 0: return 0, 0, 0, 0

    all_labels, all_probs = np.array(all_labels), np.array(all_probs)
    preds = (all_probs > THRESHOLD).astype(int)
    
    prec = precision_score(all_labels, preds, zero_division=0)
    rec = recall_score(all_labels, preds, zero_division=0)
    p_at_r9 = get_precision_at_recall(all_labels, all_probs, 0.9)
    
    return total_loss/(i+1), prec, rec, p_at_r9

def plot_metrics(history, path):
    epochs = range(1, len(history['t_loss']) + 1)
    plt.figure(figsize=(16, 10))
    
    # Loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, history['t_loss'], label='Train')
    plt.plot(epochs, history['v_loss'], label='Val')
    plt.title('Loss'); plt.legend(); plt.grid(True)
    
    # Precision
    plt.subplot(2, 2, 2)
    plt.plot(epochs, history['t_prec'], label='Train')
    plt.plot(epochs, history['v_prec'], label='Val')
    plt.title('Precision'); plt.legend(); plt.grid(True)
    
    # Recall
    plt.subplot(2, 2, 3)
    plt.plot(epochs, history['t_rec'], label='Train')
    plt.plot(epochs, history['v_rec'], label='Val')
    plt.title('Recall'); plt.legend(); plt.grid(True)
    
    # P@R0.9
    plt.subplot(2, 2, 4)
    plt.plot(epochs, history['t_p9'], label='Train')
    plt.plot(epochs, history['v_p9'], label='Val')
    plt.title('P@R0.9'); plt.legend(); plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} | MLP-BYPASS STUDY | GCN 5Layer (LR Reduced to 5e-5)")
    
    model = GCN_MLPBypass_5Layer(5, 2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5) # Reduced LR
    criterion = torch.nn.CrossEntropyLoss()
    
    # Load checkpoint 180
    checkpoint_path = f"{PROJECT_DIR}/checkpoints/model_mlp_bypass_180.pt"
    if os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
        start_epoch = 181
    else:
        start_epoch = 1

    train_loader = DataLoader(ChunkedDataset('train'), batch_size=BATCH_SIZE)
    train_eval_loader = DataLoader(ChunkedDataset('train', shuffle=True), batch_size=BATCH_SIZE)
    val_loader = DataLoader(ChunkedDataset('val', shuffle=False), batch_size=BATCH_SIZE)
    
    history = {k: [] for k in ['t_loss', 't_prec', 't_rec', 't_p9', 'v_loss', 'v_prec', 'v_rec', 'v_p9']}
    
    batch_count = 0
    epoch = start_epoch
    start_time = time.time()
    
    model.train()
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        
        batch_count += 1
        if batch_count % VIRTUAL_EPOCH_SIZE == 0:
            epoch_time = time.time() - start_time
            
            # Evaluate
            t_loss, t_prec, t_rec, t_p9 = evaluate(model, train_eval_loader, device, criterion, num_batches=100)
            v_loss, v_prec, v_rec, v_p9 = evaluate(model, val_loader, device, criterion, num_batches=100)
            
            # Log history
            history['t_loss'].append(t_loss); history['v_loss'].append(v_loss)
            history['t_prec'].append(t_prec); history['v_prec'].append(v_prec)
            history['t_rec'].append(t_rec); history['v_rec'].append(v_rec)
            history['t_p9'].append(t_p9);   history['v_p9'].append(v_p9)
            
            print(f"Epoch {epoch:04d} | Time: {epoch_time:.1f}s | "
                  f"T-Loss: {t_loss:.4f} | V-Loss: {v_loss:.4f} | "
                  f"T-Prec: {t_prec:.4f} | V-Prec: {v_prec:.4f} | "
                  f"T-Rec: {t_rec:.4f} | V-Rec: {v_rec:.4f} | "
                  f"T-P@R0.9: {t_p9:.4f} | V-P@R0.9: {v_p9:.4f}", flush=True)
            
            # Save & Plot every 10 epochs
            if epoch % 10 == 0:
                torch.save(model.state_dict(), f"{PROJECT_DIR}/checkpoints/model_mlp_bypass_{epoch}.pt")
                plot_metrics(history, f"{PROJECT_DIR}/plots/metrics_mlp_bypass.png")
            
            epoch += 1
            if epoch > TOTAL_VIRTUAL_EPOCHS: break
            start_time = time.time()
            model.train()

if __name__ == "__main__":
    train()
