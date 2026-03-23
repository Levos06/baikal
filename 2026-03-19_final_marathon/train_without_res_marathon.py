import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from sklearn.metrics import precision_score, recall_score, precision_recall_curve
import os
import matplotlib.pyplot as plt
import time
import random
import glob
import numpy as np

THRESHOLD = 0.5
DATA_DIR = "data_processed"

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
                data_list = torch.load(file, weights_only=False)
                if self.shuffle: random.shuffle(data_list)
                for data in data_list: yield data

class GCN_5Layer(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 512)
        self.conv2 = GCNConv(512, 1024)
        self.conv3 = GCNConv(1024, 1024)
        self.conv4 = GCNConv(1024, 512)
        self.conv5 = GCNConv(512, out_channels)

    def forward(self, x, edge_index):
        x = F.gelu(self.conv1(x, edge_index))
        x = F.gelu(self.conv2(x, edge_index))
        x = F.gelu(self.conv3(x, edge_index))
        x = F.gelu(self.conv4(x, edge_index))
        return self.conv5(x, edge_index)

def get_precision_at_recall(labels, probs, target_recall=0.9):
    precision, recall, _ = precision_recall_curve(labels, probs)
    if len(recall) == 0 or np.max(recall) < target_recall: return 0.0
    return np.interp(target_recall, recall[::-1], precision[::-1])

def evaluate(model, loader, device, criterion, num_batches=100):
    model.eval()
    all_probs, all_labels = [], []
    total_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= num_batches: break
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index)
            total_loss += criterion(out, batch.y).item()
            probs = F.softmax(out, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy()); all_labels.extend(batch.y.cpu().numpy())
    all_labels, all_probs = np.array(all_labels), np.array(all_probs)
    p_at_r9 = get_precision_at_recall(all_labels, all_probs, 0.9)
    preds = (all_probs > THRESHOLD).astype(int)
    return total_loss/num_batches, precision_score(all_labels, preds, zero_division=0), recall_score(all_labels, preds, zero_division=0), p_at_r9

def train():
    project_dir = "2026-03-19_final_marathon"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} | MARATHON 2000 | NO RES")
    
    train_loader = DataLoader(ChunkedDataset('train'), batch_size=256)
    val_loader = DataLoader(ChunkedDataset('val', shuffle=False), batch_size=256)

    model = GCN_5Layer(5, 2).to(device)
    
    # Load checkpoint
    checkpoint_path = f"{project_dir}/checkpoints/model_no_res_marathon_150.pt"
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
        start_epoch = 151
    else:
        start_epoch = 1
        
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    history = {'v_loss': [], 'v_prec': [], 'v_rec': [], 'v_p9': []}
    virtual_epoch_size = 400 
    batch_count = 0
    epoch = start_epoch
    start_time = time.time()
    
    model.train()
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        criterion(model(batch.x, batch.edge_index), batch.y).backward(); optimizer.step()
        
        batch_count += 1
        if batch_count % virtual_epoch_size == 0:
            epoch_time = time.time() - start_time
            v_loss, v_prec, v_rec, v_p9 = evaluate(model, val_loader, device, criterion, num_batches=100)
            
            history['v_loss'].append(v_loss); history['v_prec'].append(v_prec)
            history['v_rec'].append(v_rec); history['v_p9'].append(v_p9)
            
            print(f"Epoch {epoch:04d} | Time: {epoch_time:.1f}s | V-Loss: {v_loss:.4f} | V-P@R0.9: {v_p9:.4f}", flush=True)
            
            if epoch % 50 == 0:
                torch.save(model.state_dict(), f"{project_dir}/checkpoints/model_no_res_marathon_{epoch}.pt")
                # Plot progress
                plt.figure(figsize=(15, 5))
                plt.subplot(1, 3, 1); plt.plot(history['v_loss']); plt.title('Val Loss')
                plt.subplot(1, 3, 2); plt.plot(history['v_prec'], label='P'); plt.plot(history['v_rec'], label='R'); plt.title('P/R'); plt.legend()
                plt.subplot(1, 3, 3); plt.plot(history['v_p9']); plt.title('V-P@R0.9')
                plt.savefig(f"{project_dir}/plots/metrics_no_res_marathon.png"); plt.close()
            
            epoch += 1
            start_time = time.time()
            model.train()
            if epoch > 2000: break

if __name__ == "__main__":
    train()
