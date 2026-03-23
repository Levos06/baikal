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
        file_list = self.files.copy()
        if self.shuffle:
            random.shuffle(file_list)
        for file in file_list:
            data_list = torch.load(file, weights_only=False)
            if self.shuffle:
                random.shuffle(data_list)
            for data in data_list:
                yield data

class LinearTestGCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 5 GCN слоев
        self.conv1 = GCNConv(in_channels, 512)
        self.conv2 = GCNConv(512, 1024)
        self.conv3 = GCNConv(1024, 1024)
        self.conv4 = GCNConv(1024, 512)
        self.conv5 = GCNConv(512, out_channels)
        
        # 4 Линейных слоя (вместо residual связей)
        # Размеры подобраны так, чтобы соответствовать proj слоям из ResGCN
        self.lin1 = torch.nn.Linear(512, 512)
        self.lin2 = torch.nn.Linear(1024, 1024)
        self.lin3 = torch.nn.Linear(1024, 1024)
        self.lin4 = torch.nn.Linear(512, 512)

    def forward(self, x, edge_index):
        # Последовательная цепочка БЕЗ сложения
        x = F.gelu(self.conv1(x, edge_index))
        x = F.gelu(self.lin1(x))
        
        x = F.gelu(self.conv2(x, edge_index))
        x = F.gelu(self.lin2(x))
        
        x = F.gelu(self.conv3(x, edge_index))
        x = F.gelu(self.lin3(x))
        
        x = F.gelu(self.conv4(x, edge_index))
        x = F.gelu(self.lin4(x))
        
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
    project_dir = "2026-03-18_resgcn_optimized_full"
    # Явно используем GPU 1 для этого теста
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} | LINEAR ONLY TEST (No Residual) | Full Dataset")
    
    train_dataset = ChunkedDataset('train', shuffle=True)
    val_dataset = ChunkedDataset('val', shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=256)
    val_loader = DataLoader(val_dataset, batch_size=256)

    model = LinearTestGCN(5, 2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    virtual_epoch_size = 400 
    batch_count = 0
    epoch = 1
    start_time = time.time()
    
    history = {'t_loss': [], 'v_loss': [], 'v_p9': []}
    
    model.train()
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        criterion(model(batch.x, batch.edge_index), batch.y).backward(); optimizer.step()
        
        batch_count += 1
        if batch_count % virtual_epoch_size == 0:
            epoch_time = time.time() - start_time
            v_loss, v_prec, v_rec, v_p9 = evaluate(model, val_loader, device, criterion, num_batches=100)
            
            print(f"Epoch {epoch:03d} | Time: {epoch_time:.1f}s | V-Loss: {v_loss:.4f} | V-Prec: {v_prec:.4f} | V-Rec: {v_rec:.4f} | V-P@R0.9: {v_p9:.4f}", flush=True)
            
            if epoch % 5 == 0:
                torch.save(model.state_dict(), f"{project_dir}/checkpoints/model_linear_test_epoch_{epoch}.pt")
            
            epoch += 1
            start_time = time.time()
            model.train()
            if epoch > 150: break

if __name__ == "__main__":
    train()
