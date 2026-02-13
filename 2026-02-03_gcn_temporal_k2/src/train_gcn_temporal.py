import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, LayerNorm
from sklearn.metrics import precision_score, recall_score
import os
import matplotlib.pyplot as plt
import time

FILE_PATH = "/home3/ivkhar/Baikal/data/normed/baikal_mc2020_multi_split_0924mid_eq_norm.h5"
THRESHOLD = 0.5  # Threshold for class 1 (signal)

class BaikalTemporalGraphDataset(Dataset):
    def __init__(self, file_path, start_ev=0, num_events=1000, k=2):
        super().__init__()
        self.file_path = file_path
        self.start_ev = start_ev
        self.num_events = num_events
        self.k = k
        
        with h5py.File(self.file_path, 'r') as f:
            # Fetch starts for the requested range
            self.starts = f['train/ev_starts/data'][start_ev : start_ev + num_events + 1]
            
    def len(self):
        return self.num_events

    def get(self, idx):
        with h5py.File(self.file_path, 'r') as f:
            start = self.starts[idx]
            end = self.starts[idx + 1]
            
            x_features = torch.from_numpy(f['train/data/data'][start:end]).float()
            labels = f['train/labels/data'][start:end]
            y = torch.from_numpy((labels > 0).astype(np.int64))
            
            num_nodes = x_features.size(0)
            edge_index = self._get_temporal_edges(num_nodes, self.k)
            
            return Data(x=x_features, edge_index=edge_index, y=y)

    def _get_temporal_edges(self, num_nodes, k):
        if num_nodes <= 1:
            return torch.zeros((2, 0), dtype=torch.long)
        
        row, col = [], []
        # Connect to k nearest neighbors in time (they are already sorted)
        # We do this bi-directionally
        for d in range(1, min(k + 1, num_nodes)):
            # src = 0 to N-1-d, dst = d to N-1
            src = np.arange(num_nodes - d)
            dst = np.arange(d, num_nodes)
            
            row.append(src)
            col.append(dst)
            row.append(dst)
            col.append(src)
            
        edge_index = np.stack([np.concatenate(row), np.concatenate(col)], axis=0)
        return torch.from_numpy(edge_index).long()

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.ln1 = LayerNorm(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.ln2 = LayerNorm(hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.ln1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = self.ln2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        
        x = self.conv3(x, edge_index)
        return x

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)

def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            if batch.edge_index.numel() == 0:
                 pass
            
            out = model(batch.x, batch.edge_index)
            loss = criterion(out, batch.y)
            total_loss += loss.item()
            
            probs = F.softmax(out, dim=1)[:, 1]
            preds = (probs > threshold).cpu().numpy().astype(int)
            labels = batch.y.cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels)
            
    avg_loss = total_loss / len(loader) if len(loader) > 0 else 0
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    
    return avg_loss, precision, recall

def train():
    # Setup directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    plots_dir = os.path.join(project_dir, 'plots')
    checkpoints_dir = os.path.join(project_dir, 'checkpoints')
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    K_NEIGHBORS = 4
    BATCH_SIZE = 512
    
    print(f"Loading data with k={K_NEIGHBORS}, batch_size={BATCH_SIZE}...")
    
    # Using more data since we have faster edge calculation
    train_dataset = BaikalTemporalGraphDataset(FILE_PATH, start_ev=0, num_events=50000, k=K_NEIGHBORS)
    val_dataset = BaikalTemporalGraphDataset(FILE_PATH, start_ev=50000, num_events=10000, k=K_NEIGHBORS)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    model = GCN(in_channels=5, hidden_channels=128, out_channels=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    
    # Loss weight for imbalanced classes
    weights = torch.tensor([1.0, 5.0]).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    
    checkpoint_path = os.path.join(checkpoints_dir, 'best_model.pt')
    early_stopping = EarlyStopping(patience=15, path=checkpoint_path)
    
    epochs = 80
    print(f"Starting training for up to {epochs} epochs...")
    
    history = {
        'train_loss': [], 'val_loss': [],
        'train_prec': [], 'val_prec': [],
        'train_rec': [], 'val_rec': []
    }
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        all_train_preds = []
        all_train_labels = []
        
        start_time = time.time()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            probs = F.softmax(out, dim=1)[:, 1]
            preds = (probs > THRESHOLD).cpu().numpy().astype(int)
            labels = batch.y.cpu().numpy()
            all_train_preds.extend(preds)
            all_train_labels.extend(labels)
            
        epoch_time = time.time() - start_time
        avg_train_loss = train_loss / len(train_loader)
        train_prec = precision_score(all_train_labels, all_train_preds, zero_division=0)
        train_rec = recall_score(all_train_labels, all_train_preds, zero_division=0)
        
        val_loss, val_prec, val_rec = evaluate(model, val_loader, device, threshold=THRESHOLD)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        history['train_prec'].append(train_prec)
        history['val_prec'].append(val_prec)
        history['train_rec'].append(train_rec)
        history['val_rec'].append(val_rec)
        
        print(f"Epoch {epoch:03d} | {epoch_time:.1f}s | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Prec: {val_prec:.4f} | Val Rec: {val_rec:.4f}")
        
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    # Plotting
    plt.figure(figsize=(18, 5))
    epochs_range = range(1, len(history['train_loss']) + 1)
    
    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, history['train_loss'], label='Train')
    plt.plot(epochs_range, history['val_loss'], label='Val')
    plt.title('Loss')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, history['train_prec'], label='Train')
    plt.plot(epochs_range, history['val_prec'], label='Val')
    plt.title('Precision')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, history['train_rec'], label='Train')
    plt.plot(epochs_range, history['val_rec'], label='Val')
    plt.title('Recall')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'training_metrics.png'))
    print(f"Plots saved to {plots_dir}")

if __name__ == "__main__":
    train()
