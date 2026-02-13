import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import precision_score, recall_score
import os
import matplotlib.pyplot as plt

FILE_PATH = "/home3/ivkhar/Baikal/data/normed/baikal_mc2020_multi_split_0924mid_eq_norm.h5"
C_LIGHT = 0.299792458

class BaikalGraphDataset(Dataset):
    def __init__(self, file_path, start_ev=0, num_events=1000):
        super().__init__()
        self.file_path = file_path
        self.start_ev = start_ev
        self.num_events = num_events
        
        with h5py.File(self.file_path, 'r') as f:
            self.means = f['norm_param/mean'][()]
            self.stds = f['norm_param/std'][()]
            # Fetch starts for the requested range
            self.starts = f['train/ev_starts/data'][start_ev : start_ev + num_events + 1]
            
    def len(self):
        return self.num_events

    def get(self, idx):
        with h5py.File(self.file_path, 'r') as f:
            # Adjust index based on start_ev
            global_idx = self.start_ev + idx
            # But self.starts already contains the subset
            start = self.starts[idx]
            end = self.starts[idx + 1]
            
            x_features = torch.from_numpy(f['train/data/data'][start:end]).float()
            labels = f['train/labels/data'][start:end]
            y = torch.from_numpy((labels > 0).astype(np.int64))
            
            T_IDX, X_IDX, Y_IDX, Z_IDX = 1, 2, 3, 4
            feat_np = x_features.numpy()
            t = feat_np[:, T_IDX] * self.stds[T_IDX] + self.means[T_IDX]
            coords = feat_np[:, [X_IDX, Y_IDX, Z_IDX]] * self.stds[[X_IDX, Y_IDX, Z_IDX]] + self.means[[X_IDX, Y_IDX, Z_IDX]]
            
            dist_matrix = squareform(pdist(coords))
            time_matrix = np.abs(t[:, np.newaxis] - t[np.newaxis, :])
            
            adj = (dist_matrix < C_LIGHT * time_matrix)
            np.fill_diagonal(adj, 0)
            
            edge_index = torch.from_numpy(np.array(np.where(adj), dtype=np.int64))
            return Data(x=x_features, edge_index=edge_index, y=y)

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        return x

def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index)
            loss = criterion(out, batch.y)
            total_loss += loss.item()
            
            preds = out.argmax(dim=1).cpu().numpy()
            labels = batch.y.cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels)
            
    avg_loss = total_loss / len(loader)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    
    return avg_loss, precision, recall

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Increase samples: 1000 train, 200 validation
    train_dataset = BaikalGraphDataset(FILE_PATH, start_ev=0, num_events=10000)
    val_dataset = BaikalGraphDataset(FILE_PATH, start_ev=10000, num_events=2000)
    
    # Check balance
    all_y = []
    for i in range(min(100, len(train_dataset))):
        all_y.extend(train_dataset.get(i).y.numpy())
    unique, counts = np.unique(all_y, return_counts=True)
    print(f"Class distribution in first 100 train events: {dict(zip(unique, counts))}")

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    model = GCN(in_channels=5, hidden_channels=64, out_channels=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Weight for loss: signal is ~50x rarer than noise
    # Giving signal 20x more weight for a start
    weights = torch.tensor([1.0, 20.0]).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    
    epochs = 60 # Small number for testing
    print(f"Starting training for {epochs} epochs...")
    
    train_losses = []
    val_losses = []
    train_precs = []
    val_precs = []
    train_recs = []
    val_recs = []
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        all_train_preds = []
        all_train_labels = []
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            preds = out.argmax(dim=1).cpu().numpy()
            labels = batch.y.cpu().numpy()
            all_train_preds.extend(preds)
            all_train_labels.extend(labels)
            
        avg_train_loss = train_loss / len(train_loader)
        train_prec = precision_score(all_train_labels, all_train_preds, zero_division=0)
        train_rec = recall_score(all_train_labels, all_train_preds, zero_division=0)
        
        val_loss, val_prec, val_rec = evaluate(model, val_loader, device)
        
        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        train_precs.append(train_prec)
        val_precs.append(val_prec)
        train_recs.append(train_rec)
        val_recs.append(val_rec)
        
        print(f"Epoch {epoch:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Prec: {val_prec:.4f} | Val Rec: {val_rec:.4f}")

    # Plotting
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    plots_dir = os.path.join(project_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    plt.figure(figsize=(18, 5))
    
    # Plot Loss
    plt.subplot(1, 3, 1)
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, epochs + 1), val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss vs Epoch')
    
    # Plot Precision
    plt.subplot(1, 3, 2)
    plt.plot(range(1, epochs + 1), train_precs, label='Train Precision')
    plt.plot(range(1, epochs + 1), val_precs, label='Val Precision')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.legend()
    plt.title('Precision vs Epoch')
    
    # Plot Recall
    plt.subplot(1, 3, 3)
    plt.plot(range(1, epochs + 1), train_recs, label='Train Recall')
    plt.plot(range(1, epochs + 1), val_recs, label='Val Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.legend()
    plt.title('Recall vs Epoch')
    
    plt.tight_layout()
    save_path = os.path.join(plots_dir, 'training_metrics.png')
    plt.savefig(save_path)
    print(f"Plots saved to {save_path}")

if __name__ == "__main__":
    train()