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

class BaikalCNNDataset(Dataset):
    def __init__(self, file_path, start_ev=0, num_events=1000):
        super().__init__()
        self.file_path = file_path
        with h5py.File(self.file_path, 'r') as f:
            self.starts = f['train/ev_starts/data'][start_ev : start_ev + num_events + 1]
            
    def __len__(self): return len(self.starts) - 1

    def __getitem__(self, idx):
        with h5py.File(self.file_path, 'r') as f:
            start, end = self.starts[idx], self.starts[idx + 1]
            # CNN ожидает (Channels, Sequence_Length)
            x = f['train/data/data'][start:end].T
            y = (f['train/labels/data'][start:end] != 0).astype(np.int64)
            return torch.from_numpy(x).float(), torch.from_numpy(y).long()

class CNN1D(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(in_channels, 128, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv1d(128, 256, kernel_size=5, padding=2)
        self.conv3 = torch.nn.Conv1d(256, 128, kernel_size=3, padding=1)
        self.fc = torch.nn.Linear(128, out_channels)

    def forward(self, x):
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = F.gelu(self.conv3(x))
        # (Batch, 128, Seq) -> (Batch, Seq, 128)
        x = x.transpose(1, 2)
        return self.fc(x)

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
            # Flatten for loss calculation
            loss = criterion(out.reshape(-1, 2), y.reshape(-1))
            total_loss += loss.item()
            
            probs = F.softmax(out, dim=2)[:, :, 1].reshape(-1)
            all_probs.extend(probs.cpu().numpy()); all_labels.extend(y.reshape(-1).cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    p_at_r9 = get_precision_at_recall(np.array(all_labels), np.array(all_probs), 0.9)
    return avg_loss, p_at_r9

def train(num_train=100000, num_val=15000, epochs=20):
    project_dir = "2026-03-14_cnn_baseline"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} | 1D-CNN Baseline | Events: {num_train}")
    
    # Batch size = 1 because events have different lengths
    train_loader = DataLoader(BaikalCNNDataset(FILE_PATH, 0, num_train), batch_size=1, shuffle=True)
    val_loader = DataLoader(BaikalCNNDataset(FILE_PATH, num_train, num_val), batch_size=1)

    model = CNN1D(5, 2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out.reshape(-1, 2), y.reshape(-1))
            loss.backward(); optimizer.step()
            train_loss += loss.item()
        
        v_loss, v_p9 = evaluate(model, val_loader, device)
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch:02d} | Time: {epoch_time:.1f}s | T-Loss: {train_loss/len(train_loader):.4f} | V-Loss: {v_loss:.4f} | V-P@R0.9: {v_p9:.4f}")

    torch.save(model.state_dict(), os.path.join(project_dir, "checkpoints/model_cnn.pt"))

if __name__ == "__main__":
    train()
