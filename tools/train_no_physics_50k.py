import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import os
import time
import glob
import random

DATA_DIR = "data_processed_50k/train"
BATCH_SIZE = 512
NUM_WORKERS = 4
C_WATER = 0.225

def mock_add_features_gpu(batch):
    # Just pad with zeros to 13 channels without ANY math
    zeros = torch.zeros((batch.x.size(0), 8), device=batch.x.device)
    batch.x = torch.cat([batch.x, zeros], dim=1)
    return batch

class JKResGCN_v2(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 512); self.conv2 = GCNConv(512, 512); self.conv3 = GCNConv(512, 512)
        self.conv4 = GCNConv(512, 512); self.conv5 = GCNConv(512, 512)
        self.proj1 = torch.nn.Linear(in_channels, 512); self.proj2 = torch.nn.Linear(512, 512)
        self.proj3 = torch.nn.Linear(512, 512); self.proj4 = torch.nn.Linear(512, 512)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(512 * 5, 1024), torch.nn.GELU(), torch.nn.Dropout(0.1),
            torch.nn.Linear(1024, 512), torch.nn.GELU(), torch.nn.Dropout(0.1),
            torch.nn.Linear(512, out_channels)
        )
    def forward(self, x, edge_index):
        h1 = F.gelu(self.conv1(x, edge_index) + self.proj1(x))
        h2 = F.gelu(self.conv2(h1, edge_index) + self.proj2(h1))
        h3 = F.gelu(self.conv3(h2, edge_index) + self.proj3(h2))
        h4 = F.gelu(self.conv4(h3, edge_index) + self.proj4(h3))
        h5 = F.gelu(self.conv5(h4, edge_index))
        combined = torch.cat([h1, h2, h3, h4, h5], dim=1)
        return self.head(combined)

class MediumDataset(torch.utils.data.IterableDataset):
    def __init__(self, data_dir):
        super().__init__()
        self.files = sorted(glob.glob(os.path.join(data_dir, "medium_*.pt")))
    def __iter__(self):
        while True:
            file_list = self.files.copy()
            random.shuffle(file_list)
            for f in file_list:
                try:
                    data_list = torch.load(f, weights_only=False)
                    random.shuffle(data_list)
                    for data in data_list: yield data
                except: continue

def train():
    device = torch.device('cuda')
    print(f"Starting NO-PHYSICS Speed Test on {device} (BS={BATCH_SIZE})")
    model = JKResGCN_v2(13, 2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    loader = DataLoader(
        MediumDataset(DATA_DIR), 
        batch_size=BATCH_SIZE, 
        num_workers=NUM_WORKERS, 
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    model.train()
    start_time = time.time(); events_count = 0; batch_idx = 0
    for batch in loader:
        batch = mock_add_features_gpu(batch.to(device, non_blocking=True))
        
        optimizer.zero_grad(set_to_none=True)
        out = model(batch.x, batch.edge_index)
        loss = criterion(out, batch.y)
        loss.backward(); optimizer.step()
        
        batch_idx += 1; events_count += batch.num_graphs
        if batch_idx % 50 == 0:
            elapsed = time.time() - start_time
            print(f"Batch {batch_idx:05d} | Speed: {events_count/elapsed:.2f} events/s")

if __name__ == "__main__":
    train()
