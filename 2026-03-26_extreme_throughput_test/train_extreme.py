import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import os
import time
import glob
import math
import random

DATA_DIR = "data_processed"
PROJECT_DIR = "2026-03-26_extreme_throughput_test"
BATCH_SIZE = 512 # Targeted for RTX 4090 saturation
NUM_WORKERS = 4  # Our benchmarked optimum
C_WATER = 0.225

class ChunkedDataset(torch.utils.data.IterableDataset):
    def __init__(self, split='train'):
        super().__init__()
        self.files = sorted(glob.glob(os.path.join(DATA_DIR, split, "chunk_*.pt")))
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None: iter_files = self.files
        else:
            per_worker = int(math.ceil(len(self.files) / float(worker_info.num_workers)))
            iter_files = self.files[worker_info.id * per_worker : (worker_info.id + 1) * per_worker]
        
        while True:
            file_list = iter_files.copy()
            random.shuffle(file_list)
            for f in file_list:
                try:
                    data_list = torch.load(f, weights_only=False)
                    for data in data_list: yield data
                except: continue

def add_features(batch):
    x = batch.x
    ptr = batch.ptr
    first = ptr[:-1]; graph_sizes = ptr[1:] - ptr[:-1]
    t0 = torch.repeat_interleave(x[first, 1], graph_sizes)
    x0 = torch.repeat_interleave(x[first, 2], graph_sizes)
    y0 = torch.repeat_interleave(x[first, 3], graph_sizes)
    z0 = torch.repeat_interleave(x[first, 4], graph_sizes)
    dt = x[:, 1] - t0
    dx, dy, dz = x[:, 2]-x0, x[:, 3]-y0, x[:, 4]-z0
    dr2 = dx**2 + dy**2 + dz**2
    dr = torch.sqrt(dr2 + 1e-8)
    s2 = (C_WATER * dt)**2 - dr2
    r = torch.sqrt(x[:, 2]**2 + x[:, 3]**2 + 1e-8)
    phi = torch.atan2(x[:, 3], x[:, 2])
    rho = torch.sqrt(x[:, 2]**2 + x[:, 3]**2 + x[:, 4]**2 + 1e-8)
    cosT = x[:, 4] / (rho + 1e-8)
    tof = dt - dr/C_WATER
    ext = torch.stack([s2, dt, dr, r, phi, rho, cosT, tof], dim=1)
    batch.x = torch.cat([x, ext], dim=1)
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

def train():
    device = torch.device('cuda')
    print(f"Starting EXTREME THROUGHPUT TEST on {device} (BS={BATCH_SIZE}, Workers={NUM_WORKERS})")
    
    model = JKResGCN_v2(13, 2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    loader = DataLoader(ChunkedDataset(), batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)
    
    model.train()
    batch_count = 0
    start_time = time.time()
    events_count = 0
    
    for i, batch in enumerate(loader):
        batch = add_features(batch.to(device))
        out = model(batch.x, batch.edge_index)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        batch_count += 1
        events_count += batch.num_graphs
        
        if batch_count % 100 == 0:
            elapsed = time.time() - start_time
            throughput = events_count / elapsed
            print(f"Batch {batch_count:05d} | Loss: {loss.item():.4f} | Speed: {throughput:.2f} events/s")

if __name__ == "__main__":
    train()
