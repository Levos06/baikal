import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
from torch.utils.data import DataLoader
import os
import time
import glob
import random

DATA_DIR = "data_processed_100k/train"
BATCH_SIZE = 512
C_WATER = 0.225

def add_features_gpu(batch):
    x, ptr = batch.x, batch.ptr
    first = ptr[:-1]; sizes = ptr[1:] - ptr[:-1]
    t0, x0, y0, z0 = [torch.repeat_interleave(x[first, i], sizes) for i in [1,2,3,4]]
    dt, dx, dy, dz = x[:,1]-t0, x[:,2]-x0, x[:,3]-y0, x[:,4]-z0
    dr2 = dx**2 + dy**2 + dz**2; dr = torch.sqrt(dr2 + 1e-8)
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
    def forward(self, x, edge_index, batch_idx):
        h1 = F.gelu(self.conv1(x, edge_index) + self.proj1(x))
        h2 = F.gelu(self.conv2(h1, edge_index) + self.proj2(h1))
        h3 = F.gelu(self.conv3(h2, edge_index) + self.proj3(h2))
        h4 = F.gelu(self.conv4(h3, edge_index) + self.proj4(h3))
        h5 = F.gelu(self.conv5(h4, edge_index))
        p = torch.cat([global_mean_pool(h, batch_idx) for h in [h1,h2,h3,h4,h5]], dim=1)
        return self.head(p)

class FusedDataset(torch.utils.data.IterableDataset):
    def __init__(self, data_dir):
        super().__init__()
        self.files = sorted(glob.glob(os.path.join(data_dir, "super_fused_*.pt")))
    def __iter__(self):
        while True:
            random.shuffle(self.files)
            for f in self.files:
                chunk = torch.load(f, weights_only=False)
                x, ei, y, x_ptr, e_ptr = chunk['x'], chunk['edge_index'], chunk['y'], chunk['x_ptr'], chunk['edge_ptr']
                num_graphs = len(y)
                for i in range(0, num_graphs, BATCH_SIZE):
                    bs = min(BATCH_SIZE, num_graphs - i)
                    # 1. Slice features and labels
                    bx = x[x_ptr[i] : x_ptr[i+bs]]
                    by = y[i : i+bs]
                    # 2. Relativize edge indices for the WHOLE batch
                    node_offsets = x_ptr[i : i+bs] - x_ptr[i]
                    ei_slices = [ei[:, e_ptr[i+j] : e_ptr[i+j+1]] + node_offsets[j] for j in range(bs)]
                    bei = torch.cat(ei_slices, dim=1)
                    # 3. Create graph pointers and batch index
                    hit_counts = x_ptr[i+1 : i+bs+1] - x_ptr[i : i+bs]
                    b_idx = torch.repeat_interleave(torch.arange(bs), hit_counts)
                    ptr = x_ptr[i : i+bs+1] - x_ptr[i]
                    yield Data(x=bx, edge_index=bei, y=by, batch=b_idx, ptr=ptr)

def train():
    device = torch.device('cuda')
    print(f"Starting ABSOLUTE SPEED Training on {device} (BS={BATCH_SIZE})")
    model = JKResGCN_v2(13, 2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    # DataLoader with batch_size=None since dataset already yields batches
    loader = DataLoader(FusedDataset(DATA_DIR), batch_size=None, num_workers=4, pin_memory=True)
    
    model.train()
    start_time = time.time(); events_count = 0; batch_idx = 0
    for batch in loader:
        batch = add_features_gpu(batch.to(device, non_blocking=True))
        optimizer.zero_grad(set_to_none=True)
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out, batch.y)
        loss.backward(); optimizer.step()
        
        batch_idx += 1; events_count += len(batch.y)
        if batch_idx % 100 == 0:
            elapsed = time.time() - start_time
            print(f"Batch {batch_idx:05d} | Speed: {events_count/elapsed:.2f} events/s")

if __name__ == "__main__":
    train()
