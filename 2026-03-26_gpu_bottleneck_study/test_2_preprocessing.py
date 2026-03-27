import torch
from torch_geometric.loader import DataLoader
import os
import time
import glob
import math

DATA_DIR = "data_processed"
BATCH_SIZE = 128
C_WATER = 0.225

def add_extended_features_vectorized(batch):
    x = batch.x
    ptr = batch.ptr
    first_hit_indices = ptr[:-1]
    graph_sizes = ptr[1:] - ptr[:-1]
    t0 = torch.repeat_interleave(x[first_hit_indices, 1], graph_sizes)
    x0 = torch.repeat_interleave(x[first_hit_indices, 2], graph_sizes)
    y0 = torch.repeat_interleave(x[first_hit_indices, 3], graph_sizes)
    z0 = torch.repeat_interleave(x[first_hit_indices, 4], graph_sizes)
    dt = x[:, 1] - t0
    dx, dy, dz = x[:, 2]-x0, x[:, 3]-y0, x[:, 4]-z0
    dr2 = dx**2 + dy**2 + dz**2
    dr = torch.sqrt(dr2 + 1e-8)
    s2 = (C_WATER * dt)**2 - dr2
    r = torch.sqrt(x[:, 2]**2 + x[:, 3]**2 + 1e-8)
    phi = torch.atan2(x[:, 3], x[:, 2])
    rho = torch.sqrt(x[:, 2]**2 + x[:, 3]**2 + x[:, 4]**2 + 1e-8)
    cosTheta = x[:, 4] / (rho + 1e-8)
    tof_res = dt - dr/C_WATER
    ext = torch.stack([s2, dt, dr, r, phi, rho, cosTheta, tof_res], dim=1)
    batch.x = torch.cat([x, ext], dim=1)
    return batch

class ChunkedDataset(torch.utils.data.IterableDataset):
    def __init__(self, split='train'):
        super().__init__()
        self.files = sorted(glob.glob(os.path.join(DATA_DIR, split, "chunk_*.pt")))[:100]
    def __iter__(self):
        for file in self.files:
            data_list = torch.load(file, weights_only=False)
            for data in data_list: yield data

def run_test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = ChunkedDataset()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)
    
    print(f"Starting Preprocessing Test on {device}...")
    start = time.time()
    count = 0
    for batch in loader:
        batch = add_extended_features_vectorized(batch.to(device))
        count += batch.num_graphs
        if count >= 100000: break
    
    duration = time.time() - start
    print(f"Processed {count} events with features in {duration:.2f}s")
    print(f"Throughput: {count/duration:.2f} events/s")

if __name__ == "__main__":
    run_test()
