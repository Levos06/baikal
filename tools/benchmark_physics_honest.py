import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader
import time
import glob
import os

DATA_DIR = "data_processed_50k/train"
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

def mock_add_features_fast(batch):
    # Just pad with zeros to keep model input at 13 channels
    # but WITHOUT expensive physical calculations
    zeros = torch.zeros((batch.x.size(0), 8), device=batch.x.device)
    batch.x = torch.cat([batch.x, zeros], dim=1)
    return batch

class HonestModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Fixed 13 channels for BOTH tests
        self.conv = GCNConv(13, 512)
        self.head = torch.nn.Linear(512, 2)
    def forward(self, x, edge_index):
        return self.head(F.relu(self.conv(x, edge_index)))

def run_honest_test(mode="physics"):
    device = torch.device('cuda')
    model = HonestModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Pre-load to RAM to isolate GPU/Kernel speed from Disk I/O
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.pt")))[:10]
    data_list = []
    for f in files: data_list.extend(torch.load(f, weights_only=False))
    loader = DataLoader(data_list, batch_size=BATCH_SIZE)
    
    torch.cuda.synchronize()
    start = time.time()
    events = 0
    
    for batch in loader:
        batch = batch.to(device)
        if mode == "physics":
            batch = add_features_gpu(batch)
        else:
            batch = mock_add_features_fast(batch)
        
        optimizer.zero_grad(set_to_none=True)
        out = model(batch.x, batch.edge_index)
        loss = F.cross_entropy(out, batch.y)
        loss.backward()
        optimizer.step()
        events += batch.num_graphs
        
    torch.cuda.synchronize()
    return events / (time.time() - start)

if __name__ == "__main__":
    print("Starting HONEST Physics Overhead Test...")
    speed_phys = run_honest_test("physics")
    print(f"Speed with REAL physics:  {speed_phys:.2f} events/s")
    
    speed_mock = run_honest_test("mock")
    print(f"Speed with MOCK padding: {speed_mock:.2f} events/s")
    
    overhead = (1 - speed_phys/speed_mock) * 100
    print(f"\nPure function overhead: {overhead:.2f}%")
