import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import os
import time
import glob
import random
import sys

# --- SILENCE SYSTEM NOISE ---
class StderrFilter:
    def __init__(self, stream):
        self.stream = stream
    def write(self, data):
        if "GeneratorExit" not in data and "RuntimeError: generator ignored" not in data:
            self.stream.write(data)
            self.stream.flush()
    def flush(self):
        self.stream.flush()

sys.stderr = StderrFilter(sys.stderr)

# --- CONFIGURATION ---
DATA_DIR = "data_processed_50k"
PROJECT_DIR = "2026-03-28_pure_training_speed_test"
BATCH_SIZE = 512
NUM_WORKERS = 4
TOTAL_EPOCHS = 1500
VIRTUAL_EPOCH_SIZE = 200
C_WATER = 0.225

os.makedirs(f"{PROJECT_DIR}/checkpoints", exist_ok=True)

# --- UTILS ---
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
    def forward(self, x, edge_index):
        h1 = F.gelu(self.conv1(x, edge_index) + self.proj1(x))
        h2 = F.gelu(self.conv2(h1, edge_index) + self.proj2(h1))
        h3 = F.gelu(self.conv3(h2, edge_index) + self.proj3(h2))
        h4 = F.gelu(self.conv4(h3, edge_index) + self.proj4(h3))
        h5 = F.gelu(self.conv5(h4, edge_index))
        combined = torch.cat([h1, h2, h3, h4, h5], dim=1)
        return self.head(combined)

class MediumDataset(torch.utils.data.IterableDataset):
    def __init__(self, split='train'):
        super().__init__()
        self.split = split
        self.data_dir = os.path.join(DATA_DIR, split)
    def __iter__(self):
        try:
            while True:
                files = sorted(glob.glob(os.path.join(self.data_dir, "*.pt")))
                if not files: 
                    time.sleep(2); continue
                random.shuffle(files)
                for f in files:
                    try:
                        data_list = torch.load(f, weights_only=False)
                        random.shuffle(data_list)
                        for data in data_list: yield data
                    except GeneratorExit: return
                    except: continue
        except GeneratorExit: return

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting PURE SPEED MARATHON on {device} (1500 Epochs, BS=512)")
    
    model = JKResGCN_v2(13, 2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    train_loader = DataLoader(MediumDataset('train'), batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    
    epoch = 1
    start_time = time.time()
    model.train()
    
    for i, batch in enumerate(train_loader):
        batch = add_features_gpu(batch.to(device, non_blocking=True))
        optimizer.zero_grad(set_to_none=True)
        out = model(batch.x, batch.edge_index)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        
        if (i + 1) % VIRTUAL_EPOCH_SIZE == 0:
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch:04d} | Time: {epoch_time:.1f}s | Loss: {loss.item():.4f}")
            
            if epoch % 100 == 0:
                torch.save(model.state_dict(), f"{PROJECT_DIR}/checkpoints/model_speed_{epoch}.pt")
            
            epoch += 1
            if epoch > TOTAL_EPOCHS: break
            start_time = time.time() # Reset timer for the next epoch
            model.train()

if __name__ == "__main__":
    train()
