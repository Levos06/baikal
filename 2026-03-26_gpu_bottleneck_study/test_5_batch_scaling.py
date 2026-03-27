import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
import time

class JKResGCN_v2(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 512); self.conv2 = GCNConv(512, 512); self.conv3 = GCNConv(512, 512)
        self.conv4 = GCNConv(512, 512); self.conv5 = GCNConv(512, 512)
        self.proj1 = torch.nn.Linear(in_channels, 512); self.proj2 = torch.nn.Linear(512, 512)
        self.proj3 = torch.nn.Linear(512, 512); self.proj4 = torch.nn.Linear(512, 512)
        self.head = torch.nn.Sequential(torch.nn.Linear(512*5, 1024), torch.nn.GELU(), torch.nn.Linear(1024, 512), torch.nn.GELU(), torch.nn.Linear(512, out_channels))

    def forward(self, x, edge_index, batch):
        h1 = F.gelu(self.conv1(x, edge_index) + self.proj1(x))
        h2 = F.gelu(self.conv2(h1, edge_index) + self.proj2(h1))
        h3 = F.gelu(self.conv3(h2, edge_index) + self.proj3(h2))
        h4 = F.gelu(self.conv4(h3, edge_index) + self.proj4(h3))
        h5 = F.gelu(self.conv5(h4, edge_index))
        combined = torch.cat([global_mean_pool(h, batch) for h in [h1,h2,h3,h4,h5]], dim=1)
        return self.head(combined)

def run_test(bs):
    device = torch.device('cuda'); model = JKResGCN_v2(13, 2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4); criterion = torch.nn.CrossEntropyLoss()
    # Synthetic batch
    data_list = [Data(x=torch.randn(50, 13), edge_index=torch.tensor([[0],[1]]), y=torch.tensor([0])) for _ in range(bs)]
    batch = Batch.from_data_list(data_list).to(device)
    torch.cuda.synchronize(); start = time.time(); iters = 50
    for _ in range(iters):
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out, batch.y); loss.backward(); optimizer.step(); optimizer.zero_grad()
    torch.cuda.synchronize()
    return (iters * bs) / (time.time() - start)

if __name__ == "__main__":
    for bs in [128, 256, 512, 1024]:
        try:
            tp = run_test(bs)
            print(f"BS {bs:4d}: {tp:8.2f} events/s")
        except Exception as e: print(f"BS {bs} failed: {e}")
