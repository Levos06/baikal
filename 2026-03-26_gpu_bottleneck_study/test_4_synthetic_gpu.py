import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
import time

class JKResGCN_v2(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 512)
        self.conv2 = GCNConv(512, 512)
        self.conv3 = GCNConv(512, 512)
        self.conv4 = GCNConv(512, 512)
        self.conv5 = GCNConv(512, 512)
        self.proj1 = torch.nn.Linear(in_channels, 512)
        self.proj2 = torch.nn.Linear(512, 512)
        self.proj3 = torch.nn.Linear(512, 512)
        self.proj4 = torch.nn.Linear(512, 512)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(512 * 5, 1024),
            torch.nn.GELU(),
            torch.nn.Linear(1024, 512),
            torch.nn.GELU(),
            torch.nn.Linear(512, out_channels)
        )

    def forward(self, x, edge_index, batch):
        h1 = F.gelu(self.conv1(x, edge_index) + self.proj1(x))
        h2 = F.gelu(self.conv2(h1, edge_index) + self.proj2(h1))
        h3 = F.gelu(self.conv3(h2, edge_index) + self.proj3(h2))
        h4 = F.gelu(self.conv4(h3, edge_index) + self.proj4(h3))
        h5 = F.gelu(self.conv5(h4, edge_index))
        # Pooling to get 1 prediction per graph
        p1 = global_mean_pool(h1, batch)
        p2 = global_mean_pool(h2, batch)
        p3 = global_mean_pool(h3, batch)
        p4 = global_mean_pool(h4, batch)
        p5 = global_mean_pool(h5, batch)
        combined = torch.cat([p1, p2, p3, p4, p5], dim=1)
        return self.head(combined)

def generate_synthetic_batch(batch_size, nodes_per_graph=50, feat_dim=13):
    data_list = []
    for _ in range(batch_size):
        x = torch.randn(nodes_per_graph, feat_dim)
        edge_index = torch.stack([torch.arange(nodes_per_graph-1), torch.arange(1, nodes_per_graph)])
        data_list.append(Data(x=x, edge_index=edge_index, y=torch.tensor([0])))
    return Batch.from_data_list(data_list)

def run_test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = JKResGCN_v2(13, 2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    batch = generate_synthetic_batch(128).to(device)
    
    print(f"Starting Synthetic GPU Test on {device}...")
    for _ in range(5):
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out, batch.y); loss.backward(); optimizer.step(); optimizer.zero_grad()
    
    torch.cuda.synchronize(); start = time.time()
    iters = 100
    for _ in range(iters):
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out, batch.y); loss.backward(); optimizer.step(); optimizer.zero_grad()
    torch.cuda.synchronize()
    
    duration = time.time() - start
    tp = (iters * 128) / duration
    print(f"Max Theoretical Throughput: {tp:.2f} events/s")

if __name__ == "__main__":
    run_test()
