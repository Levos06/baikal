import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATv2Conv, GATConv

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# --- ARCHITECTURES ---

# 1. Base GCN (128 units, 3 layers) - e.g. GCN_20260303_REPRO
class GCN_Base(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.c1 = GCNConv(in_c, 128); self.c2 = GCNConv(128, 128); self.c3 = GCNConv(128, 2)
    def forward(self, x, e): pass

# 2. Capacity GCN (512 units, 3 layers) - GCN_20260308_CAPACITY_512
class GCN_Cap(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.c1 = GCNConv(in_c, 512); self.c2 = GCNConv(512, 512); self.c3 = GCNConv(512, 2)
    def forward(self, x, e): pass

# 3. Deep Wide GCN (512-768-512, 4 layers) - GCN_20260308_DEEP_WIDE_V3
class GCN_DW(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.c1 = GCNConv(in_c, 512); self.c2 = GCNConv(512, 768); self.c3 = GCNConv(768, 512); self.c4 = GCNConv(512, 2)
    def forward(self, x, e): pass

# 4. JK-ResGCN (512 units, 5 layers + Proj + MLP Head) - JKResGCN_20260328_21F / 27_13F / 25_FULLMLP
class JKResGCN(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.convs = nn.ModuleList([GCNConv(in_c, 512)] + [GCNConv(512, 512) for _ in range(4)])
        self.projs = nn.ModuleList([nn.Linear(in_c, 512)] + [nn.Linear(512, 512) for _ in range(3)])
        self.head = nn.Sequential(nn.Linear(512*5, 1024), nn.GELU(), nn.Linear(1024, 512), nn.GELU(), nn.Linear(512, 2))
    def forward(self, x, e): pass

# 5. CNN Baseline
class CNN1D(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.c1 = nn.Conv1d(in_c, 128, 3, padding=1); self.c2 = nn.Conv1d(128, 256, 5, padding=2)
        self.c3 = nn.Conv1d(256, 128, 3, padding=1); self.fc = nn.Linear(128, 2)
    def forward(self, x): pass

# 6. GATv2 Marathon (512 units total, 4 layers, JK, Res, MLP Head)
class GATv2_Marathon(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.c1 = GATv2Conv(in_c, 128, heads=4); self.p1 = nn.Linear(in_c, 512)
        self.c2 = GATv2Conv(512, 128, heads=4); self.c3 = GATv2Conv(512, 128, heads=4); self.c4 = GATv2Conv(512, 128, heads=4)
        self.head = nn.Sequential(nn.Linear(512*4, 1024), nn.GELU(), nn.Linear(1024, 512), nn.GELU(), nn.Linear(512, 2))
    def forward(self, x, e): pass

# 7. GATv2 Deep (256 units total, 6 layers, JK, Res, MLP Head)
class GATv2_Deep(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.c1 = GATv2Conv(in_c, 64, heads=4); self.p1 = nn.Linear(in_c, 256)
        self.convs = nn.ModuleList([GATv2Conv(256, 64, heads=4) for _ in range(5)])
        self.head = nn.Sequential(nn.Linear(256*6, 1024), nn.GELU(), nn.Linear(1024, 512), nn.GELU(), nn.Linear(512, 2))
    def forward(self, x, e): pass

# 8. GATv2 Wide (1024 units total, 3 layers, JK, Res, MLP Head)
class GATv2_Wide(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.c1 = GATv2Conv(in_c, 128, heads=8); self.p1 = nn.Linear(in_c, 1024)
        self.c2 = GATv2Conv(1024, 128, heads=8); self.c3 = GATv2Conv(1024, 128, heads=8)
        self.head = nn.Sequential(nn.Linear(1024*3, 1024), nn.GELU(), nn.Linear(1024, 512), nn.GELU(), nn.Linear(512, 2))
    def forward(self, x, e): pass

# 9. GATv2 Res Baseline (256 units, 4 layers, NO JK)
class GATv2_Baseline(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.c1 = GATv2Conv(in_c, 64, heads=4); self.p1 = nn.Linear(in_c, 256)
        self.convs = nn.ModuleList([GATv2Conv(256, 64, heads=4) for _ in range(3)])
        self.head = nn.Sequential(nn.Linear(256, 512), nn.GELU(), nn.Linear(512, 256), nn.GELU(), nn.Linear(256, 2))
    def forward(self, x, e): pass

# 10. GATv1 Res Baseline
class GATv1_Baseline(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.c1 = GATConv(in_c, 64, heads=4); self.p1 = nn.Linear(in_c, 256)
        self.convs = nn.ModuleList([GATConv(256, 64, heads=4) for _ in range(3)])
        self.head = nn.Sequential(nn.Linear(256, 512), nn.GELU(), nn.Linear(512, 256), nn.GELU(), nn.Linear(256, 2))
    def forward(self, x, e): pass

# 11. Wide Last Layer (1536 last)
class GATv2_WideLast(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.c1 = GATv2Conv(in_c, 64, heads=4); self.p1 = nn.Linear(in_c, 256)
        self.c2 = GATv2Conv(256, 64, heads=4); self.c3 = GATv2Conv(256, 64, heads=4)
        self.c4 = GATv2Conv(256, 256, heads=6); self.p4 = nn.Linear(256, 1536)
        self.head = nn.Sequential(nn.Linear(1536, 1024), nn.GELU(), nn.Linear(1024, 512), nn.GELU(), nn.Linear(512, 2))
    def forward(self, x, e): pass

# 12. GCN 5-Layer Basics (128-512-512-512-128)
class GCN_5L(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.c1 = GCNConv(in_c, 128); self.c2 = GCNConv(128, 512); self.c3 = GCNConv(512, 512)
        self.c4 = GCNConv(512, 512); self.c5 = GCNConv(512, 128); self.fc = nn.Linear(128, 2)
    def forward(self, x, e): pass

print(f"GCN_REPRO (5 in): {count_params(GCN_Base(5))}")
print(f"GCN_CAP_512 (5 in): {count_params(GCN_Cap(5))}")
print(f"GCN_DEEP_WIDE (5 in): {count_params(GCN_DW(5))}")
print(f"JKResGCN_13F: {count_params(JKResGCN(13))}")
print(f"JKResGCN_21F: {count_params(JKResGCN(21))}")
print(f"CNN_BASELINE: {count_params(CNN1D(5))}")
print(f"GATv2_MARATHON: {count_params(GATv2_Marathon(21))}")
print(f"GATv2_DEEP_6L: {count_params(GATv2_Deep(21))}")
print(f"GATv2_WIDE_3L: {count_params(GATv2_Wide(21))}")
print(f"GATv2_RES_4L: {count_params(GATv2_Baseline(21))}")
print(f"GATv1_RES_4L: {count_params(GATv1_Baseline(21))}")
print(f"GATv2_WIDELAST: {count_params(GATv2_WideLast(21))}")
print(f"GCN_5L_BASICS: {count_params(GCN_5L(5))}")
