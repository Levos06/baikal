import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader
from torch_geometric.utils import scatter
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from tqdm import tqdm

# --- CONFIG ---
DATA_DIR = "data_processed_50k/val"
CHECKPOINT = "2026-03-28_extended_21features/checkpoints/model_21feat_norm_2500.pt"
STATS_21 = torch.load('tools/feat_stats_21.pt', weights_only=True)
C_WATER = 0.225
BATCH_SIZE = 512

FEATURE_NAMES = [
    "0: Charge", "1: Time", "2: X", "3: Y", "4: Z",
    "5: Minkowski s2", "6: dt", "7: dr", "8: r_xy", "9: phi",
    "10: rho", "11: cosTheta", "12: ToF Res", "13: NeighDist",
    "14: NeighQ", "15: Q/Mean", "16: cosAlpha", "17: StrHits",
    "18: StrZSpan", "19: EventNhits", "20: Duration"
]

class JKResGCN_21(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(21, 512); self.conv2 = GCNConv(512, 512); self.conv3 = GCNConv(512, 512)
        self.conv4 = GCNConv(512, 512); self.conv5 = GCNConv(512, 512)
        self.proj1 = torch.nn.Linear(21, 512); self.proj2 = torch.nn.Linear(512, 512)
        self.proj3 = torch.nn.Linear(512, 512); self.proj4 = torch.nn.Linear(512, 512)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(512 * 5, 1024), torch.nn.GELU(), torch.nn.Dropout(0.1),
            torch.nn.Linear(1024, 512), torch.nn.GELU(), torch.nn.Dropout(0.1),
            torch.nn.Linear(512, 2)
        )
    def forward(self, x, edge_index):
        h1 = F.gelu(self.conv1(x, edge_index) + self.proj1(x))
        h2 = F.gelu(self.conv2(h1, edge_index) + self.proj2(h1))
        h3 = F.gelu(self.conv3(h2, edge_index) + self.proj3(h2))
        h4 = F.gelu(self.conv4(h3, edge_index) + self.proj4(h3))
        h5 = F.gelu(self.conv5(h4, edge_index))
        combined = torch.cat([h1, h2, h3, h4, h5], dim=1)
        return self.head(combined)

def add_features_21(batch):
    device = batch.x.device
    x, edge_index, ptr, b_idx = batch.x, batch.edge_index, batch.ptr, batch.batch
    num_nodes = x.size(0)
    first = ptr[:-1]; sizes = ptr[1:] - ptr[:-1]
    t0, x0, y0, z0 = [torch.repeat_interleave(x[first, i], sizes) for i in [1,2,3,4]]
    dt, dx, dy, dz = x[:,1]-t0, x[:,2]-x0, x[:,3]-y0, x[:,4]-z0
    dr2 = dx**2 + dy**2 + dz**2; dr = torch.sqrt(dr2 + 1e-8)
    s2, tof = (C_WATER * dt)**2 - dr2, dt - dr/C_WATER
    r_xy, phi = torch.sqrt(x[:, 2]**2 + x[:, 3]**2 + 1e-8), torch.atan2(x[:, 3], x[:, 2])
    rho = torch.sqrt(x[:, 2]**2 + x[:, 3]**2 + x[:, 4]**2 + 1e-8)
    cosTheta = x[:, 4] / (rho + 1e-8)
    row, col = edge_index
    dist_edges = torch.sqrt(torch.sum((x[row, 2:5] - x[col, 2:5])**2, dim=1) + 1e-8)
    mean_dist_neigh = scatter(dist_edges, row, dim=0, dim_size=num_nodes, reduce='mean')
    neigh_charge = scatter(x[col, 0], row, dim=0, dim_size=num_nodes, reduce='sum')
    event_mean_q = scatter(x[:, 0], b_idx, dim=0, reduce='mean')
    q_rel_mean = x[:, 0] / (torch.gather(event_mean_q, 0, b_idx) + 1e-8)
    cos_alpha = dz / (dr + 1e-8)
    x_bin, y_bin = torch.round(x[:, 2] / 0.03), torch.round(x[:, 3] / 0.03)
    xy_bins = torch.stack([b_idx, x_bin, y_bin], dim=1)
    _, str_ids = torch.unique(xy_bins, dim=0, return_inverse=True)
    hits_on_string = scatter(torch.ones_like(x[:, 0]), str_ids, dim=0, reduce='sum')[str_ids]
    max_z, min_z = scatter(x[:, 4], str_ids, dim=0, reduce='max')[str_ids], scatter(x[:, 4], str_ids, dim=0, reduce='min')[str_ids]
    z_span = max_z - min_z
    event_n_hits = sizes.float(); n_hits = torch.gather(event_n_hits, 0, b_idx)
    duration = torch.gather(scatter(x[:, 1], b_idx, dim=0, reduce='max') - scatter(x[:, 1], b_idx, dim=0, reduce='min'), 0, b_idx)
    raw_extra = torch.stack([s2, dt, dr, r_xy, phi, rho, cosTheta, tof, mean_dist_neigh, neigh_charge, q_rel_mean, cos_alpha, hits_on_string, z_span, n_hits, duration], dim=1)
    norm_extra = (raw_extra - STATS_21['means'].to(device)) / (STATS_21['stds'].to(device) + 1e-8)
    return torch.cat([x, norm_extra], dim=1)

def run_evaluation(model, device, loader, shuffle_idx=None):
    all_labels, all_probs = [], []
    with torch.no_grad():
        for batch in loader:
            x_f = add_features_21(batch.to(device))
            if shuffle_idx is not None:
                x_f = x_f.clone()
                x_f[:, shuffle_idx] = x_f[torch.randperm(x_f.size(0)), shuffle_idx]
            out = model(x_f, batch.edge_index)
            all_probs.extend(F.softmax(out, dim=1)[:, 1].cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
    p, r, _ = precision_recall_curve(all_labels, all_probs)
    return np.interp(0.9, r[::-1], p[::-1])

def gradient_importance(model, device, loader):
    print("Calculating gradient-based importance...")
    grad_acc = torch.zeros(21).to(device)
    model.train() # Enable gradients
    for i, batch in enumerate(loader):
        batch = batch.to(device)
        x_f = add_features_21(batch)
        x_f.requires_grad = True
        out = model(x_f, batch.edge_index)
        prob = torch.sigmoid(out[:, 1]).mean()
        prob.backward()
        grad_acc += x_f.grad.abs().mean(dim=0)
    return (grad_acc / len(loader)).cpu().numpy()

if __name__ == "__main__":
    device = torch.device('cuda')
    model = JKResGCN_21().to(device)
    model.load_state_dict(torch.load(CHECKPOINT, weights_only=True))
    
    val_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.pt")))
    data_list = []
    for f in val_files: data_list.extend(torch.load(f, weights_only=False))
    loader = DataLoader(data_list, batch_size=512)
    
    print(f"Deep Importance Analysis for BEST SAVED model: {os.path.basename(CHECKPOINT)}")
    baseline = run_evaluation(model, device, loader)
    print(f"Baseline P@R0.9: {baseline:.4f}")
    
    # 1. Permutation Importance
    perms = []
    for i in range(21):
        p9 = run_evaluation(model, device, loader, shuffle_idx=i)
        perms.append(baseline - p9)
        print(f"Permuted {FEATURE_NAMES[i]:15s} | Drop: {perms[-1]:.4f}")
        
    # 2. Gradient Importance
    grads = gradient_importance(model, device, loader)
    
    # Save results to TXT
    results_txt = "2026-03-28_extended_21features/plots/importance_final_report_2500.txt"
    with open(results_txt, 'w') as f:
        f.write(f"BEST MODEL AT EPOCH 2500\nBaseline P@R0.9: {baseline:.4f}\n\n")
        f.write("Feature | Permutation Drop | Gradient Sensitivity\n")
        f.write("-" * 50 + "\n")
        for i in range(21):
            f.write(f"{FEATURE_NAMES[i]:15s} | {perms[i]:.6f} | {grads[i]:.6f}\n")

    # 3. Plotting
    plt.figure(figsize=(16, 10))
    plt.subplot(1, 2, 1)
    idx = np.argsort(perms)
    plt.barh(range(21), [perms[i] for i in idx], color='darkblue')
    plt.yticks(range(21), [FEATURE_NAMES[i] for i in idx])
    plt.title('Global Contribution (Permutation)')
    plt.xlabel('Drop in P@R0.9')

    plt.subplot(1, 2, 2)
    idx_g = np.argsort(grads)
    plt.barh(range(21), [grads[i] for i in idx_g], color='darkred')
    plt.yticks(range(21), [FEATURE_NAMES[i] for i in idx_g])
    plt.title('Feature Sensitivity (Gradients)')
    plt.xlabel('Mean |Grad|')

    plt.suptitle(f'Deep Feature Analysis: model_2500.pt (Peak Val: {baseline:.4f})', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('2026-03-28_extended_21features/plots/importance_best_comparison.png')
    print(f"\nAnalysis complete. Plot saved to plots/importance_best_comparison.png")
    print(f"Text report saved to {results_txt}")
