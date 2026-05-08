import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.metrics import precision_recall_curve
import numpy as np
import os
import sys

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
NUM_WORKERS = 4

def fast_get_complete_edge_index(batch_vector):
    device = batch_vector.device
    counts = torch.bincount(batch_vector)
    edge_indices = []
    offset = 0
    for count in counts:
        if count > 0:
            nodes = torch.arange(offset, offset + count, device=device)
            row = nodes.repeat_interleave(count)
            col = nodes.repeat(count)
            edge_indices.append(torch.stack([row, col], dim=0))
            offset += count
    if not edge_indices:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    return torch.cat(edge_indices, dim=1)

# Import datasets and models
sys.path.append('/home/levos/experiments/2026-04-24_gat_cos_sim_beta07_k4')
import train_cos_sim_k4
train_cos_sim_k4.get_complete_edge_index = fast_get_complete_edge_index

sys.path.append('/home/levos/experiments/2026-04-23_gat_cos_sim_beta07')
import train_cos_sim
train_cos_sim.get_complete_edge_index = fast_get_complete_edge_index

sys.path.append('/home/levos/experiments/2026-04-22_gat_fc_last_k_scaling')
import train_fc_last_k8
train_fc_last_k8.get_complete_edge_index = fast_get_complete_edge_index
import train_fc_last_k12
train_fc_last_k12.get_complete_edge_index = fast_get_complete_edge_index
import train_fc_last_k16
train_fc_last_k16.get_complete_edge_index = fast_get_complete_edge_index

def eval_model(model, loader, add_features_func, num_batches=100):
    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= num_batches: break
            batch = add_features_func(batch.to(DEVICE))
            out = model(batch.x, batch.edge_index, batch.batch)
            all_probs.extend(F.softmax(out, dim=1)[:, 1].cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
    
    p, r, _ = precision_recall_curve(all_labels, all_probs)
    p_at_r095 = np.interp(0.95, r[::-1], p[::-1])
    return p_at_r095

def main():
    models_to_eval = [
        {
            "name": "GATv2 CosSim beta=0.7 + k=4",
            "model": train_cos_sim_k4.GATv2_CosSimBeta(21, 2).to(DEVICE),
            "ckpt": "/home/levos/experiments/2026-04-24_gat_cos_sim_beta07_k4/checkpoints/model_cos_sim_beta07_k4_900.pt",
            "dataset": train_cos_sim_k4.MediumDataset('val', k=4),
            "add_feat": train_cos_sim_k4.add_features_21_norm
        },
        {
            "name": "GATv2 CosSim beta=0.7",
            "model": train_cos_sim.GATv2_CosSimBeta(21, 2).to(DEVICE),
            "ckpt": "/home/levos/experiments/2026-04-23_gat_cos_sim_beta07/checkpoints/model_cos_sim_beta07_1000.pt",
            "dataset": train_cos_sim.MediumDataset('val', k=1),
            "add_feat": train_cos_sim.add_features_21_norm
        },
        {
            "name": "GATv2 FC-Last-Layer + k=16",
            "model": train_fc_last_k16.GATv2_FCLast(21, 2).to(DEVICE),
            "ckpt": "/home/levos/experiments/2026-04-22_gat_fc_last_k_scaling/checkpoints/model_fc_last_k16_1000.pt",
            "dataset": train_fc_last_k16.MediumDataset('val', k=16),
            "add_feat": train_fc_last_k16.add_features_21_norm
        },
        {
            "name": "GATv2 FC-Last-Layer + k=12",
            "model": train_fc_last_k12.GATv2_FCLast(21, 2).to(DEVICE),
            "ckpt": "/home/levos/experiments/2026-04-22_gat_fc_last_k_scaling/checkpoints/model_fc_last_k12_1000.pt",
            "dataset": train_fc_last_k12.MediumDataset('val', k=12),
            "add_feat": train_fc_last_k12.add_features_21_norm
        },
        {
            "name": "GATv2 FC-Last-Layer + k=8",
            "model": train_fc_last_k8.GATv2_FCLast(21, 2).to(DEVICE),
            "ckpt": "/home/levos/experiments/2026-04-22_gat_fc_last_k_scaling/checkpoints/model_fc_last_k8_1000.pt",
            "dataset": train_fc_last_k8.MediumDataset('val', k=8),
            "add_feat": train_fc_last_k8.add_features_21_norm
        }
    ]
    
    for m in models_to_eval:
        print(f"Evaluating {m['name']}...")
        m['model'].load_state_dict(torch.load(m['ckpt'], map_location=DEVICE, weights_only=True))
        loader = DataLoader(m['dataset'], batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
        p95 = eval_model(m['model'], loader, m['add_feat'], num_batches=200)
        print(f"  -> P@R0.95: {p95:.4f}")

if __name__ == "__main__":
    main()