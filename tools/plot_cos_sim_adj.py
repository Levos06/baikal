import os
import sys
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
import importlib.util
from torch_geometric.data import Batch

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

mod = load_module("cos_sim", "/home/levos/experiments/2026-04-23_gat_cos_sim_beta07/train_cos_sim.py")
GATv2_CosSimBeta = mod.GATv2_CosSimBeta
add_features = mod.add_features_21_norm
get_similarity_edges = mod.get_similarity_edges

def get_events():
    data_dir = "/home/levos/experiments/data_processed_50k/val"
    files = sorted(glob.glob(os.path.join(data_dir, "*.pt")))
    events = []
    for f in files:
        data_list = torch.load(f, weights_only=False)
        for data in data_list:
            if 60 <= data.x.size(0) <= 80:
                events.append(data)
                if len(events) == 3:
                    return events
    return events

events = get_events()
# Re-create k=1 edge_index for these events as done in the dataset
for data in events:
    num_nodes = data.x.size(0)
    indices = np.arange(num_nodes)
    mask = (np.abs(indices[:, None] - indices) <= 1) & (indices[:, None] != indices)
    data.edge_index = torch.from_numpy(np.array(np.where(mask))).long()

batch = Batch.from_data_list(events).to(DEVICE)
batch = add_features(batch)

epochs = [100, 500, 1000]
fig, axes = plt.subplots(3, 3, figsize=(12, 12))
fig.suptitle("Adjacency Matrix Evolution (Layer 2, CosSim > 0.7 + k=1)", fontsize=16)

for col_idx, ep in enumerate(epochs):
    model = GATv2_CosSimBeta(21, 2).to(DEVICE)
    chkpt_path = f"/home/levos/experiments/2026-04-23_gat_cos_sim_beta07/checkpoints/model_cos_sim_beta07_{ep}.pt"
    model.load_state_dict(torch.load(chkpt_path, map_location=DEVICE, weights_only=True))
    model.eval()
    
    with torch.no_grad():
        h = batch.x
        # Layer 1
        h = torch.nn.functional.gelu(model.convs[0](h, batch.edge_index) + model.projs[0](h))
        # Layer 2 edges (Base + CosSim > 0.7)
        sim_edges = get_similarity_edges(h, batch.batch, threshold=0.7)
        curr_edge_index = torch.cat([batch.edge_index, sim_edges], dim=1)
        curr_edge_index = torch.unique(curr_edge_index, dim=1)
        
    # Plot for each event
    for row_idx in range(3):
        mask = (batch.batch == row_idx)
        nodes_in_event = mask.sum().item()
        start_idx = torch.where(mask)[0][0].item()
        
        # Filter edges for this event
        edge_mask = (curr_edge_index[0] >= start_idx) & (curr_edge_index[0] < start_idx + nodes_in_event)
        ev_edges = curr_edge_index[:, edge_mask] - start_idx
        
        adj = np.zeros((nodes_in_event, nodes_in_event))
        adj[ev_edges[0].cpu().numpy(), ev_edges[1].cpu().numpy()] = 1
        
        ax = axes[row_idx, col_idx]
        ax.imshow(adj, cmap="Blues", interpolation='nearest')
        if row_idx == 0:
            ax.set_title(f"Epoch {ep}")
        if col_idx == 0:
            ax.set_ylabel(f"Event {row_idx+1}\nNodes: {nodes_in_event}")
        ax.set_xticks([])
        ax.set_yticks([])

plt.tight_layout()
os.makedirs("/home/levos/experiments/plots/rewiring_evolution", exist_ok=True)
save_path = "/home/levos/experiments/plots/rewiring_evolution/CosSim_Beta07.png"
plt.savefig(save_path)
plt.close()
print(f"Saved {save_path}")
