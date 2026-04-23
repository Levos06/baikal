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

mod = load_module("k4", "/home/levos/experiments/2026-04-15_gat_fc_last_k4/train_fc_last_k4.py")
GATv2_FCLast = mod.GATv2_FCLast
add_features = mod.add_features_21_norm

# Patch forward to return h
def patched_forward(self, x, edge_index, batch_vector):
    import torch.nn.functional as F
    h = x
    # First 3 layers (Local)
    for i in range(3):
        h = F.gelu(self.convs[i](h, edge_index) + self.projs[i](h))
    
    # Last layer (Global / Fully Connected)
    fc_edge_index = mod.get_complete_edge_index(batch_vector)
    h = F.gelu(self.fc_conv(h, fc_edge_index) + h) # Residual skip
    
    # return h before classification head
    return h

GATv2_FCLast.forward = patched_forward

# Get 5 random events
def get_events():
    data_dir = "/home/levos/experiments/data_processed_50k/val"
    files = sorted(glob.glob(os.path.join(data_dir, "*.pt")))
    events = []
    for f in files:
        data_list = torch.load(f, weights_only=False)
        for data in data_list:
            if 60 <= data.x.size(0) <= 80:
                events.append(data)
                if len(events) == 5:
                    return events
    return events

events = get_events()
batch = Batch.from_data_list(events).to(DEVICE)
batch = add_features(batch)

epochs = [100, 500, 1000]

os.makedirs("/home/levos/experiments/plots", exist_ok=True)

for ep in epochs:
    print(f"Processing Epoch {ep}...")
    model = GATv2_FCLast(21, 2).to(DEVICE)
    chkpt_path = f"/home/levos/experiments/2026-04-15_gat_fc_last_k4/checkpoints/model_fc_last_k4_{ep}.pt"
    
    # We are loading the state_dict, but our patched model has the exact same parameters
    model.load_state_dict(torch.load(chkpt_path, map_location=DEVICE, weights_only=True))
    model.eval()
    
    with torch.no_grad():
        h_batch = model(batch.x, batch.edge_index, batch.batch)
        
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    fig.suptitle(f"Cosine Similarity of Node Representations (Epoch {ep})", fontsize=16)
    
    for i in range(5):
        mask = (batch.batch == i)
        h_event = h_batch[mask]
        
        # Normalize
        h_norm = h_event / (h_event.norm(dim=1, keepdim=True) + 1e-8)
        # Cosine similarity
        cos_sim = torch.mm(h_norm, h_norm.t()).cpu().numpy()
        
        ax = axes[i]
        im = ax.imshow(cos_sim, cmap="viridis", vmin=-1, vmax=1)
        ax.set_title(f"Event {i+1} (Nodes: {h_event.size(0)})")
        ax.set_xticks([])
        ax.set_yticks([])
        
        if i == 4:
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
    plt.tight_layout()
    save_path = f"/home/levos/experiments/plots/representations_cos_sim_epoch_{ep}.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Saved {save_path}")

print("Done.")