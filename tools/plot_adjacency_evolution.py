import os
import sys
import glob
import torch
import importlib.util
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import Batch

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

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
node_counts = [e.x.size(0) for e in events]

exps = [
    {
        "name": "Dynamic_Edges_k2",
        "path": "/home/levos/experiments/2026-04-19_gat_dynamic_edges_k2/train_dynamic_k2.py",
        "cls": "GATv2_DynamicEdges",
        "chkpt_dir": "/home/levos/experiments/2026-04-19_gat_dynamic_edges_k2/checkpoints",
        "chkpt_prefix": "model_dynamic_",
        "rewire_fn": "rewire_edges"
    },
    {
        "name": "Dynamic_Input_k2",
        "path": "/home/levos/experiments/2026-04-19_gat_dynamic_input_k2/train_dynamic_input.py",
        "cls": "GATv2_DynamicInput",
        "chkpt_dir": "/home/levos/experiments/2026-04-19_gat_dynamic_input_k2/checkpoints",
        "chkpt_prefix": "model_dynamic_input_",
        "rewire_fn": "rewire_edges"
    },
    {
        "name": "Diff_Weights",
        "path": "/home/levos/experiments/2026-04-19_gat_differentiable_rewiring/train_differentiable_weights.py",
        "cls": "GATv2_DiffWeights",
        "chkpt_dir": "/home/levos/experiments/2026-04-19_gat_differentiable_rewiring/checkpoints",
        "chkpt_prefix": "model_weights_",
        "rewire_fn": "rewire_edges_weights"
    },
    {
        "name": "Diff_STE",
        "path": "/home/levos/experiments/2026-04-19_gat_differentiable_rewiring/train_differentiable_ste.py",
        "cls": "GATv2_DiffSTE",
        "chkpt_dir": "/home/levos/experiments/2026-04-19_gat_differentiable_rewiring/checkpoints",
        "chkpt_prefix": "model_ste_",
        "rewire_fn": "rewire_edges_ste"
    },
    {
        "name": "Diff_Gumbel",
        "path": "/home/levos/experiments/2026-04-19_gat_differentiable_rewiring/train_differentiable_gumbel.py",
        "cls": "GATv2_DiffGumbel",
        "chkpt_dir": "/home/levos/experiments/2026-04-19_gat_differentiable_rewiring/checkpoints",
        "chkpt_prefix": "model_gumbel_",
        "rewire_fn": "rewire_edges_gumbel"
    },
    {
        "name": "Custom_Weights",
        "path": "/home/levos/experiments/2026-04-19_gat_custom_rewiring/train_custom_weights.py",
        "cls": "CustomGAT_DiffWeights",
        "chkpt_dir": "/home/levos/experiments/2026-04-19_gat_custom_rewiring/checkpoints",
        "chkpt_prefix": "model_custom_weights_",
        "rewire_fn": "rewire_edges_custom_weights"
    },
    {
        "name": "Custom_STE",
        "path": "/home/levos/experiments/2026-04-19_gat_custom_rewiring/train_custom_ste.py",
        "cls": "CustomGAT_DiffSTE",
        "chkpt_dir": "/home/levos/experiments/2026-04-19_gat_custom_rewiring/checkpoints",
        "chkpt_prefix": "model_custom_ste_",
        "rewire_fn": "rewire_edges_custom_ste"
    },
    {
        "name": "Custom_Gumbel",
        "path": "/home/levos/experiments/2026-04-19_gat_custom_rewiring/train_custom_gumbel.py",
        "cls": "CustomGAT_DiffGumbel",
        "chkpt_dir": "/home/levos/experiments/2026-04-19_gat_custom_rewiring/checkpoints",
        "chkpt_prefix": "model_custom_gumbel_",
        "rewire_fn": "rewire_edges_custom_gumbel"
    }
]

os.makedirs("/home/levos/experiments/plots/rewiring_evolution", exist_ok=True)

captured_edge_index = None
captured_edge_weights = None

for exp in exps:
    print(f"Processing {exp['name']}...")
    try:
        mod = load_module(exp['name'], exp['path'])
        ModelClass = getattr(mod, exp['cls'])
        
        # Monkey patch the rewire function
        original_rewire = getattr(mod, exp['rewire_fn'])
        
        def patched_rewire(*args, **kwargs):
            global captured_edge_index, captured_edge_weights
            res = original_rewire(*args, **kwargs)
            if isinstance(res, tuple):
                captured_edge_index, captured_edge_weights = res[0].detach().cpu(), res[1].detach().cpu()
            else:
                captured_edge_index = res.detach().cpu()
                captured_edge_weights = None
            return res
            
        setattr(mod, exp['rewire_fn'], patched_rewire)
        
        add_features = mod.add_features_21_norm
        batch = Batch.from_data_list(events).to(DEVICE)
        batch = add_features(batch)
        
        chkpts = sorted(glob.glob(os.path.join(exp['chkpt_dir'], f"{exp['chkpt_prefix']}*.pt")), key=lambda x: int(x.split('_')[-1].split('.')[0]))
        if not chkpts:
            print(f"No checkpoints found for {exp['name']}")
            continue
            
        n_epochs = len(chkpts)
        fig, axes = plt.subplots(3, n_epochs, figsize=(3 * n_epochs, 9))
        fig.suptitle(f"Evolution of Adjacency Matrices - {exp['name']}", fontsize=16)
        
        for j, chkpt in enumerate(chkpts):
            epoch = int(chkpt.split('_')[-1].split('.')[0])
            
            model = ModelClass(21, 2).to(DEVICE)
            model.load_state_dict(torch.load(chkpt, map_location=DEVICE))
            model.eval()
            model.current_epoch = epoch 
            
            captured_edge_index = None
            captured_edge_weights = None
            with torch.no_grad():
                model(batch.x, batch.edge_index, batch.batch)
                
            if captured_edge_index is None:
                captured_edge_index = batch.edge_index.cpu()
                
            for i in range(3):
                node_start = sum(node_counts[:i])
                node_end = node_start + node_counts[i]
                
                mask = (captured_edge_index[0] >= node_start) & (captured_edge_index[0] < node_end)
                ev_edges = captured_edge_index[:, mask] - node_start
                
                adj = np.zeros((node_counts[i], node_counts[i]))
                if captured_edge_weights is not None:
                    ev_weights = captured_edge_weights[mask].squeeze().numpy()
                    for e, w in zip(ev_edges.T, ev_weights):
                        adj[e[0], e[1]] = w
                else:
                    for e in ev_edges.T:
                        adj[e[0], e[1]] = 1.0
                        
                ax = axes[i, j] if n_epochs > 1 else axes[i]
                ax.imshow(adj, cmap="viridis", vmin=0, vmax=1)
                if i == 0:
                    ax.set_title(f"Ep {epoch}")
                ax.set_xticks([])
                ax.set_yticks([])
                
        plt.tight_layout()
        plt.savefig(f"/home/levos/experiments/plots/rewiring_evolution/{exp['name']}.png")
        plt.close()
        print(f"Saved plot for {exp['name']}")
        
    except Exception as e:
        print(f"Failed {exp['name']}: {e}")
        import traceback
        traceback.print_exc()

