"""
Build a 2x3 collage of binary adjacency matrices for layer-2 graphs:
base temporal k=4 edges + top-8 cosine-similarity edges on hidden states after layer 1.
"""
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import random

from train_mpnn_cos_sim_top8 import (
    MPNN_CosSimTopK_FCLast,
    add_features_21_norm,
    get_topk_similarity_edges,
    DEVICE,
    DATA_DIR,
    PROJECT_DIR,
)


def temporal_k4_edge_index(num_nodes: int, device) -> torch.Tensor:
    if num_nodes <= 1:
        return torch.zeros((2, 0), dtype=torch.long, device=device)
    indices = np.arange(num_nodes)
    mask = (np.abs(indices[:, None] - indices) <= 4) & (indices[:, None] != indices)
    return torch.from_numpy(np.array(np.where(mask))).long().to(device)


def edge_index_to_symmetric_adj(edge_index: torch.Tensor, n: int) -> np.ndarray:
    """Undirected adjacency from possibly directed edges."""
    adj = np.zeros((n, n), dtype=np.float32)
    row, col = edge_index.cpu().numpy()
    adj[row, col] = 1.0
    adj[col, row] = 1.0
    np.fill_diagonal(adj, 0.0)
    return adj


def layer2_edges(h: torch.Tensor, base_ei: torch.Tensor, batch_vec: torch.Tensor) -> torch.Tensor:
    sim_ei = get_topk_similarity_edges(h, batch_vec, k=8)
    curr = torch.cat([base_ei, sim_ei], dim=1)
    return torch.unique(curr, dim=1)


def main():
    random.seed(42)
    torch.manual_seed(42)

    ckpt_dir = os.path.join(PROJECT_DIR, "checkpoints")
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "model_mpnn_cossim_top8_k4_*.pt")))
    if not ckpts:
        raise FileNotFoundError("No checkpoints found in " + ckpt_dir)
    ckpt_path = ckpts[-1]
    print("Using checkpoint:", ckpt_path)

    model = MPNN_CosSimTopK_FCLast(21, 2).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
    model.eval()

    val_files = sorted(glob.glob(os.path.join(DATA_DIR, "val", "*.pt")))
    random.shuffle(val_files)

    events = []
    for f in val_files:
        if len(events) >= 6:
            break
        try:
            data_list = torch.load(f, weights_only=False)
        except Exception:
            continue
        random.shuffle(data_list)
        for data in data_list:
            n = data.x.size(0)
            if n >= 15 and n <= 80:  # readable matrix size
                events.append(data)
                break
        if len(events) >= 6:
            break

    if len(events) < 6:
        # Relax constraints
        for f in val_files:
            if len(events) >= 6:
                break
            try:
                data_list = torch.load(f, weights_only=False)
            except Exception:
                continue
            for data in data_list:
                n = data.x.size(0)
                if n >= 8:
                    events.append(data)
                    if len(events) >= 6:
                        break

    assert len(events) == 6, f"Need 6 events, got {len(events)}"

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    with torch.no_grad():
        for ax, data in zip(axes, events):
            n = data.x.size(0)
            x = data.x.to(DEVICE)
            y = data.y.to(DEVICE)
            edge_index = temporal_k4_edge_index(n, DEVICE)

            batch_vec = torch.zeros(n, dtype=torch.long, device=DEVICE)
            # Fake batch for add_features_21_norm (expects Batch-like object)
            class _B:
                pass

            b = _B()
            b.x, b.ptr = x, torch.tensor([0, n], device=DEVICE, dtype=torch.long)
            b.batch = batch_vec
            b.edge_index = edge_index
            b = add_features_21_norm(b)

            h = F.gelu(model.convs[0](b.x, b.edge_index) + model.projs[0](b.x))
            ei2 = layer2_edges(h, b.edge_index, b.batch)
            adj = edge_index_to_symmetric_adj(ei2, n)

            im = ax.imshow(adj, cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
            ax.set_title(f"n={n}, |E|={ei2.size(1)}", fontsize=9)
            ax.set_xlabel("j")
            ax.set_ylabel("i")
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(
        "Layer-2 undirected adjacency: temporal k=4 + top-8 cos-sim (after conv1)\n"
        f"checkpoint: {os.path.basename(ckpt_path)}",
        fontsize=11,
    )
    fig.tight_layout()
    out_path = os.path.join(PROJECT_DIR, "adjacency_collage_layer2.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
