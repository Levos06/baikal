#!/usr/bin/env python3
"""
Single-event adjacency heatmaps: base k=4 vs base ∪ cos(h)>beta after layer-1 hidden state.
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import colors as mcolors
from matplotlib.gridspec import GridSpec
from torch_geometric.data import Batch, Data

EXP_DIR = "/home/levos/experiments/2026-05-11_general_mpnn_cos_sim_beta05_k4"
sys.path.insert(0, EXP_DIR)
from train_mpnn_cos_sim_beta05 import (  # noqa: E402
    COS_SIM_BETA,
    MPNN_CosSim_FCLast,
    add_features_21_norm,
    get_similarity_edges,
)

DATA_VAL = "/home/levos/experiments/data_processed_50k/val/medium_001.pt"
CKPT = os.path.join(EXP_DIR, "checkpoints/model_mpnn_cossim_beta05_k4_1000.pt")
OUT_PATH = os.path.join(EXP_DIR, "figures/adjacency_heatmaps_val_events.png")
K = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def temporal_edges(num_nodes: int, k: int) -> torch.Tensor:
    if num_nodes <= 1:
        return torch.zeros((2, 0), dtype=torch.long)
    indices = np.arange(num_nodes)
    mask = (np.abs(indices[:, None] - indices) <= k) & (indices[:, None] != indices)
    return torch.from_numpy(np.array(np.where(mask))).long()


def to_symmetric_adj(n: int, edge_index: torch.Tensor) -> np.ndarray:
    adj = np.zeros((n, n), dtype=np.float32)
    if edge_index.numel() == 0:
        return adj
    r, c = edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy()
    adj[r, c] = 1.0
    adj[c, r] = 1.0
    return adj


def pick_one_event(data_list):
    """Smallest graph in shard (val 50k shards have n >= 40 here; minimizes matrix size)."""
    return min(data_list, key=lambda d: d.x.size(0))


def plot_panel(ax, mat, title, n, cmap):
    im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=1, interpolation="nearest", aspect="equal")
    ax.set_title(title, fontsize=12, pad=10, fontweight="600")
    step = 5 if n > 30 else (2 if n > 16 else 1)
    ticks = np.arange(0, n, step)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlabel("node j", fontsize=10, labelpad=6)
    ax.set_ylabel("node i", fontsize=10, labelpad=6)
    ax.tick_params(axis="both", labelsize=9, length=0)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.9)
    ax.tick_params(which="minor", bottom=False, left=False)
    return im


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#fafcff",
            "font.size": 10,
            "axes.titlecolor": "#1e293b",
            "axes.labelcolor": "#475569",
            "xtick.color": "#64748b",
            "ytick.color": "#64748b",
        }
    )

    data_list = torch.load(DATA_VAL, weights_only=False)
    data = pick_one_event(data_list)
    if data is None:
        raise SystemExit("No suitable graph in validation shard.")

    model = MPNN_CosSim_FCLast(21, 2, cos_threshold=COS_SIM_BETA).to(DEVICE)
    state = torch.load(CKPT, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    n = data.x.size(0)
    ei = temporal_edges(n, K)
    g = Data(x=data.x.clone(), edge_index=ei, y=data.y.clone())
    batch = Batch.from_data_list([g.to(DEVICE)])
    batch = add_features_21_norm(batch)
    base_adj = to_symmetric_adj(n, batch.edge_index)

    with torch.no_grad():
        h = batch.x
        h = F.gelu(model.convs[0](h, batch.edge_index) + model.projs[0](h))
        sim_e = get_similarity_edges(h, batch.batch, COS_SIM_BETA)
        combined = torch.cat([batch.edge_index, sim_e], dim=1)
        combined = torch.unique(combined, dim=1)
    comb_adj = to_symmetric_adj(n, combined)

    n_base_dir = batch.edge_index.size(1)
    n_extra_dir = max(0, combined.size(1) - batch.edge_index.size(1))

    cmap = mcolors.LinearSegmentedColormap.from_list("adj", ["#e8eef5", "#0c2d4d"], N=256)

    fig = plt.figure(figsize=(10.5, 4.8), dpi=120)
    gs = GridSpec(1, 3, width_ratios=[1, 1, 0.06], wspace=0.28)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[0, 2])

    plot_panel(
        ax0,
        base_adj,
        f"Temporal base  |i−j| ≤ {K}\nn = {n}   directed edges = {n_base_dir}",
        n,
        cmap,
    )
    im = plot_panel(
        ax1,
        comb_adj,
        f"Base ∪ cos-sim on h (layer 1)\nβ = {COS_SIM_BETA}   + directed edges = {n_extra_dir}",
        n,
        cmap,
    )
    cb = fig.colorbar(im, cax=cax, ticks=[0, 0.5, 1])
    cb.ax.set_yticklabels(["no edge", "", "edge"])
    cb.ax.tick_params(labelsize=9)

    fig.suptitle(
        "Symmetric adjacency (val shard, checkpoint epoch 1000)",
        fontsize=13,
        fontweight="600",
        y=1.02,
    )
    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Wrote", OUT_PATH, f"(n={n})")


if __name__ == "__main__":
    main()
