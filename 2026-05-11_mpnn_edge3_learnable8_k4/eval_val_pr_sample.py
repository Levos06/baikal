#!/usr/bin/env python3
"""
Lightweight eval: fixed number of val batches, no full-shard preload (avoids RAM blowup).
Optional: run `watch -n1 nvidia-smi` in another terminal during eval.
"""
import argparse
import glob
import os
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_curve, precision_score, recall_score
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader

EXP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXP)
import train_mpnn_edge3_learn8 as T

DATA_DIR = T.DATA_DIR
K = 4


def _add_temporal_edges(data: Data, k: int) -> Data:
    n = data.x.size(0)
    if n <= 1:
        data.edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        import numpy as np

        indices = np.arange(n)
        mask = (np.abs(indices[:, None] - indices) <= k) & (indices[:, None] != indices)
        data.edge_index = torch.from_numpy(np.array(np.where(mask))).long()
    return data


def collect_limited_graphs(max_graphs: int):
    out = []
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "val", "*.pt"))):
        lst = torch.load(path, weights_only=False)
        for data in lst:
            out.append(_add_temporal_edges(data.clone(), K))
            if len(out) >= max_graphs:
                return out
    return out


def p_at_r(labels, probs, r_target):
    p, r, _ = precision_recall_curve(labels, probs)
    if len(r) == 0 or np.max(r) < r_target:
        return float("nan")
    return float(np.interp(r_target, r[::-1], p[::-1]))


def nvidia_smi_snapshot():
    try:
        r = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return (r.stdout or "").strip()
    except Exception as e:
        return f"(nvidia-smi failed: {e})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(EXP, "checkpoints/model_mpnn_edge3_learn8_k4_900.pt"))
    ap.add_argument("--max-graphs", type=int, default=2000, help="Cap graphs loaded into RAM")
    ap.add_argument("--batch-size", type=int, default=T.BATCH_SIZE)
    ap.add_argument("--num-workers", type=int, default=0, help="0 = less RAM spikes from workers")
    ap.add_argument("--device", default=None, help="cuda:0 or cpu; default from train script")
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else T.DEVICE
    print("GPU snapshot (before load):", nvidia_smi_snapshot(), flush=True)

    t0 = time.perf_counter()
    graphs = collect_limited_graphs(args.max_graphs)
    t_load = time.perf_counter() - t0
    print(f"Loaded {len(graphs)} graphs in {t_load:.2f}s", flush=True)
    print("GPU snapshot (after CPU load):", nvidia_smi_snapshot(), flush=True)

    loader = DataLoader(
        graphs,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    m = T.MPNN_FCLastEdgeProductEmb(21, 2).to(device)
    m.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=True))
    m.eval()
    print("GPU snapshot (after model to GPU):", nvidia_smi_snapshot(), flush=True)

    all_labels, all_probs, losses = [], [], []
    t_inf0 = time.perf_counter()
    n_batches = 0
    with torch.no_grad():
        for batch in loader:
            batch = T.add_features_21_norm(batch.to(device))
            out = m(batch.x, batch.edge_index, batch.batch)
            losses.append(F.cross_entropy(out, batch.y).item())
            all_probs.extend(F.softmax(out, dim=1)[:, 1].cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
            n_batches += 1
    t_inf = time.perf_counter() - t_inf0

    y = np.asarray(all_labels)
    pr = np.asarray(all_probs)
    pred = (pr > 0.5).astype(int)

    print("GPU snapshot (after inference):", nvidia_smi_snapshot(), flush=True)
    print(f"Inference: {n_batches} batches, wall {t_inf:.2f}s", flush=True)
    print(f"mean_loss: {np.mean(losses):.4f}", flush=True)
    print(f"prec@0.5: {precision_score(y, pred, zero_division=0):.4f}", flush=True)
    print(f"rec@0.5: {recall_score(y, pred, zero_division=0):.4f}", flush=True)
    print(f"P@R0.9 (sample): {p_at_r(y, pr, 0.9):.4f}", flush=True)
    print(f"P@R0.95 (sample): {p_at_r(y, pr, 0.95):.4f}", flush=True)


if __name__ == "__main__":
    main()
