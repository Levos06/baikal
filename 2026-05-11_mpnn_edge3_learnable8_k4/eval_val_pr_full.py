#!/usr/bin/env python3
"""Full val pass: P@R0.9 and P@R0.95 for MPNN edge3 learnable 8."""
import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_curve, precision_score, recall_score
from torch.utils.data import Dataset
from torch_geometric.data import Data
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


class ValShardDataset(Dataset):
    def __init__(self, k: int = K):
        super().__init__()
        self.k = k
        self._graphs = []
        paths = sorted(glob.glob(os.path.join(DATA_DIR, "val", "*.pt")))
        if not paths:
            raise FileNotFoundError(f"No val shards under {DATA_DIR}/val")
        for path in paths:
            for data in torch.load(path, weights_only=False):
                self._graphs.append(_add_temporal_edges(data.clone(), k))

    def __len__(self):
        return len(self._graphs)

    def __getitem__(self, idx):
        return self._graphs[idx]


def p_at_r(labels: np.ndarray, probs: np.ndarray, r_target: float) -> float:
    p, r, _ = precision_recall_curve(labels, probs)
    if len(r) == 0 or np.max(r) < r_target:
        return float("nan")
    return float(np.interp(r_target, r[::-1], p[::-1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpt",
        type=str,
        default=os.path.join(EXP, "checkpoints/model_mpnn_edge3_learn8_k4_900.pt"),
    )
    ap.add_argument("--batch-size", type=int, default=T.BATCH_SIZE)
    args = ap.parse_args()

    device = T.DEVICE
    ds = ValShardDataset(K)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=T.NUM_WORKERS,
        pin_memory=True,
    )

    m = T.MPNN_FCLastEdgeProductEmb(21, 2).to(device)
    m.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=True))
    m.eval()

    all_labels, all_probs, losses = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = T.add_features_21_norm(batch.to(device))
            out = m(batch.x, batch.edge_index, batch.batch)
            losses.append(F.cross_entropy(out, batch.y).item())
            all_probs.extend(F.softmax(out, dim=1)[:, 1].cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())

    y = np.asarray(all_labels)
    pr = np.asarray(all_probs)
    pred = (pr > 0.5).astype(int)

    print(f"ckpt: {args.ckpt}")
    print(f"split: val (full), n_samples={len(y)}")
    print(f"mean_loss: {np.mean(losses):.4f}")
    print(f"prec@0.5: {precision_score(y, pred, zero_division=0):.4f}")
    print(f"rec@0.5: {recall_score(y, pred, zero_division=0):.4f}")
    print(f"P@R0.9: {p_at_r(y, pr, 0.9):.4f}")
    print(f"P@R0.95: {p_at_r(y, pr, 0.95):.4f}")


if __name__ == "__main__":
    main()
