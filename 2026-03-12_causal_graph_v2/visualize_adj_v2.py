import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

FILE_PATH = "/home3/ivkhar/Baikal/data/normed/baikal_mc2020_multi_split_0924mid_eq_norm.h5"
OUTPUT_DIR = "2026-03-12_causal_graph_v2/adj_matrices"
V_WATER = 0.225
T_CUT = 3.0
D_MAX = 100.0

def get_causal_edges(pos, t):
    n = pos.shape[0]
    if n <= 1: return np.zeros((2, 0))
    dist = np.sqrt(np.sum((pos[:, None, :] - pos[None, :, :])**2, axis=2))
    dt = np.abs(t[:, None] - t[None, :])
    mask = (np.abs(dt - dist/V_WATER) < T_CUT) & (dist < D_MAX) & (dist > 0)
    return np.where(mask)

def visualize():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with h5py.File(FILE_PATH, 'r') as f:
        starts = f['train/ev_starts/data'][:6]
        for i in range(5):
            start, end = starts[i], starts[i+1]
            data = f['train/data/data'][start:end]
            pos, t = data[:, 2:5], data[:, 1]
            n = end - start
            
            rows, cols = get_causal_edges(pos, t)
            density = len(rows) / (n * (n-1)) if n > 1 else 0
            
            plt.figure(figsize=(6, 6))
            adj = np.zeros((n, n))
            adj[rows, cols] = 1
            plt.imshow(adj, cmap='binary')
            plt.title(f"Causal v2 (T_CUT=3.0) | Density: {density:.4f}")
            plt.savefig(os.path.join(OUTPUT_DIR, f"event_{i}_adj_v2.png"))
            plt.close()
            print(f"Event {i} Density: {density:.4f}")

if __name__ == "__main__":
    visualize()
