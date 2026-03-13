import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

FILE_PATH = "/home3/ivkhar/Baikal/data/normed/baikal_mc2020_multi_split_0924mid_eq_norm.h5"
OUTPUT_DIR = "2026-03-12_causal_graph/adj_matrices"
V_WATER = 0.225
T_CUT = 50.0
D_MAX = 100.0

def get_temporal_edges(n, k=2):
    if n <= 1: return np.zeros((2, 0))
    idx = np.arange(n)
    mask = (np.abs(idx[:, None] - idx) <= k) & (idx[:, None] != idx)
    return np.where(mask)

def get_causal_edges(pos, t):
    n = pos.shape[0]
    if n <= 1: return np.zeros((2, 0))
    dist = np.sqrt(np.sum((pos[:, None, :] - pos[None, :, :])**2, axis=2))
    dt = np.abs(t[:, None] - t[None, :])
    mask = (np.abs(dt - dist/V_WATER) < T_CUT) & (dist < D_MAX) & (dist > 0)
    return np.where(mask)

def visualize_comparison():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with h5py.File(FILE_PATH, 'r') as f:
        starts = f['train/ev_starts/data'][:11]
        for i in range(10):
            start, end = starts[i], starts[i+1]
            data = f['train/data/data'][start:end]
            pos, t = data[:, 2:5], data[:, 1]
            n = end - start
            
            # Старый граф k=2
            rows_t, cols_t = get_temporal_edges(n, 2)
            # Новый причинный граф
            rows_c, cols_c = get_causal_edges(pos, t)
            
            density_t = len(rows_t) / (n * (n-1)) if n > 1 else 0
            density_c = len(rows_c) / (n * (n-1)) if n > 1 else 0
            
            print(f"Event {i} | Nodes: {n} | Density k=2: {density_t:.4f} | Density Causal: {density_c:.4f}")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            adj_t = np.zeros((n, n))
            adj_t[rows_t, cols_t] = 1
            ax1.imshow(adj_t, cmap='binary')
            ax1.set_title(f"Temporal k=2 (Density: {density_t:.3f})")
            
            adj_c = np.zeros((n, n))
            adj_c[rows_c, cols_c] = 1
            ax2.imshow(adj_c, cmap='binary')
            ax2.set_title(f"Causal (Density: {density_c:.3f})")
            
            plt.savefig(os.path.join(OUTPUT_DIR, f"event_{i}_comparison.png"))
            plt.close()

if __name__ == "__main__":
    visualize_comparison()
