import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

FILE_PATH = "/home3/ivkhar/Baikal/data/normed/baikal_mc2020_multi_split_0924mid_eq_norm.h5"
OUTPUT_DIR = "2026-03-13_causal_window_study/plots/adj_matrices_01"
V_WATER = 0.225
T_CUT_HALF = 0.1

def get_asymmetric_edges(pos, t, t_cut_half):
    n = pos.shape[0]
    if n <= 1: return np.zeros((2, 0))
    dist = np.sqrt(np.sum((pos[:, None, :] - pos[None, :, :])**2, axis=2))
    dt = t[None, :] - t[:, None]
    # Условие: 0 < dt - dist/v < 2*t_cut_half
    mask = (dt - dist/V_WATER > 0) & (dt - dist/V_WATER < 2*t_cut_half) & (dist < 100.0)
    return np.where(mask)

def visualize_and_calculate():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    densities = []
    with h5py.File(FILE_PATH, 'r') as f:
        starts = f['train/ev_starts/data'][:11]
        for i in range(10):
            start, end = starts[i], starts[i+1]
            data = f['train/data/data'][start:end]
            pos, t = data[:, 2:5], data[:, 1]
            n = end - start
            
            rows, cols = get_asymmetric_edges(pos, t, T_CUT_HALF)
            density = len(rows) / (n * (n-1)) if n > 1 else 0
            densities.append(density)
            
            plt.figure(figsize=(6, 6))
            adj = np.zeros((n, n))
            adj[rows, cols] = 1
            plt.imshow(adj, cmap='binary')
            plt.title(f"Asymmetric 0.1ns | Density: {density:.6f}")
            plt.savefig(os.path.join(OUTPUT_DIR, f"event_{i}_adj_01ns.png"))
            plt.close()
            print(f"Event {i} | Nodes: {n} | Edges: {len(rows)} | Density: {density:.6f}")
            
    print(f"\nAverage Density for 0.1ns window: {np.mean(densities):.6f}")

if __name__ == "__main__":
    visualize_and_calculate()
