import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

FILE_PATH = "/home3/ivkhar/Baikal/data/normed/baikal_mc2020_multi_split_0924mid_eq_norm.h5"
OUTPUT_DIR = "2026-03-13_causal_window_study/plots/adj_matrices"
V_WATER = 0.225
T_CUT_HALF = 0.5

def get_symmetric_edges(pos, t, t_cut):
    n = pos.shape[0]
    if n <= 1: return np.zeros((2, 0))
    dist = np.sqrt(np.sum((pos[:, None, :] - pos[None, :, :])**2, axis=2))
    dt = np.abs(t[:, None] - t[None, :])
    mask = (np.abs(dt - dist/V_WATER) < t_cut) & (dist < 100.0) & (dist > 0)
    return np.where(mask)

def get_asymmetric_edges(pos, t, t_cut_half):
    n = pos.shape[0]
    if n <= 1: return np.zeros((2, 0))
    dist = np.sqrt(np.sum((pos[:, None, :] - pos[None, :, :])**2, axis=2))
    # dt[i, j] = t[j] - t[i]
    dt = t[None, :] - t[:, None]
    # Условие: 0 < dt - dist/v < 2*t_cut_half
    mask = (dt - dist/V_WATER > 0) & (dt - dist/V_WATER < 2*t_cut_half) & (dist < 100.0)
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
            
            rows_s, cols_s = get_symmetric_edges(pos, t, T_CUT_HALF)
            rows_a, cols_a = get_asymmetric_edges(pos, t, T_CUT_HALF)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            adj_s = np.zeros((n, n))
            adj_s[rows_s, cols_s] = 1
            ax1.imshow(adj_s, cmap='binary')
            ax1.set_title(f"Symmetric (|diff| < 0.5ns)")
            
            adj_a = np.zeros((n, n))
            adj_a[rows_a, cols_a] = 1
            ax2.imshow(adj_a, cmap='binary')
            ax2.set_title(f"Asymmetric (0 < diff < 1.0ns)")
            
            plt.suptitle(f"Event {i} Topology Comparison")
            plt.savefig(os.path.join(OUTPUT_DIR, f"event_{i}_topology.png"))
            plt.close()
            print(f"Event {i} processed.")

if __name__ == "__main__":
    visualize()
