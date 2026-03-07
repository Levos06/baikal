import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

FILE_PATH = "/home3/ivkhar/Baikal/data/normed/baikal_mc2020_multi_split_0924mid_eq_norm.h5"
OUTPUT_DIR = "2026-03-07_investigation_of_2026-02-28_results/adj_matrices"

def get_temporal_edges(n, k):
    if n <= 1:
        return np.zeros((2, 0), dtype=np.int64)
    idx = np.arange(n)
    mask = (np.abs(idx[:, None] - idx) <= k) & (idx[:, None] != idx)
    return np.where(mask)

def visualize_adj():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    k_values = [2, 3, 4, 12]
    
    with h5py.File(FILE_PATH, 'r') as f:
        starts = f['train/ev_starts/data'][:6] # 5 events
        
        for i in range(5):
            start = starts[i]
            end = starts[i+1]
            num_nodes = end - start
            
            if num_nodes == 0:
                continue
                
            fig, axes = plt.subplots(1, len(k_values), figsize=(20, 5))
            fig.suptitle(f"Event {i} Adjacency Matrices (nodes={num_nodes})")
            
            for ax, k in zip(axes, k_values):
                row, col = get_temporal_edges(num_nodes, k)
                adj = np.zeros((num_nodes, num_nodes))
                adj[row, col] = 1
                
                ax.imshow(adj, cmap='binary', interpolation='nearest')
                ax.set_title(f"k={k}")
                ax.set_xlabel("Node")
                ax.set_ylabel("Node")
            
            save_path = os.path.join(OUTPUT_DIR, f"event_{i}_comparison.png")
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
            print(f"Saved comparison for event {i} to {save_path}")

if __name__ == "__main__":
    visualize_adj()
