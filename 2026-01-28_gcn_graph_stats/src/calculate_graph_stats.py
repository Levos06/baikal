import h5py
import numpy as np
import os
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import connected_components

FILE_PATH = "/home3/ivkhar/Baikal/data/normed/baikal_mc2020_multi_split_0924mid_eq_norm.h5"
C_LIGHT = 0.299792458  # m/ns
NUM_EVENTS = 1000

def calculate_stats():
    with h5py.File(FILE_PATH, 'r') as f:
        # Load normalization parameters
        means = f['norm_param/mean'][()]
        stds = f['norm_param/std'][()]
        
        # Feature mapping (assumed based on exploration)
        # 1: time, 2: x, 3: y, 4: z
        T_IDX, X_IDX, Y_IDX, Z_IDX = 1, 2, 3, 4
        
        starts = f['train/ev_starts/data'][:NUM_EVENTS + 1]
        data = f['train/data/data'][starts[0]:starts[NUM_EVENTS]]
        labels = f['train/labels/data'][starts[0]:starts[NUM_EVENTS]]
        
        all_stats = []
        
        for i in range(NUM_EVENTS):
            ev_start = starts[i] - starts[0]
            ev_end = starts[i+1] - starts[0]
            
            ev_data = data[ev_start:ev_end]
            ev_labels = labels[ev_start:ev_end]
            
            num_nodes = ev_data.shape[0]
            if num_nodes <= 1:
                all_stats.append({
                    'num_nodes': num_nodes,
                    'num_edges': 0,
                    'num_isolated': num_nodes,
                    'num_components': num_nodes,
                    'is_connected': True if num_nodes == 1 else False
                })
                continue
            
            # Denormalize
            t = ev_data[:, T_IDX] * stds[T_IDX] + means[T_IDX]
            x = ev_data[:, X_IDX] * stds[X_IDX] + means[X_IDX]
            y = ev_data[:, Y_IDX] * stds[Y_IDX] + means[Y_IDX]
            z = ev_data[:, Z_IDX] * stds[Z_IDX] + means[Z_IDX]
            
            coords = np.stack([x, y, z], axis=1)
            
            # Pairwise distances and time differences
            dist_matrix = squareform(pdist(coords))
            time_matrix = np.abs(t[:, np.newaxis] - t[np.newaxis, :])
            
            # Edge condition: d < c * dt
            # Using i != j to avoid self-loops
            adj = (dist_matrix < C_LIGHT * time_matrix)
            
            if i == 0:
                print(f"\nSample event 0 (num_nodes={num_nodes}):")
                print(f"  Dist matrix sample (5x5):\n{dist_matrix[:5, :5]}")
                print(f"  c * Time matrix sample (5x5):\n{(C_LIGHT * time_matrix)[:5, :5]}")
                print(f"  Edges in 5x5: {np.sum(adj[:5, :5])}")

            np.fill_diagonal(adj, 0)
            
            num_edges = np.sum(adj) // 2
            degrees = np.sum(adj, axis=1)
            num_isolated = np.sum(degrees == 0)
            
            n_components, labels_comp = connected_components(adj, directed=False)
            
            all_stats.append({
                'num_nodes': num_nodes,
                'num_edges': num_edges,
                'num_isolated': num_isolated,
                'num_components': n_components,
                'is_connected': n_components == 1
            })
            
        return all_stats

def main():
    print(f"Calculating stats for {NUM_EVENTS} events...")
    stats = calculate_stats()
    
    nodes = [s['num_nodes'] for s in stats]
    edges = [s['num_edges'] for s in stats]
    isolated = [s['num_isolated'] for s in stats]
    components = [s['num_components'] for s in stats]
    connected = [s['is_connected'] for s in stats]
    
    print("\n=== Statistics ===")
    print(f"Avg Nodes: {np.mean(nodes):.2f} (min: {np.min(nodes)}, max: {np.max(nodes)})")
    print(f"Avg Edges: {np.mean(edges):.2f} (min: {np.min(edges)}, max: {np.max(edges)})")
    print(f"Avg Isolated: {np.mean(isolated):.2f} ({np.mean(isolated)/np.mean(nodes)*100:.1f}%)")
    print(f"Avg Components: {np.mean(components):.2f}")
    print(f"Events fully connected: {np.sum(connected)} / {NUM_EVENTS}")
    
    # Check edge density
    densities = [s['num_edges'] / (s['num_nodes']*(s['num_nodes']-1)/2) if s['num_nodes'] > 1 else 0 for s in stats]
    print(f"Avg Edge Density: {np.mean(densities):.4f}")

if __name__ == "__main__":
    main()
