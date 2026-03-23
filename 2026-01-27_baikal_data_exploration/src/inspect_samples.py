import h5py
import numpy as np

FILE_PATH = "/home3/ivkhar/Baikal/data/normed/baikal_mc2020_multi_split_0924mid_eq_norm.h5"

def inspect():
    with h5py.File(FILE_PATH, 'r') as f:
        print(f"Dataset structure: {list(f['train'].keys())}")
        
        starts = f['train/ev_starts/data'][:5]
        data = f['train/data/data']
        labels = f['train/labels/data']
        
        for i in range(len(starts)-1):
            s, e = starts[i], starts[i+1]
            ev_data = data[s:e]
            ev_labels = labels[s:e]
            
            print(f"\n--- Event {i} (Hits: {e-s}) ---")
            print(f"Labels summary: Signal={np.sum(ev_labels!=0)}, Background={np.sum(ev_labels==0)}")
            
            # Названия признаков (известные нам): Charge, Time, X, Y, Z
            print("First 5 hits (features):")
            print("  Charge |  Time  |   X    |   Y    |   Z")
            for hit_idx in range(min(5, len(ev_data))):
                print("  " + " | ".join([f"{v:7.2f}" for v in ev_data[hit_idx]]))
            
            # Проверка на аномалии
            if np.any(np.isnan(ev_data)) or np.any(np.isinf(ev_data)):
                print("!!! WARNING: NaN or Inf found in data!")
            
            t_vals = ev_data[:, 1]
            if np.all(np.diff(t_vals) >= 0):
                print("Data is sorted by time.")
            else:
                print("Data is NOT sorted by time.")

if __name__ == "__main__":
    inspect()
