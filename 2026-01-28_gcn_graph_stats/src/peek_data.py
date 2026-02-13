import h5py
import numpy as np

FILE_PATH = "/home3/ivkhar/Baikal/data/normed/baikal_mc2020_multi_split_0924mid_eq_norm.h5"

with h5py.File(FILE_PATH, 'r') as f:
    print("=== norm_param ===")
    for k in f['norm_param']:
        print(f"{k}: {f['norm_param'][k][()]}")
    
    print("\n=== train/data/data (first 5 hits) ===")
    print(f['train/data/data'][:5])
    
    print("\n=== train/t_res/data (first 5 hits) ===")
    print(f['train/t_res/data'][:5])

    print("\n=== train/ev_starts/data (first 5 starts) ===")
    starts = f['train/ev_starts/data'][:5]
    print(starts)
    
    # Get first event hits
    ev0_hits = f['train/data/data'][starts[0]:starts[1]]
    print(f"\nFirst event hits shape: {ev0_hits.shape}")
    print(ev0_hits[:5])
