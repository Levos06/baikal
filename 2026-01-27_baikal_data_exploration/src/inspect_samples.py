import h5py
import numpy as np
from collections import Counter

path = "/home3/ivkhar/Baikal/data/normed/baikal_mc2020_multi_split_0924mid_eq_norm.h5"

with h5py.File(path, "r") as f:
    print("Scanning ALL event IDs for prefixes...")
    ev_ids_ds = f["train/ev_ids/data"]
    total = len(ev_ids_ds)
    chunk_size = 1000000
    prefixes = Counter()
    
    for i in range(0, total, chunk_size):
        data = ev_ids_ds[i : i + chunk_size]
        for x in data:
            s = x.decode('utf-8') if isinstance(x, bytes) else str(x)
            prefixes[s[:5]] += 1
            
    print(f"\nUnique prefixes found in ALL {total} events:")
    for p, count in prefixes.most_common():
        print(f"'{p}': {count}")
