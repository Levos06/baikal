import h5py
import time

FILE_PATH = "/home3/ivkhar/Baikal/data/normed/baikal_mc2020_multi_split_0924mid_eq_norm.h5"

try:
    print(f"Opening {FILE_PATH}...")
    start = time.time()
    with h5py.File(FILE_PATH, 'r') as f:
        print(f"Success! Keys: {list(f.keys())}")
        print(f"Time taken: {time.time() - start:.2f}s")
        print(f"Train starts: {f['train/ev_starts/data'][:5]}")
except Exception as e:
    print(f"Error: {e}")
