import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

FILE_PATH = "/home3/ivkhar/Baikal/data/normed/baikal_mc2020_multi_split_0924mid_eq_norm.h5"
PLOT_DIR = "../plots"

def print_structure(name, obj):
    print(name, obj)

def explore():
    if not os.path.exists(PLOT_DIR):
        os.makedirs(PLOT_DIR)

    with h5py.File(FILE_PATH, 'r') as f:
        print("=== File Structure ===")
        f.visititems(print_structure)
        print("======================\n")

        # Specifically analyze train/data/data
        target_dataset = "train/data/data"
        
        if target_dataset in f:
            print(f"Analyzing dataset: {target_dataset}")
            ds = f[target_dataset]
            print(f"Shape: {ds.shape}")
            print(f"Type: {ds.dtype}")

            # Sample first 1000 items
            sample_size = min(1000, ds.shape[0])
            data_sample = ds[:sample_size]

            # data_sample shape is (1000, 5)
            # Calculate stats for each feature
            print(f"\nStats for first {sample_size} elements (per feature):")
            feature_names = ["Feature 0", "Feature 1", "Feature 2", "Feature 3", "Feature 4"]
            
            for i in range(data_sample.shape[1]):
                feat_data = data_sample[:, i]
                print(f"\n{feature_names[i]}:")
                print(f"  Min: {np.min(feat_data)}")
                print(f"  Max: {np.max(feat_data)}")
                print(f"  Mean: {np.mean(feat_data)}")
                print(f"  Std: {np.std(feat_data)}")

                # Plot histogram for this feature
                plt.figure(figsize=(10, 6))
                plt.hist(feat_data, bins=50, alpha=0.7, color='green', edgecolor='black')
                plt.title(f"Distribution of {target_dataset} - {feature_names[i]}")
                plt.xlabel("Value")
                plt.ylabel("Frequency")
                plt.grid(True, alpha=0.3)
                
                plot_path = os.path.join(PLOT_DIR, f"train_data_feat_{i}_hist.png")
                plt.savefig(plot_path)
                print(f"  Saved plot to {plot_path}")
                plt.close()
        else:
            print(f"Dataset {target_dataset} not found.")

if __name__ == "__main__":
    explore()