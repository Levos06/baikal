import torch
import os
import glob
from torch_geometric.loader import DataLoader
from tqdm import tqdm

DATA_DIR = "data_processed_50k/train"

def calculate_z_scale():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.pt")))[:100]
    all_centered_z = []
    
    print("Calculating global Z-scale (event-centered)...")
    for f in tqdm(files):
        data_list = torch.load(f, weights_only=False)
        loader = DataLoader(data_list, batch_size=512)
        for batch in loader:
            z = batch.x[:, 4] # Current normalized Z
            ptr = batch.ptr
            
            # Event-based centering
            for i in range(len(ptr)-1):
                z_event = z[ptr[i]:ptr[i+1]]
                z_centered = z_event - torch.mean(z_event)
                all_centered_z.append(z_centered)
                
    all_z = torch.cat(all_centered_z)
    global_std = torch.std(all_z).item()
    
    print(f"\n--- Z-UPGRADE RESULTS ---")
    print(f"Global STD of event-centered Z: {global_std:.6f}")
    
    # Save the scale factor
    os.makedirs('tools', exist_ok=True)
    with open('tools/z_scale_factor.txt', 'w') as f:
        f.write(str(global_std))
    
    return global_std

if __name__ == "__main__":
    calculate_z_scale()
