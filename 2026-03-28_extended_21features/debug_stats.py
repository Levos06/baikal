import torch
import glob
import os
from torch_geometric.loader import DataLoader
# Import existing logic to test it
from train_21feat import add_features_21

DATA_DIR = "../data_processed_50k/val"

def debug_features():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.pt")))[:1]
    data_list = torch.load(files[0], weights_only=False)
    loader = DataLoader(data_list, batch_size=100)
    
    for batch in loader:
        batch = add_features_21(batch.to(device))
        x = batch.x.cpu().numpy()
        
        print(f"{'Feature':<25} | {'Min':>10} | {'Max':>10} | {'Mean':>10}")
        print("-" * 65)
        for i in range(x.shape[1]):
            name = f"F{i}"
            if i < 5: names = ["Charge", "Time", "X", "Y", "Z"]
            else: names = [] # Simplified for debug
            
            f_name = names[i] if i < len(names) else f"Extra_{i}"
            print(f"{f_name:<25} | {x[:,i].min():10.2f} | {x[:,i].max():10.2f} | {x[:,i].mean():10.2f}")
        break

if __name__ == "__main__":
    debug_features()
