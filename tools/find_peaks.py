import re

def find_peaks(log_path, name):
    best_p9 = 0.0
    best_p9_epoch = 0
    min_loss = 100.0
    min_loss_epoch = 0
    
    current_epoch = 0
    with open(log_path, 'r') as f:
        for line in f:
            e_match = re.search(r"Epoch (\d+)", line)
            if e_match:
                current_epoch = int(e_match.group(1))
            
            if "Val  :" in line:
                loss_m = re.search(r"Loss\s+([0-9.]+)", line)
                p9_m = re.search(r"P@R0.9\s+([0-9.]+)", line)
                if loss_m and p9_m:
                    loss = float(loss_m.group(1))
                    p9 = float(p9_m.group(1))
                    
                    if p9 > best_p9:
                        best_p9 = p9
                        best_p9_epoch = current_epoch
                    if loss < min_loss:
                        min_loss = loss
                        min_loss_epoch = current_epoch
                        
    print(f"--- {name} ---")
    print(f"Peak V-P@R0.9: {best_p9:.4f} (Epoch {best_p9_epoch})")
    print(f"Min V-Loss:    {min_loss:.4f} (Epoch {min_loss_epoch})")

find_peaks('2026-03-25_full_mlp_training_1000ep/train_full_mlp.log', 'Full MLP')
find_peaks('2026-03-25_learnable_c_water/train_learnable_c.log', 'Learnable C_Water')
