import matplotlib.pyplot as plt
import re

def extract_data(log_file):
    epochs, p9_values = [], []
    with open(log_file, 'r') as f:
        current_epoch = None
        for line in f:
            # Match "Epoch 0123"
            epoch_match = re.search(r"Epoch (\d+)", line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
            
            # Match "V-P@R0.9 0.8576" or "Val  : Loss 0.0407 | Prec 0.8882 | Rec 0.8577 | P@R0.9 0.8236"
            p9_match = re.search(r"P@R0.9\s+([0-9.]+)", line)
            if p9_match and "Val" in line and current_epoch is not None:
                epochs.append(current_epoch)
                p9_values.append(float(p9_match.group(1)))
    return epochs, p9_values

e_mlp, p_mlp = extract_data('2026-03-25_full_mlp_training_1000ep/train_full_mlp.log')
e_lrn, p_lrn = extract_data('2026-03-25_learnable_c_water/train_learnable_c.log')

plt.figure(figsize=(12, 7))
plt.plot(e_mlp, p_mlp, label='Full MLP Training (Fixed C=0.225)', color='green', alpha=0.8)
plt.plot(e_lrn, p_lrn, label='Learnable C_Water Training', color='blue', alpha=0.8)

# Add smoothing (moving average)
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

import numpy as np
if len(p_mlp) > 10:
    plt.plot(e_mlp, smooth(p_mlp, 5), color='darkgreen', linewidth=2, label='Full MLP (Smooth)')
if len(p_lrn) > 10:
    plt.plot(e_lrn, smooth(p_lrn, 5), color='darkblue', linewidth=2, label='Learnable C (Smooth)')

plt.axhline(y=0.8726, color='red', linestyle='--', label='Previous Best (0.8726)')
plt.xlabel('Epoch')
plt.ylabel('V-P@R0.9')
plt.title('Comparison of Training Strategies: Full MLP vs. Learnable Physics')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.ylim(0.75, 0.90)
plt.tight_layout()
plt.savefig('comparison_p9_results.png')
print(f"Comparison plot saved to comparison_p9_results.png")
