import matplotlib.pyplot as plt
import re
import numpy as np
import os

def extract_p9(log_path):
    epochs, p9_vals = [], []
    if not os.path.exists(log_path): return [], []
    with open(log_path, 'r') as f:
        for line in f:
            e_match = re.search(r"Epoch (\d+)", line)
            p_match = re.search(r"Val P@R0.9:\s+([0-9.]+)", line)
            if e_match and p_match:
                epochs.append(int(e_match.group(1)))
                p9_vals.append(float(p_match.group(1)))
    return epochs, p9_vals

# 1. Load data
e21, p21 = extract_p9('2026-03-28_extended_21features/train.log')
e13, p13 = extract_p9('2026-03-30_refined_13features/train.log')

plt.figure(figsize=(14, 8))

# Helper for smoothing
def smooth(y, window=20):
    if len(y) < window: return y
    return np.convolve(y, np.ones(window)/window, mode='valid')

# Plot Raw and Smooth for 21-feat
plt.plot(e21, p21, color='blue', alpha=0.15)
s21 = smooth(p21)
plt.plot(e21[len(e21)-len(s21):], s21, color='blue', linewidth=2, label='21 Features (Previous Best)')

# Plot Raw and Smooth for Refined 13-feat
plt.plot(e13, p13, color='green', alpha=0.15)
s13 = smooth(p13)
plt.plot(e13[len(e13)-len(s13):], s13, color='green', linewidth=2, label='Refined 13 Features + Z-Upgrade')

plt.axhline(y=0.90, color='red', linestyle='--', alpha=0.5, label='Target 0.90')
plt.xlabel('Epoch')
plt.ylabel('Val P@R0.9 Score')
plt.title('Strategy Comparison: 21 Features vs. Refined 13 Features with Z-Centering')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.ylim(0.4, 0.95)
plt.xlim(0, 1500) # Compare first 1500 epochs
plt.tight_layout()

output_path = 'tools/comparison_refined_strategy.png'
plt.savefig(output_path)
print(f"Comparison plot saved to {output_path}")
