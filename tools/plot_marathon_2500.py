import matplotlib.pyplot as plt
import re
import numpy as np
import os

def plot_final_marathon():
    log_path = '2026-03-28_extended_21features/train.log'
    epochs = []
    p9_vals = []
    
    if not os.path.exists(log_path):
        print("Log not found")
        return

    with open(log_path, 'r') as f:
        for line in f:
            e_match = re.search(r"Epoch (\d+)", line)
            p_match = re.search(r"Val P@R0.9:\s+([0-9.]+)", line)
            if e_match and p_match:
                epochs.append(int(e_match.group(1)))
                p9_vals.append(float(p_match.group(1)))
    
    plt.figure(figsize=(14, 8))
    plt.plot(epochs, p9_vals, color='cyan', alpha=0.3, label='Val P@R0.9 (Raw)')
    
    # Smooth trend
    window = 50
    if len(p9_vals) > window:
        smooth = np.convolve(p9_vals, np.ones(window)/window, mode='valid')
        plt.plot(epochs[window-1:], smooth, color='blue', linewidth=2, label=f'Trend (MA {window})')

    plt.axhline(y=0.90, color='red', linestyle='--', alpha=0.5, label='Target 0.90')
    plt.axvline(x=1500, color='orange', linestyle=':', label='Fine-tuning Start (LR=2e-5)')
    
    max_val = max(p9_vals)
    plt.axhline(y=max_val, color='green', linestyle=':', label=f'Record: {max_val:.4f}')
    
    plt.xlabel('Epoch')
    plt.ylabel('P@R0.9 Score')
    plt.title('2500-Epoch Marathon Results: 21 Features Model')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.ylim(0.4, 0.95)
    plt.tight_layout()
    
    plt.savefig('2026-03-28_extended_21features/plots/marathon_2500_final.png')
    print("Final plot saved.")

if __name__ == "__main__":
    plot_final_marathon()
