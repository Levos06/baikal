import matplotlib.pyplot as plt
import re
import numpy as np

def plot_marathon_metrics():
    log_path = '2026-03-28_extended_21features/train.log'
    epochs = []
    p9_vals = []
    
    with open(log_path, 'r') as f:
        for line in f:
            # Epoch 0123 | ... | Val P@R0.9: 0.8567
            e_match = re.search(r"Epoch (\d+)", line)
            p_match = re.search(r"Val P@R0.9:\s+([0-9.]+)", line)
            if e_match and p_match:
                epochs.append(int(e_match.group(1)))
                p9_vals.append(float(p_match.group(1)))
    
    if not epochs:
        print("No metrics found in log")
        return

    plt.figure(figsize=(12, 7))
    plt.plot(epochs, p9_vals, color='blue', alpha=0.3, label='Val P@R0.9')
    
    # Simple moving average for trend
    if len(p9_vals) > 20:
        window = 20
        smooth_vals = np.convolve(p9_vals, np.ones(window)/window, mode='valid')
        plt.plot(epochs[window-1:], smooth_vals, color='darkblue', linewidth=2, label=f'Trend (MA {window})')

    plt.axhline(y=0.90, color='red', linestyle='--', alpha=0.5, label='Target 0.90')
    plt.axhline(y=max(p9_vals), color='green', linestyle=':', alpha=0.8, label=f'Peak {max(p9_vals):.4f}')
    
    plt.xlabel('Epoch')
    plt.ylabel('P@R0.9 Score')
    plt.title('Marathon Results: 21 Features Model (1500 Epochs)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.ylim(0.4, 0.95)
    plt.tight_layout()
    
    output_path = '2026-03-28_extended_21features/plots/marathon_performance_1500.png'
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    print(f"Max Val P@R0.9: {max(p9_vals):.4f} (Epoch {epochs[p9_vals.index(max(p9_vals))]})")

if __name__ == "__main__":
    plot_marathon_metrics()
