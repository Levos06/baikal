import matplotlib.pyplot as plt
import csv
import os

def plot_load():
    input_file = 'logs/last_hour_gpu.csv'
    output_file = '2026-03-27_full_mlp_optimized_1500ep/plots/gpu_load_profile.png'
    
    util = []
    power = []
    
    with open(input_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 9: continue
            try:
                gpu_id = int(row[1])
                if gpu_id == 0: # RTX 4090
                    util.append(float(row[3]))
                    power.append(float(row[8]))
            except: continue

    if not util:
        print("No data for GPU 0 found")
        return

    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('GPU Utilization (%)', color='tab:blue')
    ax1.plot(util, color='tab:blue', alpha=0.5, label='GPU Util (%)')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_ylim(0, 105)
    ax1.grid(True, axis='y', alpha=0.3)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Power Draw (Watts)', color='tab:red')
    ax2.plot(power, color='tab:red', alpha=0.3, label='Power (W)')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    
    plt.title('High-Resolution GPU Load Profile (Last Hour)\nRTX 4090 | 1 Sample per Second')
    fig.tight_layout()
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    plot_load()
