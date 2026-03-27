import matplotlib.pyplot as plt
import re
import os

epochs = []
c_values = []

# Path to data file relative to root
data_file = '2026-03-25_learnable_c_water/src/c_water_data.txt'

if not os.path.exists(data_file):
    print(f"Error: {data_file} not found")
    exit(1)

with open(data_file, 'r') as f:
    for line in f:
        matches = re.findall(r"([0-9.]+)", line)
        if len(matches) >= 2:
            epochs.append(int(matches[0]))
            c_values.append(float(matches[-1]))

plt.figure(figsize=(10, 6))
plt.plot(epochs, c_values, label='Learned C_Water', color='blue', linewidth=2)
plt.axhline(y=0.225, color='red', linestyle='--', label='Initial C (0.225)')
plt.xlabel('Epoch')
plt.ylabel('C_Water (m/ns)')
plt.title('Evolution of Learned Speed of Light (1000 Epochs)')
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()

# Save to plots directory relative to root
os.makedirs('2026-03-25_learnable_c_water/plots', exist_ok=True)
plt.savefig('2026-03-25_learnable_c_water/plots/c_water_evolution_1000.png')
print(f"Plot saved to 2026-03-25_learnable_c_water/plots/c_water_evolution_1000.png")
print(f"Final C_Water: {c_values[-1]:.6f}")
