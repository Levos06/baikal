import matplotlib.pyplot as plt
import re

epochs = []
c_values = []

with open('c_water_data.txt', 'r') as f:
    for line in f:
        # Try to find two numbers in the line
        matches = re.findall(r"([0-9.]+)", line)
        if len(matches) >= 2:
            # The first number is epoch, the last one is usually C_Water
            # In "Epoch 0001 | Time: 100.4s | C_Water: 0.225012", matches are ['0001', '100.4', '0.225012']
            # In "0190 0.225535", matches are ['0190', '0.225535']
            epochs.append(int(matches[0]))
            c_values.append(float(matches[-1]))

plt.figure(figsize=(10, 6))
plt.plot(epochs, c_values, label='Learned C_Water', color='blue', linewidth=2)
plt.axhline(y=0.225, color='red', linestyle='--', label='Initial C (0.225)')
plt.xlabel('Epoch')
plt.ylabel('C_Water (m/ns)')
plt.title('Evolution of Learned Speed of Light (C_Water) in Baikal Water')
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('2026-03-25_learnable_c_water/plots/c_water_evolution.png')
print(f"Plot saved to 2026-03-25_learnable_c_water/plots/c_water_evolution.png")
print(f"Final C_Water: {c_values[-1]:.6f}")
