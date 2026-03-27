import numpy as np
import matplotlib.pyplot as plt
import re

# Load data
epochs, c_values = [], []
with open('2026-03-25_learnable_c_water/src/c_water_data.txt', 'r') as f:
    for line in f:
        matches = re.findall(r"([0-9.]+)", line)
        if len(matches) >= 2:
            epochs.append(float(matches[0]))
            c_values.append(float(matches[-1]))

c_values = np.array(c_values)
epochs = np.array(epochs)

# Invert model formula: c = 0.220 + 0.01 * sigmoid(alpha)
# sigmoid(alpha) = (c - 0.220) / 0.01
# alpha = -ln(1 / sig - 1) = logit(sig)
def get_alpha(c):
    sig = (c - 0.220) / 0.01
    sig = np.clip(sig, 1e-7, 1 - 1e-7)
    return np.log(sig / (1 - sig))

alphas = get_alpha(c_values)

plt.figure(figsize=(10, 6))
plt.plot(epochs, alphas, label='Internal Model Parameter (alpha)', color='purple', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Alpha (Internal Weight)')
plt.title('Evolution of Internal Learnable Parameter alpha')
plt.grid(True, linestyle='--', alpha=0.6)

# Fit a straight line to alpha
z = np.polyfit(epochs, alphas, 1)
p = np.poly1d(z)
plt.plot(epochs, p(epochs), "r--", alpha=0.8, label=f'Linear Trend (slope={z[0]:.6f})')

plt.legend()
plt.tight_layout()
plt.savefig('2026-03-25_learnable_c_water/plots/internal_alpha_evolution.png')

print(f"Internal alpha trend:")
print(f"Initial alpha (epoch 1): {alphas[0]:.6f}")
print(f"Current alpha (epoch {int(epochs[-1])}): {alphas[-1]:.6f}")
print(f"Linearity (R^2): {np.corrcoef(epochs, alphas)[0,1]**2:.8f}")
