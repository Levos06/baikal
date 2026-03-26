import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import re

# Load data
epochs, c_values = [], []
with open('c_water_data.txt', 'r') as f:
    for line in f:
        matches = re.findall(r"([0-9.]+)", line)
        if len(matches) >= 2:
            epochs.append(float(matches[0]))
            c_values.append(float(matches[-1]))

x = np.array(epochs)
y = np.array(c_values)

# User formula: f(x) = 1/(1 + e^{-ax}) + bx - 0.5
# We add S (scale) and O (offset) to match c_water range (~0.225)
def fit_func(x, a, b, S, O):
    # Stabilizing exp by clipping
    arg = -a * x
    arg = np.clip(arg, -50, 50)
    sigmoid = 1.0 / (1.0 + np.exp(arg))
    return S * (sigmoid + b * x - 0.5) + O

# Initial guess for [a, b, S, O]
# a: very small (linear region), b: very small, S: ~0.01 (range), O: ~0.225
p0 = [0.001, 0.0001, 0.01, 0.225]

popt, pcov = curve_fit(fit_func, x, y, p0=p0, maxfev=10000)
a_fit, b_fit, S_fit, O_fit = popt

print(f"Fitted parameters:")
print(f"a = {a_fit:.8f}")
print(f"b = {b_fit:.8f}")
print(f"S = {S_fit:.8f}")
print(f"O = {O_fit:.8f}")

# Generate fitted curve
y_fit = fit_func(x, *popt)

plt.figure(figsize=(10, 6))
plt.scatter(x, y, label='Real C_Water Data', color='gray', s=10, alpha=0.5)
plt.plot(x, y_fit, label='Fitted Model: S*(sigmoid(ax)+bx-0.5)+O', color='red', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('C_Water (m/ns)')
plt.title(f'Approximation of Learned Speed of Light\na={a_fit:.6f}, b={b_fit:.6f}')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig('2026-03-25_learnable_c_water/plots/c_water_approximation.png')

# Extrapolate to epoch 1000
x_future = np.arange(1, 1001)
y_future = fit_func(x_future, *popt)
print(f"Predicted C_Water at Epoch 1000: {y_future[-1]:.6f}")

plt.figure(figsize=(10, 6))
plt.plot(x_future, y_future, '--', label='Prediction (Future)', color='blue')
plt.scatter(x, y, color='gray', s=5, alpha=0.3, label='Historical Data')
plt.xlabel('Epoch')
plt.ylabel('C_Water (m/ns)')
plt.title('Extrapolation of C_Water to 1000 Epochs')
plt.grid(True)
plt.legend()
plt.savefig('2026-03-25_learnable_c_water/plots/c_water_prediction_1000.png')
