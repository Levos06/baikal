import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import re

# 1. Load and prepare alpha data
epochs, c_values = [], []
with open('2026-03-25_learnable_c_water/src/c_water_data.txt', 'r') as f:
    for line in f:
        matches = re.findall(r"([0-9.]+)", line)
        if len(matches) >= 2:
            epochs.append(float(matches[0]))
            c_values.append(float(matches[-1]))

c_values = np.array(c_values)
x = np.array(epochs)

# Model inversion: alpha = logit((c - 0.220) / 0.01)
def get_alpha(c):
    sig = (c - 0.220) / 0.01
    sig = np.clip(sig, 1e-7, 1 - 1e-7)
    return np.log(sig / (1 - sig))

y = get_alpha(c_values)

# 2. Define fit function (User Formula)
def fit_func(x, a, b, S, O):
    arg = -a * x
    arg = np.clip(arg, -50, 50)
    sigmoid = 1.0 / (1.0 + np.exp(arg))
    return S * (sigmoid + b * x - 0.5) + O

# Initial guess: S ~ 0.3 (range of alpha), a ~ 0.01, b ~ 0.001, O ~ 0.0
p0 = [0.01, 0.001, 0.3, 0.0]

popt, pcov = curve_fit(fit_func, x, y, p0=p0, maxfev=10000)
a_fit, b_fit, S_fit, O_fit = popt

print(f"Fitted parameters for ALPHA:")
print(f"a = {a_fit:.8f}")
print(f"b = {b_fit:.8f}")
print(f"S = {S_fit:.8f}")
print(f"O = {O_fit:.8f}")

# 3. Visualization
y_fit = fit_func(x, *popt)
from sklearn.metrics import r2_score
r2 = r2_score(y, y_fit)

plt.figure(figsize=(10, 6))
plt.scatter(x, y, label='Actual alpha (from log)', color='purple', s=10, alpha=0.4)
plt.plot(x, y_fit, label=f'Fit (R²={r2:.5f})', color='red', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Internal Weight alpha')
plt.title('Approximation of Internal Parameter alpha Evolution')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig('2026-03-25_learnable_c_water/plots/alpha_approximation.png')

print(f"Plot saved to 2026-03-25_learnable_c_water/plots/alpha_approximation.png")
print(f"R2 Score for alpha fit: {r2:.8f}")
