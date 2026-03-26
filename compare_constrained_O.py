import numpy as np
from scipy.optimize import curve_fit
import re
from sklearn.metrics import r2_score, mean_squared_error

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

# Constrained fit function: O is FIXED at 0.225
def fit_func_fixed_O(x, a, b, S):
    O = 0.22500000
    arg = -a * x
    arg = np.clip(arg, -50, 50)
    sigmoid = 1.0 / (1.0 + np.exp(arg))
    return S * (sigmoid + b * x - 0.5) + O

# Initial guess for [a, b, S]
p0 = [0.05, 0.001, 0.0006]

popt, pcov = curve_fit(fit_func_fixed_O, x, y, p0=p0, maxfev=10000)
a_fit, b_fit, S_fit = popt

y_pred_constrained = fit_func_fixed_O(x, a_fit, b_fit, S_fit)

# Metrics for constrained fit
r2_c = r2_score(y, y_pred_constrained)
rmse_c = np.sqrt(mean_squared_error(y, y_pred_constrained))

# Previous metrics for unconstrained fit (O=0.22500581)
r2_u = 0.99989231
rmse_u = 2.0339e-06

print(f"--- Constrained Fit (O = 0.225 exactly) ---")
print(f"a = {a_fit:.8f}")
print(f"b = {b_fit:.8f}")
print(f"S = {S_fit:.8f}")
print(f"O = 0.22500000 (FIXED)")
print(f"\n--- Accuracy Comparison ---")
print(f"Unconstrained R^2: {r2_u:.8f}")
print(f"Constrained R^2:   {r2_c:.8f}")
print(f"Degradation (RMSE ratio): {rmse_c / rmse_u:.2f}x")
print(f"-------------------------------------------")
