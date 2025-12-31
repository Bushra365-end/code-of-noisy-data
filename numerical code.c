
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline

np.random.seed(42)

x = np.linspace(0, 4 * np.pi, 200)
dx = x[1] - x[0]

y_clean = np.sin(x) + 0.5 * np.cos(2 * x)
dy_clean = np.cos(x) - np.sin(2 * x)
noise_level = 0.15
noise = np.random.normal(0, noise_level, size=len(x))
y_noisy = y_clean + noise


dy_noisy = np.gradient(y_noisy, dx)

window_length = 31
poly_order = 3

y_savgol = savgol_filter(y_noisy, window_length, poly_order)

dy_savgol = savgol_filter(y_noisy, window_length, poly_order, deriv=1, delta=dx)

spline = UnivariateSpline(x, y_noisy, s=2)
y_spline = spline(x)
dy_spline = spline.derivative()(x)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

ax1.set_title("Raw vs Smoothed Data (Signal Reconstruction)", fontsize=14)
ax1.plot(x, y_clean, 'k--', label='True Signal (Hidden)', alpha=0.6)
ax1.plot(x, y_noisy, 'gray', label='Noisy Raw Data', alpha=0.5)
ax1.plot(x, y_savgol, 'r-', linewidth=2, label='Smoothed (Savitzky-Golay)')
ax1.plot(x, y_spline, 'b:', linewidth=2, label='Smoothed (Spline)')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylabel("Amplitude")
ax2.set_title("Derivative Comparison (Noise Amplification)", fontsize=14)
ax2.plot(x, dy_clean, 'k--', label='True Derivative', alpha=0.8)

ax2.plot(x, dy_noisy, 'gray', alpha=0.5, label='Finite Diff on Noisy Data')

ax2.plot(x, dy_savgol, 'r-', linewidth=2, label='Savitzky-Golay Derivative')
ax2.plot(x, dy_spline, 'b:', linewidth=2, label='Spline Derivative')

ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)
ax2.set_xlabel("X")
ax2.set_ylabel("dy/dx")
ax2.set_ylim(-4, 4)

plt.tight_layout()
plt.show()
