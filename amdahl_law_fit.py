import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Amdahl's Law model
def amdahls_law(n_procs, F):
    return 1 / ((1 - F) + (F / n_procs))

# Plot Amdahl's Law fit to speedup data
n_processes = np.arange(1, 65536)
speed_up_95 = amdahls_law(n_processes, 0.95)  # Example with F = 0.9
speed_up_90 = amdahls_law(n_processes, 0.90)  # Example with F = 0.9
speed_up_75 = amdahls_law(n_processes, 0.75)  # Example with F = 0.9

plt.figure(figsize=(8, 5))
#plt.plot(n_processes, speed_up, '-', label='Measured speedup')
plt.semilogx(n_processes, speed_up_95, '-')
plt.semilogx(n_processes, speed_up_90, '-')
plt.semilogx(n_processes, speed_up_75, '-')
plt.xlabel('Number of Processes')
plt.ylabel('Speedup')
plt.title('Speedup vs Number of Processes')
plt.legend(["F = 0.95", "F = 0.90", "F = 0.75"], loc='upper left', fontsize=12)  # Adjust legend labels as needed
plt.grid(True)
plt.show()

# Your data
n_processes = np.array([1, 2, 4, 8, 16, 32])  # example
times1 = [0.3552052503257406, 0.3007059870436738, 0.1455757122702553, 0.08730617339763436, 0.06059182422658051, 0.049057833301184676]
times2 = [0.3537135922365852, 0.2608217356733386, 0.12369120546679482, 0.0682518230090958, 0.03264487462237543, 0.01875642729662863]
times3 = [0.38628819112283974, 0.27335328918560114, 0.13695697057602405, 0.07050651718925488, 0.036699544365788546, 0.02111041104183528]
times = np.mean(np.stack([times1, times2, times3]), axis=0)

speedup = [times[0] / time for time in times]  # speed-up is fraction: old_time / new_time
print(times[-1])
#speedups = np.array([1.0, 1.8, 3.2, 5.5, 7.0, 8.0])  # your actual measurements

# Fit the model
popt, pcov = curve_fit(amdahls_law, n_processes, speedup, bounds=(0, 1))
F_estimated = popt[0]

# Generate fitted curve
n_proc_fit_max = 128 # max(n_processes)
n_fit = np.logspace(np.log10(min(n_processes)), np.log10(n_proc_fit_max), 200)
speedup_fit = amdahls_law(n_fit, F_estimated)

# Plot with semilogx
plt.figure(figsize=(12, 6))
plt.semilogx(n_processes, speedup, 'o', label='Measured speedup')
plt.semilogx(n_fit, speedup_fit, '-', label=f'Amdahl fit (F ≈ {F_estimated:.3f})')

# Ticks: make sure they appear at your actual process counts
n_processes = np.array([1, 2, 4, 8, 16, 32, 64, 128])  # example
plt.xticks(n_processes, labels=[str(n) for n in n_processes])
plt.xlabel('No. of dataloader processes', fontsize=14)
plt.ylabel('Speed-up', fontsize=14)
plt.title('Amdahl\'s law fit', fontsize=16)
plt.grid(True, which="both", linestyle='--', linewidth=0.5)
plt.legend(fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig('figures/amdahls_law_fit_no_cache.pdf', dpi=300, bbox_inches='tight')



# # Your data
# n_processes = np.array([1, 2, 4, 8, 16, 24])  # example
# speedups = np.array([1.0, 1.8, 3.2, 5.5, 7.0, 8.0])  # replace with your measurements
#
# # Fit the model
# popt, pcov = curve_fit(amdahls_law, n_processes, speedups, bounds=(0, 1))
# F_estimated = popt[0]
#
# # Plot results
# n_fit = np.linspace(1, max(n_processes), 100)
# speedup_fit = amdahls_law(n_fit, F_estimated)
#
# plt.figure(figsize=(8, 5))
# plt.semilogx(n_processes, speedups, 'o', label='Measured speedup')
# plt.semilogx(n_fit, speedup_fit, '-', label=f'Amdahl fit (F ≈ {F_estimated:.3f})')
# plt.xlabel('Number of Processes')
# plt.ylabel('Speedup')
# plt.xticks(n_processes)
# plt.title('Speedup vs Number of Processes')
# plt.legend()
# plt.grid(True)
# plt.show()