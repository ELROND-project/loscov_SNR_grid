import numpy as np

# Define your parameter ranges
sigma_L_values = np.logspace(-3, 1, 90) 
Nlens_values = np.logspace(1, 8, 90, dtype=int) 
# sigma_L_values = np.linspace(0.05, 0.2, 8) 
# Nlens_values = np.linspace(1000, 10000, 8, dtype=int) 
# sigma_L_values = np.array([0.04,0.06,0.08])
# Nlens_values = np.array([1e6,1e7,1e8])

# Create all combinations
with open('params.txt', 'w') as f:
    for sigma_L in sigma_L_values:
        for Nlens in Nlens_values:
            f.write(f"{sigma_L:.6e} {Nlens}\n")  # Using scientific notation

total = len(sigma_L_values) * len(Nlens_values)
print(f"Total combinations: {total}")
print(f"Use: #SBATCH --array=0-{total-1}")