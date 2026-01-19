import numpy as np
from config import *

# Define parameter ranges
sigma_L_values = np.logspace(sigL_lower, sigL_upper, sigL_n) 
Nlens_values = np.logspace(Nlens_lower, Nlens_upper, Nlens_n, dtype=int) 

# Create all combinations
with open('params.txt', 'w') as f:
    for sigma_L in sigma_L_values:
        for Nlens in Nlens_values:
            f.write(f"{sigma_L:.6e} {Nlens}\n") 

total = len(sigma_L_values) * len(Nlens_values)
print(f"Total combinations: {total}")