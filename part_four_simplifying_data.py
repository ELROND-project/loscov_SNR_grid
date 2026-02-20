from config import *                                #all constants, defined in the config.py file
from functions.useful_functions import *            #useful functions, defined in the functions/useful_functions.py file

# Read parameters from params.txt
sigma_L_array = []
Nlens_array = []

with open('params.txt', 'r') as f:
    for line in f:
        parts = line.strip().split()  # Split on any whitespace
        sigma_L_array.append(float(parts[0]))
        Nlens_array.append(int(float(parts[1])))

sigma_L_array = np.array(sigma_L_array)
Nlens_array = np.array(Nlens_array)

print(f"Loaded {len(sigma_L_array)} parameter combinations")

# Initialize arrays for covariance matrices
n_params = len(sigma_L_array)

LLLL_plus_dicts = []
LLLL_minus_dicts = []
LELE_plus_dicts = []
LELE_minus_dicts = []
LPLP_dicts = []

# Load covariance files for each parameter combination
for i, (sigma_L, Nlens) in enumerate(zip(sigma_L_array, Nlens_array)):
    base_dir = f'data/covariance/sigma_L={sigma_L}_Nlens={Nlens}'
    
    # Load LLLL_plus
    llll_plus_dir = os.path.join(base_dir, 'LLLL')
    with open(os.path.join(llll_plus_dir, 'plus'), 'rb') as f:
        data = pickle.load(f)
        LLLL_plus_dicts.append(data)
    
    # Load LLLL_minus
    llll_minus_dir = os.path.join(base_dir, 'LLLL')
    with open(os.path.join(llll_minus_dir, 'minus'), 'rb') as f:
        data = pickle.load(f)
        LLLL_minus_dicts.append(data)
    
    # Load LELE_plus
    lele_plus_dir = os.path.join(base_dir, 'LELE')
    with open(os.path.join(lele_plus_dir, 'plus'), 'rb') as f:
        data = pickle.load(f)
        LELE_plus_dicts.append(data)
    
    # Load LELE_minus
    lele_minus_dir = os.path.join(base_dir, 'LELE')
    with open(os.path.join(lele_minus_dir, 'minus'), 'rb') as f:
        data = pickle.load(f)
        LELE_minus_dicts.append(data)
    
    # Load LPLP
    lplp_dir = os.path.join(base_dir, 'LPLP')
    with open(os.path.join(lplp_dir, 'data'), 'rb') as f:
        data = pickle.load(f)
        LPLP_dicts.append(data)

full_dict = {'LLp': LLLL_plus_dicts,
            'LLm': LLLL_minus_dicts,
            'LEp': LELE_plus_dicts,
            'LEm': LELE_minus_dicts,
            'LP': LPLP_dicts,
            'sigL': sigma_L_array,
            'Nlens': Nlens_array}

save_pickle(full_dict, f'data/covariance_dict', f"Saved covariance dictionary")