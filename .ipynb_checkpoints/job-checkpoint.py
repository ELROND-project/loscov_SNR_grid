import sys

##############################################################################################################################
################################################ 1. PRELIMINARIES ############################################################
##############################################################################################################################

from config import *                                #all constants, defined in the config.py file

from functions.useful_functions import *            #useful functions, defined in the functions/useful_functions.py file
from functions.distributions_and_correlations import *

load_correlations(filename="correlations")
distributions =  load_file("distributions")
add_dict(distributions)

from functions.covariance.LLLL_cov import *
from functions.covariance.LeLe_cov import *
from functions.covariance.LLLe_cov import *
from functions.covariance.LpLp_cov import *
from functions.covariance.LLLp_cov import *
from functions.covariance.LpLe_cov import *

##############################################################################################################################
######################################### 5. GENERATING COVARIANCE MATRICES ##################################################
##############################################################################################################################

def process_pair(args):
    """Handles correlation function computations and file saving."""
    b1, b2, distributions, sigma_noise, sigma_shape, cov_matrix = args

    # Create output directory if it doesn't exist
    output_dir = f"{matrices_folder}/{cov_matrix}"
    os.makedirs(output_dir, exist_ok=True)

    # Generate binned correlation functions
    correlation_data = generate_binned_correlation(distributions, cov_matrix, b1, b2)

    if correlation_data:
        for suffix, data in correlation_data.items():
            filename = f"{output_dir}/{suffix}{b1}" if b1 is not None else f"{output_dir}/{suffix}"
            save_pickle(data, filename, f"{suffix} for b1={b1}, b2={b2}, cov_matrix={cov_matrix}")
            
def compute_covariance_piece(args):
    """Computes a specific covariance component."""
    b1, b2, distributions, sigma_noise, sigma_shape, cov_matrix, cov_type = args

    func = globals().get(f"generate_{cov_type}_{cov_matrix}")
    if func is None:
        print(f"Error: Covariance function {cov_type}_{cov_matrix} not found.")
        return (b1, b2, cov_matrix, cov_type, None)

    if cov_type == "ncov":
        result = func(distributions, sigma_noise, sigma_shape, b1, b2, use_approx)
    else:
        result = func(distributions, b1, b2, use_approx)

    return (b1, b2, cov_matrix, cov_type, {
        "full": result[0],
        "pp": result[1],
        "mm": result[2],
        "pm": result[3],
        "mp": result[4]
    })

# Read command-line arguments
b1, b2, cov_matrix, cov_type = sys.argv[1:5]

# Convert arguments to correct types
b1 = None if b1 == "None" else int(b1)
b2 = None if b2 == "None" else int(b2)

# Compute correlation function
process_pair((b1, b2, distributions, sigma_noise, sigma_shape, cov_matrix))

# Compute covariance piece
result = compute_covariance_piece((b1, b2, distributions, sigma_noise, sigma_shape, cov_matrix, cov_type))

# Save the result immediately
if result is not None:
    b1, b2, cov_matrix, cov_type, data = result

    if data is not None:
        output_dir = f"{matrices_folder}/{cov_matrix}"
        os.makedirs(output_dir, exist_ok=True)

        filename = f"{output_dir}/{cov_type}"
        if b1 is not None:
            filename += f"{b1}"
        if b2 is not None:
            filename += f"{b2}"

        save_pickle(data, filename, f"{cov_type} for b1={b1}, b2={b2}, cov_matrix={cov_matrix}")

print(f"Finished task: {b1}, {b2}, {cov_matrix}, {cov_type}")