import sys

##############################################################################################################################
################################################ 1. PRELIMINARIES ############################################################
###############################################to###############################################################################

if len(sys.argv) >= 3:
    sigma_L = float(sys.argv[1])
    Nlens = int(sys.argv[2])
else:
    print('problem!!!')
    
suffix = f'sigma_L={sigma_L}_Nlens={Nlens}'

from config import *                                #all constants, defined in the config.py file

from functions.useful_functions import *            #useful functions, defined in the functions/useful_functions.py file

load_correlations(filename=f'correlations_NE={Nbinz_E}_NP={Nbinz_P}{correlation_notes}')
angular_distributions = load_file(f"data/{suffix}/angular_distributions")
redshift_distributions = load_file(f"data/{suffix}/redshift_distributions")

add_dict(angular_distributions)
add_dict(redshift_distributions)

from functions.angular_distributions import *

from functions.covariance.LLLL import *
from functions.covariance.LELE import *
from functions.covariance.LPLP import *

##############################################################################################################################
######################################### 5. GENERATING COVARIANCE MATRICES ##################################################
##############################################################################################################################
            
def compute_covariance_piece(args):
    """Computes a specific covariance component."""
    b1, b2, cov_matrix, cov_type = args

    func = globals().get(f"generate_{cov_type}_{cov_matrix}")
    if func is None:
        print(f"Error: Covariance function {cov_type}_{cov_matrix} not found.")
        return (b1, b2, cov_matrix, cov_type, None)

    # Dynamically build argument list
    func_args = [sigma_L, Nlens]

    # Conditionally add b1 and b2 if they were given
    if b1 is not None:
        func_args.append(b1)
    if b2 is not None:
        func_args.append(b2)

    data, error = func(*func_args)

    return (b1, b2, cov_matrix, cov_type, data, error)

# Read command-line arguments
args = sys.argv[1:]

if len(args) == 5:
    b1 = b2 = None
    cov_matrix, cov_type = args
elif len(args) == 6:
    b1 = int(args[3])
    b2 = None
    cov_matrix, cov_type = args[4:]
elif len(args) == 7:
    b1 = int(args[3])
    b2 = int(args[4])
    cov_matrix, cov_type = args[5:]
else:
    # print("arguments = ", args)
    raise ValueError("Expected either 5, 6, or 7 arguments: sigma_L Nlens [b1] [b2] cov_matrix cov_type")

# Compute covariance piece
result = compute_covariance_piece((b1, b2, cov_matrix, cov_type))

def save_data(cov_matrix, cov_type):
    
    matrices_folder = f'data/{suffix}'

    #saving the data
    if data is not None:
        output_dir = f"{matrices_folder}/{cov_matrix}"
        os.makedirs(output_dir, exist_ok=True)

        filename = f"{output_dir}/{cov_type}"

        if cov_type == 'ncov':
            filename_s = f"{output_dir}/scov"

        if cov_type == 'ccov':
            save_pickle(data, filename, f"{cov_type}, cov_matrix={cov_matrix}")
        
        elif cov_type == 'ncov':
            save_pickle(data[0], filename, f"{cov_type}, cov_matrix={cov_matrix}")
            save_pickle(data[1], filename_s, f"scov, cov_matrix={cov_matrix}")

# Save the result immediately
if result is not None:
    b1, b2, cov_matrix, cov_type, data, error = result

    save_data(cov_matrix, cov_type)