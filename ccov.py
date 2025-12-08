import sys

##############################################################################################################################
################################################ 1. PRELIMINARIES ############################################################
###############################################to###############################################################################

from config import *                                #all constants, defined in the config.py file

from functions.useful_functions import *            #useful functions, defined in the functions/useful_functions.py file

load_correlations(filename=f'correlations_NE={Nbinz_E}_NP={Nbinz_P}{correlation_notes}')
angular_distributions = load_file(f"data/angular_distributions")
redshift_distributions = load_file(f"data/redshift_distributions")

add_dict(angular_distributions)
add_dict(redshift_distributions)

from functions.angular_distributions import *

from functions.covariance.LLLL import *
from functions.covariance.LELE import *
from functions.covariance.LPLP import *

##############################################################################################################################
######################################### 5. GENERATING COVARIANCE MATRICES ##################################################
##############################################################################################################################

# Loop through covariance matrices with fixed b1=1, b2=1, cov_type='ccov'
b1 = 0
b2 = 0
cov_type = 'ccov'
cov_matrix = ['LLLL', 'LELE', 'LPLP']

for cov_mat in cov_matrix:

    print(f"generate_{cov_type}_{cov_mat}")
    
    func = globals().get(f"generate_{cov_type}_{cov_mat}")
    
    if cov_mat == 'LLLL':
        data, error = func()
    else:
        data, error = func(b1,b2)
    
    matrices_folder = f'data/covariance'
    output_dir = f"{matrices_folder}/{cov_mat}"
    os.makedirs(output_dir, exist_ok=True)

    filename = f"{output_dir}/{cov_type}"

    save_pickle(data, filename, f"{cov_type}, cov_matrix={cov_mat}")