import sys

##############################################################################################################################
################################################ 1. PRELIMINARIES ############################################################
###############################################to###############################################################################

if len(sys.argv) == 3:
    sigma_L = float(sys.argv[1])
    Nlens = int(float(sys.argv[2]))
else:
    print('problem!!!')
    
suffix = f'sigma_L={sigma_L}_Nlens={Nlens}'

from config import *                                #all constants, defined in the config.py file

from functions.useful_functions import *            #useful functions, defined in the functions/useful_functions.py file

load_correlations(filename=f'correlations_NE={Nbinz_E}_NP={Nbinz_P}{correlation_notes}')
redshift_distributions = load_file(f"data/redshift_distributions")

add_dict(redshift_distributions)

if smooth:
    LLLL_ccov_plus = load_file(f'data/Interpolations/LLLL_ccov_plus_smooth')
    LLLL_ccov_minus = load_file(f'data/Interpolations/LLLL_ccov_minus_smooth')
    LELE_ccov_plus = load_file(f'data/Interpolations/LELE_ccov_plus_smooth')
    LELE_ccov_minus = load_file(f'data/Interpolations/LELE_ccov_minus_smooth')
    LPLP_ccov = load_file(f'data/Interpolations/LPLP_ccov_smooth')

else: 
    LLLL_ccov_plus = load_file(f'data/Interpolations/LLLL_ccov_plus')
    LLLL_ccov_minus = load_file(f'data/Interpolations/LLLL_ccov_minus')
    LELE_ccov_plus = load_file(f'data/Interpolations/LELE_ccov_plus')
    LELE_ccov_minus = load_file(f'data/Interpolations/LELE_ccov_minus')
    LPLP_ccov = load_file(f'data/Interpolations/LPLP_ccov')

add_dict(LLLL_ccov_plus, LLLL_ccov_minus, LELE_ccov_plus, LELE_ccov_minus, LPLP_ccov)

LLLL_int_pp = load_file(f'data/Interpolations/LLLL_int_pp')
LLLL_int_px = load_file(f'data/Interpolations/LLLL_int_px')
LLLL_int_xp = load_file(f'data/Interpolations/LLLL_int_xp')
LLLL_int_xx = load_file(f'data/Interpolations/LLLL_int_xx')

add_dict(LLLL_int_pp, LLLL_int_px, LLLL_int_xp, LLLL_int_xx)

LELE_int_pp = load_file(f'data/Interpolations/LELE_int_pp')
LELE_int_px = load_file(f'data/Interpolations/LELE_int_px')
LELE_int_xp = load_file(f'data/Interpolations/LELE_int_xp')
LELE_int_xx = load_file(f'data/Interpolations/LELE_int_xx')

add_dict(LELE_int_pp, LELE_int_px, LELE_int_xp, LELE_int_xx)

LPLP_int = load_file(f'data/Interpolations/LPLP_int')

add_dict(LPLP_int)

get_item('LL_plus_primitive', 'LL_minus_primitive', 'LE_plus_primitive', 'LE_minus_primitive', 'LP_primitive')

from functions.angular_distributions import *

from functions.covariance.LLLL import *
from functions.covariance.LELE import *
from functions.covariance.LPLP import *

matrices_folder = f'data/covariance/{suffix}'

thetas = np.logspace(
    np.log10(theta_min_interpolation),
    np.log10(theta_max_interpolation),
    theta_res_interpolation
)

##############################################################################################################################
######################################################### 2. LLLL ############################################################
##############################################################################################################################

output_dir_LLLL = f"{matrices_folder}/LLLL"
os.makedirs(output_dir_LLLL, exist_ok=True) 

def LLLL_ncov(theta): 
    """ 
    a function which returns the noise and sparsity as a function of theta only (for use in 
    the optimisation). Works for the specific sigma_L and Nlens provided by the system 
    arguments of this run.

    returns [[ncov_plus, ncov_minus], [scov_plus, scov_minus]]
    """
    
    angular_distribution = Angular_Distributions(binscheme=[0,theta], Nbin_a=1, Thetamax=None)

    return generate_ncov_LLLL(sigma_L, Nlens, angular_distribution)

######################################################## 2.1 plus ############################################################
##############################################################################################################################

def total_variance_LLLL_plus(theta):
    """ 
    a function which returns the total LLLL plus variance as a function of theta
    (hence for use in the optimisation).
    """

    noise_sparsity = LLLL_ncov(theta) #generates the noise and sparsity as a function of theta

    noise = noise_sparsity[0] #extracts [noise_plus, noise_minus]
    sparsity = noise_sparsity[1] #extracts [sparsity_plus, sparsity_minus]

    noise_plus = noise[0] #extracts noise_plus
    sparsity_plus = sparsity[0] #extracts sparsity_plus
    cosmic_plus = LLLL_ccov_plus(theta).item()

    noise_plus = max(noise_plus,0.0)
    sparsity_plus = max(sparsity_plus,0.0)
    cosmic_plus = max(cosmic_plus,0.0)
    
    variance = noise_plus + sparsity_plus + cosmic_plus #the total variance
    
    return np.abs(variance)

values = []

for theta in thetas:
    values.append(total_variance_LLLL_plus(theta))

values = np.array(values)

mask = values > 0

thetas = thetas[mask]
values = values[mask]
    
#optimise the binning
LLLL_plus_optimised_theta, LLLL_plus_optimised_signal, LLLL_plus_optimised_std, LLLL_plus_optimised_SNR = optimise_bins(thetas, LL_plus_primitive, values)

#evaluate the noise and sparsity at the optimised theta
LLLL_plus_optimised_noise_sparsity = LLLL_ncov(LLLL_plus_optimised_theta)

LLLL_plus_optimised_noise = LLLL_plus_optimised_noise_sparsity[0][0] #extract the optimised noise_plus component
LLLL_plus_optimised_sparsity = LLLL_plus_optimised_noise_sparsity[1][0] #extract the optimised sparsity_plus component
LLLL_plus_optimised_cosmic = LLLL_ccov_plus(LLLL_plus_optimised_theta).item() #extract the optimised cosmic_plus component

LLLL_plus_dict = {'optimised_theta': LLLL_plus_optimised_theta,
                  'optimised_signal': LLLL_plus_optimised_signal,
                  'optimised_std': LLLL_plus_optimised_std,
                  'optimised_SNR': LLLL_plus_optimised_SNR,
                  'optimised_noise': LLLL_plus_optimised_noise,
                  'optimised_sparsity': LLLL_plus_optimised_sparsity,
                  'optimised_cosmic': LLLL_plus_optimised_cosmic}

filename_plus = f"{output_dir_LLLL}/plus"
save_pickle(LLLL_plus_dict, filename_plus, f"LLLL plus sigma_L = {sigma_L} Nlens = {Nlens}")

######################################################## 2.2 minus ############################################################
###############################################################################################################################

def total_variance_LLLL_minus(theta):
    """ 
    a function which returns the total LLLL minus variance as a function of theta
    (hence for use in the optimisation).
    """

    noise_sparsity = LLLL_ncov(theta) #generates the noise and sparsity as a function of theta

    noise = noise_sparsity[0] #extracts [noise_plus, noise_minus]
    sparsity = noise_sparsity[1] #extracts [sparsity_plus, sparsity_minus]

    noise_minus = noise[1] #extracts noise_minus
    sparsity_minus = sparsity[1] #extracts sparsity_minus
    cosmic_minus = LLLL_ccov_minus(theta).item()

    noise_minus = max(noise_minus,0.0)
    sparsity_minus = max(sparsity_minus,0.0)
    cosmic_minus = max(cosmic_minus,0.0)
    
    variance = noise_minus + sparsity_minus + cosmic_minus #the total variance
    
    return np.abs(variance)

values = []

for theta in thetas:
    values.append(total_variance_LLLL_minus(theta))

values = np.array(values)

mask = values > 0

thetas = thetas[mask]
values = values[mask]

#optimise the binning
LLLL_minus_optimised_theta, LLLL_minus_optimised_signal, LLLL_minus_optimised_std, LLLL_minus_optimised_SNR = optimise_bins(thetas, LL_minus_primitive, values)

#evaluate the noise and sparsity at the optimised theta
LLLL_minus_optimised_noise_sparsity = LLLL_ncov(LLLL_minus_optimised_theta)

LLLL_minus_optimised_noise = LLLL_minus_optimised_noise_sparsity[0][1] #extract the optimised noise_minus component
LLLL_minus_optimised_sparsity = LLLL_minus_optimised_noise_sparsity[1][1] #extract the optimised sparsity_minus component
LLLL_minus_optimised_cosmic = LLLL_ccov_minus(LLLL_minus_optimised_theta).item() #extract the optimised cosmic_minus component

LLLL_minus_dict = {'optimised_theta': LLLL_minus_optimised_theta,
                  'optimised_signal': LLLL_minus_optimised_signal,
                  'optimised_std': LLLL_minus_optimised_std,
                  'optimised_SNR': LLLL_minus_optimised_SNR,
                  'optimised_noise': LLLL_minus_optimised_noise,
                  'optimised_sparsity': LLLL_minus_optimised_sparsity,
                  'optimised_cosmic': LLLL_minus_optimised_cosmic}

filename_minus = f"{output_dir_LLLL}/minus"
save_pickle(LLLL_minus_dict, filename_minus, f"LLLL minus sigma_L = {sigma_L} Nlens = {Nlens}")

##############################################################################################################################
######################################################### 3. LELE ############################################################
##############################################################################################################################

output_dir_LELE = f"{matrices_folder}/LELE"
os.makedirs(output_dir_LELE, exist_ok=True) 

def LELE_ncov(theta): 
    """ 
    a function which returns the noise and sparsity as a function of theta only (for use in 
    the optimisation). Works for the specific sigma_L and Nlens provided by the system 
    arguments of this run.

    returns [[ncov_plus, ncov_minus], [scov_plus, scov_minus]]
    """
    
    angular_distribution = Angular_Distributions(binscheme=[0,theta], Nbin_a=1, Thetamax=None)

    return generate_ncov_LELE(sigma_L, Nlens, angular_distribution)

######################################################## 3.1 plus ############################################################
##############################################################################################################################

def total_variance_LELE_plus(theta):
    """ 
    a function which returns the total LELE plus variance as a function of theta
    (hence for use in the optimisation).
    """

    noise_sparsity = LELE_ncov(theta) #generates the noise and sparsity as a function of theta

    noise = noise_sparsity[0] #extracts [noise_plus, noise_minus]
    sparsity = noise_sparsity[1] #extracts [sparsity_plus, sparsity_minus]

    noise_plus = noise[0] #extracts noise_plus
    sparsity_plus = sparsity[0] #extracts sparsity_plus
    cosmic_plus = LELE_ccov_plus(theta).item()

    noise_plus = max(noise_plus,0.0)
    sparsity_plus = max(sparsity_plus,0.0)
    cosmic_plus = max(cosmic_plus,0.0)
    
    variance = noise_plus + sparsity_plus + cosmic_plus #the total variance
    
    return np.abs(variance)

values = []

for theta in thetas:
    values.append(total_variance_LELE_plus(theta))

values = np.array(values)

mask = values > 0

thetas = thetas[mask]
values = values[mask]

#optimise the binning
LELE_plus_optimised_theta, LELE_plus_optimised_signal, LELE_plus_optimised_std, LELE_plus_optimised_SNR = optimise_bins(thetas, LE_plus_primitive[0], values)

#evaluate the noise and sparsity at the optimised theta
LELE_plus_optimised_noise_sparsity = LELE_ncov(LELE_plus_optimised_theta)

LELE_plus_optimised_noise = LELE_plus_optimised_noise_sparsity[0][0] #extract the optimised noise_plus component
LELE_plus_optimised_sparsity = LELE_plus_optimised_noise_sparsity[1][0] #extract the optimised sparsity_plus component
LELE_plus_optimised_cosmic = LELE_ccov_plus(LELE_plus_optimised_theta).item() #extract the optimised cosmic_plus component

LELE_plus_dict = {'optimised_theta': LELE_plus_optimised_theta,
                  'optimised_signal': LELE_plus_optimised_signal,
                  'optimised_std': LELE_plus_optimised_std,
                  'optimised_SNR': LELE_plus_optimised_SNR,
                  'optimised_noise': LELE_plus_optimised_noise,
                  'optimised_sparsity': LELE_plus_optimised_sparsity,
                  'optimised_cosmic': LELE_plus_optimised_cosmic}

filename_plus = f"{output_dir_LELE}/plus"
save_pickle(LELE_plus_dict, filename_plus, f"LELE plus sigma_L = {sigma_L} Nlens = {Nlens}")

######################################################## 3.2 minus ############################################################
##############################################################################################################################

def total_variance_LELE_minus(theta):
    """ 
    a function which returns the total LELE minus variance as a function of theta
    (hence for use in the optimisation).
    """

    noise_sparsity = LELE_ncov(theta) #generates the noise and sparsity as a function of theta

    noise = noise_sparsity[0] #extracts [noise_plus, noise_minus]
    sparsity = noise_sparsity[1] #extracts [sparsity_plus, sparsity_minus]

    noise_minus = noise[1] #extracts noise_minus
    sparsity_minus = sparsity[1] #extracts sparsity_minus
    cosmic_minus = LELE_ccov_minus(theta).item()

    noise_minus = max(noise_minus,0.0)
    sparsity_minus = max(sparsity_minus,0.0)
    cosmic_minus = max(cosmic_minus,0.0)
    
    variance = noise_minus + sparsity_minus + cosmic_minus #the total variance
    
    return np.abs(variance)

values = []

for theta in thetas:
    values.append(total_variance_LELE_minus(theta))

values = np.array(values)

mask = values > 0

thetas = thetas[mask]
values = values[mask]

#optimise the binning
LELE_minus_optimised_theta, LELE_minus_optimised_signal, LELE_minus_optimised_std, LELE_minus_optimised_SNR = optimise_bins(thetas, LE_minus_primitive[0], values)

#evaluate the noise and sparsity at the optimised theta
LELE_minus_optimised_noise_sparsity = LELE_ncov(LELE_minus_optimised_theta)

LELE_minus_optimised_noise = LELE_minus_optimised_noise_sparsity[0][1] #extract the optimised noise_minus component
LELE_minus_optimised_sparsity = LELE_minus_optimised_noise_sparsity[1][1] #extract the optimised sparsity_minus component
LELE_minus_optimised_cosmic = LELE_ccov_minus(LELE_minus_optimised_theta).item() #extract the optimised cosmic_minus component

LELE_minus_dict = {'optimised_theta': LELE_minus_optimised_theta,
                  'optimised_signal': LELE_minus_optimised_signal,
                  'optimised_std': LELE_minus_optimised_std,
                  'optimised_SNR': LELE_minus_optimised_SNR,
                  'optimised_noise': LELE_minus_optimised_noise,
                  'optimised_sparsity': LELE_minus_optimised_sparsity,
                  'optimised_cosmic': LELE_minus_optimised_cosmic}

filename_minus = f"{output_dir_LELE}/minus"
save_pickle(LELE_minus_dict, filename_minus, f"LELE minus sigma_L = {sigma_L} Nlens = {Nlens}")

##############################################################################################################################
######################################################### 4. LPLP ############################################################
##############################################################################################################################

output_dir_LPLP = f"{matrices_folder}/LPLP"
os.makedirs(output_dir_LPLP, exist_ok=True) 

def LPLP_ncov(theta): 
    """ 
    a function which returns the noise and sparsity as a function of theta only (for use in 
    the optimisation). Works for the specific sigma_L and Nlens provided by the system 
    arguments of this run.

    returns [ncov, scov]
    """
    
    angular_distribution = Angular_Distributions(binscheme=[0,theta], Nbin_a=1, Thetamax=None)

    return generate_ncov_LPLP(sigma_L, Nlens, angular_distribution)

def total_variance_LPLP(theta):
    """ 
    a function which returns the total LPLP variance as a function of theta
    (hence for use in the optimisation).
    """

    noise_sparsity = LPLP_ncov(theta) #generates the noise and sparsity as a function of theta

    noise = noise_sparsity[0] #extracts noise
    sparsity = noise_sparsity[1] #extracts sparsity
    cosmic = LPLP_ccov(theta).item()

    noise = max(noise, 0.0)
    sparsity = max(sparsity, 0.0)
    cosmic = max(cosmic, 0.0)
    
    variance = noise + sparsity + cosmic #the total variance
    
    return np.abs(variance)

values = []

for theta in thetas:
    values.append(total_variance_LPLP(theta))

values = np.array(values)

mask = values > 0

thetas = thetas[mask]
values = values[mask]

#optimise the binning
LPLP_optimised_theta, LPLP_optimised_signal, LPLP_optimised_std, LPLP_optimised_SNR = optimise_bins(thetas, LP_primitive[0], values)

#evaluate the noise and sparsity at the optimised theta
LPLP_optimised_noise_sparsity = LPLP_ncov(LPLP_optimised_theta)

LPLP_optimised_noise = LPLP_optimised_noise_sparsity[0] #extract the optimised noise component
LPLP_optimised_sparsity = LPLP_optimised_noise_sparsity[1] #extract the optimised sparsity component
LPLP_optimised_cosmic = LPLP_ccov(LPLP_optimised_theta).item() #extract the optimised cosmic component

LPLP_dict = {'optimised_theta': LPLP_optimised_theta,
                  'optimised_signal': LPLP_optimised_signal,
                  'optimised_std': LPLP_optimised_std,
                  'optimised_SNR': LPLP_optimised_SNR,
                  'optimised_noise': LPLP_optimised_noise,
                  'optimised_sparsity': LPLP_optimised_sparsity,
                  'optimised_cosmic': LPLP_optimised_cosmic}

filename = f"{output_dir_LPLP}/data"
save_pickle(LPLP_dict, filename, f"LPLP sigma_L = {sigma_L} Nlens = {Nlens}")