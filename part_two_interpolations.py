##############################################################################################################################
#################################################### 1. IMPORTS ##############################################################
##############################################################################################################################

from config import *                                #all constants, defined in the config.py file
from functions.useful_functions import *            #useful functions, defined in the functions/useful_functions.py file
from functions.angular_distributions import *      #angular distributions class and functions


thetas = np.logspace(
    np.log10(theta_min_interpolation),
    np.log10(theta_max_interpolation),
    theta_res_interpolation
)

load_correlations(filename=f"correlations_NE={Nbinz_E}_NP={Nbinz_P}{correlation_notes}")
redshift_distributions = load_file(f"data/redshift_distributions")

if len(sys.argv) == 2:
    task = sys.argv[1]
else:
    print('Invalid task array specification in part 2 step 1')

from functions.covariance.LLLL import *
from functions.covariance.LELE import *
from functions.covariance.LPLP import *

def split_array(array, n_chunks):
        return np.array_split(array, n_chunks)

##############################################################################################################################
#################################################### 2. COSMIC ###############################################################
##############################################################################################################################

################################################ 2.1 LLLL_ccov ###############################################################
##############################################################################################################################

if task == 'LLLL_ccov':

    n_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))

    def evaluate_chunk(theta_chunk):
        plus = []
        minus = []
        
        for theta in theta_chunk:
            p, m = LLLL_ccov_v_theta(theta)
            plus.append(p)
            minus.append(m)
            
        return np.array(plus), np.array(minus)

    theta_chunks = split_array(thetas, n_cpus)

    with Pool(processes=n_cpus) as pool:
        results = pool.map(evaluate_chunk, theta_chunks)

    plus_values = np.concatenate([r[0] for r in results])
    minus_values = np.concatenate([r[1] for r in results])
    
    # LELE_ccov_plus_values, LELE_ccov_minus_values = LELE_ccov_v_theta(thetas)

    LLLL_ccov_plus = interpolation(thetas, plus_values)
    LLLL_ccov_minus = interpolation(thetas, minus_values)

    save_pickle(LLLL_ccov_plus, f'data/Interpolations/LLLL_ccov_plus', f"Saved LLLL_ccov_plus")
    save_pickle(LLLL_ccov_minus, f'data/Interpolations/LLLL_ccov_minus', f"Saved LLLL_ccov_minus")

################################################ 2.2 LELE_ccov ###############################################################
##############################################################################################################################

elif task == 'LELE_ccov':

    n_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))

    def evaluate_chunk(theta_chunk):
        plus = []
        minus = []
        
        for theta in theta_chunk:
            p, m = LELE_ccov_v_theta(theta)
            plus.append(p)
            minus.append(m)
            
        return np.array(plus), np.array(minus)

    theta_chunks = split_array(thetas, n_cpus)

    with Pool(processes=n_cpus) as pool:
        results = pool.map(evaluate_chunk, theta_chunks)

    plus_values = np.concatenate([r[0] for r in results])
    minus_values = np.concatenate([r[1] for r in results])
    
    # LELE_ccov_plus_values, LELE_ccov_minus_values = LELE_ccov_v_theta(thetas)
    
    LELE_ccov_plus = interpolation(thetas, plus_values)
    LELE_ccov_minus = interpolation(thetas, minus_values)
    
    save_pickle(LELE_ccov_plus, f'data/Interpolations/LELE_ccov_plus', f"Saved LELE_ccov_plus")
    save_pickle(LELE_ccov_minus, f'data/Interpolations/LELE_ccov_minus', f"Saved LELE_ccov_minus")

################################################ 2.3 LPLP_ccov ###############################################################
##############################################################################################################################

elif task == 'LPLP_ccov':

    n_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))

    def evaluate_chunk(theta_chunk):
        values = []
        
        for theta in theta_chunk:
            v = LPLP_ccov_v_theta(theta)
            values.append(v)
            
        return np.array(values)

    theta_chunks = split_array(thetas, n_cpus)

    with Pool(processes=n_cpus) as pool:
        results = pool.map(evaluate_chunk, theta_chunks)

    values = np.concatenate([r for r in results])
    
    # LPLP_ccov_values = LPLP_ccov_v_theta(thetas)
    
    LPLP_ccov = interpolation(thetas, values)
    
    save_pickle(LPLP_ccov, f'data/Interpolations/LPLP_ccov', f"Saved LPLP_ccov")

##############################################################################################################################
##################################################### 3. Noise ###############################################################
##############################################################################################################################

################################################ 3.1 LLLL_ncov ###############################################################
##############################################################################################################################

elif task == 'LLLL_ncov':

    n_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))

    def evaluate_chunk(theta_chunk):
        pp_list = []
        px_list = []
        xp_list = []
        xx_list = []
        
        for theta in theta_chunk:
            pp, px, xp, xx = LLLL_ncov_v_theta(theta)
            pp_list.append(pp)
            px_list.append(px)
            xp_list.append(xp)
            xx_list.append(xx)
            
        return np.array(pp_list), np.array(px_list), np.array(xp_list), np.array(xx_list) 

    theta_chunks = split_array(thetas, n_cpus)

    with Pool(processes=n_cpus) as pool:
        results = pool.map(evaluate_chunk, theta_chunks)

    intpp_values = np.concatenate([r[0] for r in results])
    intpx_values = np.concatenate([r[1] for r in results])
    intxp_values = np.concatenate([r[2] for r in results])
    intxx_values = np.concatenate([r[3] for r in results])
    
    # LLLL_intpp_values, LLLL_intpx_values, LLLL_intxp_values, LLLL_intxx_values = LLLL_ncov_v_theta(thetas)
    
    LLLL_int_pp = interpolation(thetas, intpp_values)
    LLLL_int_px = interpolation(thetas, intpx_values)
    LLLL_int_xp = interpolation(thetas, intxp_values)
    LLLL_int_xx = interpolation(thetas, intxx_values)
    
    save_pickle(LLLL_int_pp, f'data/Interpolations/LLLL_int_pp', f"Saved LLLL_int_pp")
    save_pickle(LLLL_int_px, f'data/Interpolations/LLLL_int_px', f"Saved LLLL_int_px")
    save_pickle(LLLL_int_xp, f'data/Interpolations/LLLL_int_xp', f"Saved LLLL_int_xp")
    save_pickle(LLLL_int_xx, f'data/Interpolations/LLLL_int_xx', f"Saved LLLL_int_xx")

################################################ 3.2 LELE_ncov ###############################################################
##############################################################################################################################

elif task == 'LELE_ncov':

    n_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))

    def evaluate_chunk(theta_chunk):
        pp_list = []
        px_list = []
        xp_list = []
        xx_list = []
        
        for theta in theta_chunk:
            pp, px, xp, xx = LELE_ncov_v_theta(theta)
            pp_list.append(pp)
            px_list.append(px)
            xp_list.append(xp)
            xx_list.append(xx)
            
        return np.array(pp_list), np.array(px_list), np.array(xp_list), np.array(xx_list) 

    theta_chunks = split_array(thetas, n_cpus)

    with Pool(processes=n_cpus) as pool:
        results = pool.map(evaluate_chunk, theta_chunks)

    intpp_values = np.vstack([r[0] for r in results])
    intpx_values = np.vstack([r[1] for r in results])
    intxp_values = np.vstack([r[2] for r in results])
    intxx_values = np.vstack([r[3] for r in results])
    
    # LELE_intpp_values, LELE_intpx_values, LELE_intxp_values, LELE_intxx_values = LELE_ncov_v_theta(thetas)
    
    LELE_int_pp = [interpolation(thetas, intpp_values[:, 0]), interpolation(thetas, intpp_values[:, 1])]
    LELE_int_px = [interpolation(thetas, intpx_values[:, 0]), interpolation(thetas, intpx_values[:, 1])]
    LELE_int_xp = [interpolation(thetas, intxp_values[:, 0]), interpolation(thetas, intxp_values[:, 1])]
    LELE_int_xx = [interpolation(thetas, intxx_values[:, 0]), interpolation(thetas, intxx_values[:, 1])]
    
    save_pickle(LELE_int_pp, f'data/Interpolations/LELE_int_pp', f"Saved LELE_int_pp")
    save_pickle(LELE_int_px, f'data/Interpolations/LELE_int_px', f"Saved LELE_int_px")
    save_pickle(LELE_int_xp, f'data/Interpolations/LELE_int_xp', f"Saved LELE_int_xp")
    save_pickle(LELE_int_xx, f'data/Interpolations/LELE_int_xx', f"Saved LELE_int_xx")

################################################ 3.3 LPLP_ncov ###############################################################
##############################################################################################################################

elif task == 'LPLP_ncov':

    n_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))

    def evaluate_chunk(theta_chunk):
        values = []

        for theta in theta_chunk:
            values.append(LPLP_ncov_v_theta(theta))

        return np.array(values)

    theta_chunks = split_array(thetas, n_cpus)

    with Pool(processes=n_cpus) as pool:
        results = pool.map(evaluate_chunk, theta_chunks)

    int_values = np.vstack([r for r in results])
    
    # LPLP_int_values = LPLP_ncov_v_theta(thetas)
    
    LPLP_int = [interpolation(thetas, int_values[:, 0]), interpolation(thetas, int_values[:, 1])]
    
    save_pickle(LPLP_int, f'data/Interpolations/LPLP_int', f"Saved LPLP_int")

else:
    print("ERROR: task specification not valid")
