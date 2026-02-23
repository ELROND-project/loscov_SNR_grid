#use if you want to created smoothed functions using a different cosmic_smoothing value

from config import *                                #all constants, defined in the config.py file
from functions.useful_functions import *            #useful functions, defined in the functions/useful_functions.py file
from functions.angular_distributions import *      #angular distributions class and functions

thetas = load_file(f'data/Interpolations/raw/thetas')
LLLL_ccov_plus_values = load_file(f'data/Interpolations/raw/LLLL_ccov_plus')
LLLL_ccov_minus_values = load_file(f'data/Interpolations/raw/LLLL_ccov_minus')
LELE_ccov_plus_values = load_file(f'data/Interpolations/raw/LELE_ccov_plus')
LELE_ccov_minus_values = load_file(f'data/Interpolations/raw/LELE_ccov_minus')
LPLP_ccov_values = load_file(f'data/Interpolations/raw/LPLP_ccov')

#LLLL
LLLL_ccov_plus_values_smooth  = smoothing(LLLL_ccov_plus_values,  sigma=cosmic_smoothing)
LLLL_ccov_minus_values_smooth = smoothing(LLLL_ccov_minus_values, sigma=cosmic_smoothing)

LLLL_ccov_plus_smooth  = interpolation(thetas, LLLL_ccov_plus_values_smooth)
LLLL_ccov_minus_smooth = interpolation(thetas, LLLL_ccov_minus_values_smooth)

save_pickle(LLLL_ccov_plus_smooth, f'data/Interpolations/LLLL_ccov_plus_smooth', f"Saved LLLL_ccov_plus_smooth")
save_pickle(LLLL_ccov_minus_smooth, f'data/Interpolations/LLLL_ccov_minus_smooth', f"Saved LLLL_ccov_minus_smooth")

#LELE
LELE_ccov_plus_values_smooth  = smoothing(LELE_ccov_plus_values,  sigma=cosmic_smoothing)
LELE_ccov_minus_values_smooth = smoothing(LELE_ccov_minus_values, sigma=cosmic_smoothing)

LELE_ccov_plus_smooth  = interpolation(thetas, LELE_ccov_plus_values_smooth)
LELE_ccov_minus_smooth = interpolation(thetas, LELE_ccov_minus_values_smooth)

save_pickle(LELE_ccov_plus_smooth, f'data/Interpolations/LELE_ccov_plus_smooth', f"Saved LELE_ccov_plus_smooth")
save_pickle(LELE_ccov_minus_smooth, f'data/Interpolations/LELE_ccov_minus_smooth', f"Saved LELE_ccov_minus_smooth")


#LPLP
LPLP_ccov_values_smooth = smoothing(LPLP_ccov_values,  sigma=cosmic_smoothing)

LPLP_ccov_smooth = interpolation(thetas, LPLP_ccov_values_smooth)

save_pickle(LPLP_ccov_smooth, f'data/Interpolations/LPLP_ccov_smooth', f"Saved LPLP_ccov_smooth")