import numpy as np
import sys, platform, os

#the range over which to compute things (used in part one to create params.txt)
sigL_lower = -4
sigL_upper = 1
sigL_n = 90

Nlens_lower = 1
Nlens_upper = 10
Nlens_n = 90

compute_correlations = True     #computes the correlation functions necessary for calculating the covariance matrices,
                                 #and saves them in a pickle file. Set to False if this has already been done, which
                                 #saves computation time
                                 #Note, this is automatically set to True if the correlations were calculated for different
                                 #redshift binning, but NB if the number of bins stays the same but the binscheme changes,
                                 #it should be manually set to True to ensure the old correlations are recalculated

##############################################################################################################
############################################# Survey stats ###################################################
##############################################################################################################

sky_coverage = 15e3     #area of sky covered by the survey (deg^2)

NGal = 2e9              #total number of galaxies (expect 2e9)

######################################## Redshift distribution ##########################################

zmax_dist = 3           #the default maximum redshift being considered
Nbin_z = 1              #the default number of redshift bins

#automatically calculate the redshift bin limits
Nbinz_E = Nbin_z        #the number of redshift bins for galaxy shapes
Nbinz_P = Nbin_z      #the number of redshift bins for galaxy positions

binscheme_E = Nbinz_E
binscheme_P = Nbinz_P

zmax_E = zmax_dist
zmax_P = zmax_dist

#################################### Angles for correlation functions ##################################

Thetamin_arcmin = 0.1                            #minimum theta from which we calculate correlation functions (in arcmin)

Thetamin = Thetamin_arcmin / (60 * 180 / np.pi)  #minimum theta from which we calculate correlation functions (in radians)
Thetamax = np.pi                                 #maximum theta to which we calculate correlation functions (in radians)
nTheta = 10000                                   #number of points used to compute the correlation function

lmax = 1e8
nl = 1000

######################################## Angles for interpolations ##########################################

theta_max_interpolation_arcmin = 300 #the maximum theta that we interpolate to (arcminutes)

theta_min_interpolation = Thetamin #we're working in log space, so we don't start from 0
theta_max_interpolation = theta_max_interpolation_arcmin / (60 * 180 / np.pi)
theta_res_interpolation = 640 #the resolution of thetas in the interpolation (ideally divisible by 16)

Thetamax_LL = theta_max_interpolation          
Thetamax_LE = theta_max_interpolation            
Thetamax_LP = theta_max_interpolation

Thetamax_LL_plus = Thetamax_LL
Thetamax_LL_minus = Thetamax_LL

Thetamax_LE_plus = Thetamax_LE
Thetamax_LE_minus = Thetamax_LE
    
Omegatot = sky_coverage * (np.pi / 180)**2                                #sky coverage (in rad^2)
n_b = ( NGal / sky_coverage * (np.pi / 180)**2 ) / Nbin_z                 #number density of galaxies per redshift bin (in rad^-2)
r2_max = np.sqrt(Omegatot/np.pi)                                          #the maximum theta used in integrals which would normally run from 0 to infty 

thetamin_optimiser_arcmin = 0
thetamin_optimiser = thetamin_optimiser_arcmin / (60 * 180 / np.pi)

################################### cosmic variance smoothing ################################################

smooth = True #if True, make use of a Gaussian smoothing of the cosmic variance
smoothing_method = 'Gaussian' #'median' or 'Gaussian', for different smoothing methods
cosmic_smoothing = 30 #smoothing the cosmic variance (roughly how many adjacent points are considered. 0 for no smoothing)
smoothing_value = 0  #smoothing the interpolation 

########################################## cosmology #########################################################

H0=67.37       #Hubble constant
h=H0/100
ombh2=0.0223   #baryon density parameter
omch2=0.1198   #dark matter density parameter
ns = 0.965     #primordial power spectrum
c = 3e8        #speed of light

Omega_M = (ombh2 + omch2)/(h**2)
Omega_L = 1 - Omega_M

zmax = 7     #the maximum redshift to which we compute the Weyl power spectrum
kmax = 5e2 #(inverse Mpc) the maximum wavenumber, ie. the smallest spatial scales to which we determine the power spectrum
extrap_kmax = 1e10 #the maximum k for extrapolation beyond kmax
chimin = 1e-5

############################################ noise ###########################################################

sigma_E = np.sqrt(2) * 0.3                                            #the noise from the galaxy shapes on cosmic shear

##################################### numerical stuff ########################################################

max_cpus = 512
nsamp_string = '1e7'
nsamp = int(float(nsamp_string))
# nsamp = 2**19
Csamp = nsamp*2e2    #default number of samples in the Monte Carlo integrator for triple cosmic integrals
# Csamp = nsamp*(2**9)    #default number of samples in the Monte Carlo integrator for triple cosmic integrals
Nsamp = nsamp           #default number of samples in the Monte Carlo integrator for double noise/sparsity integrals
num_batches = 10000      #should be > maxsamp * 373 / (ram per node)
desired_error = 1       #percentage desired fractional error in integrals
confidence = 0.95       #confidence level for the error estimate (not currently used)
warning_level = 500     #level above which we print an integration error
total_error_threshold = 0.2  #the threshold for the total error on a term to be too high and to print a warning

################################################## Imports #####################################################

import matplotlib
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

from scipy import constants, special, integrate, stats
from scipy.interpolate import CubicSpline, RegularGridInterpolator
from hankel import HankelTransform

camb_path = os.path.realpath(os.path.join(os.getcwd(),'..'))
sys.path.insert(0,camb_path)
import camb
from camb import model, initialpower

import pickle
from multiprocessing import Pool, cpu_count
from multiprocessing import Process, Manager
from itertools import product
import inspect
from scipy.stats import norm, qmc
from scipy.integrate import quad
from scipy.optimize import root_scalar, minimize_scalar
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import median_filter
from scipy.ndimage import gaussian_filter1d

############################################### Cosmology #####################################################

# CAMB parameters
pars = camb.CAMBparams()                              #initialise the CAMBparams object, which contains all cosmological parameters and settings
pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2)   #define the cosmological model
pars.InitPower.set_params(ns=ns)                      #set the primordial power spectrum parameters
background = camb.get_background(pars)                #compute the background cosmological evolution

#this gives us an interpolator which can be used to generate the Weyl power spectrum for any range of z and k
Weyl_power_spectra = camb.get_matter_power_interpolator(pars, zmax=zmax, kmax=kmax, zs=None,
hubble_units=False, k_hunit=False, var1=model.Transfer_Weyl, var2=model.Transfer_Weyl, extrap_kmax=extrap_kmax)

############################################## shared variables ##############################################

# Shared variables
global_dict = {}  

####################### the suffix defining the folder names ###############################################

notes = '' #anything particularly unique about a particular run (eg different redshift binning)
correlation_notes = ''    #needed only to specify that a particular binscheme has been used

def format_sci(n):
    return f'{n:.0e}'.replace('+00', '').replace('+0', '').replace('+', '').replace('-0', '-') 

if not os.path.exists(f'correlations_NE={Nbinz_E}_NP={Nbinz_P}{correlation_notes}'):
    compute_correlations = True      