Nlens = 1e5             #number of lenses
sky_coverage = 15e3     #area of sky covered by the survey
Nbina = 5               #number of angular separation bins
NGal = 1e9              #total number of galaxies
nsamp = int(1e6)        #number of samples in the Monte Carlo integrator
sigma_noise = 0.05      #noise on the LOS shear
n_b = 1e8               #number density of galaxies per redshift bin (in radians^-2)
nTheta = 1000           #number of points used to compute the correlation function
Thetamax_dist = 3e2     #maximum theta to calculate distributions inputted into covariance matrices (arcmin?)

import numpy as np

sigma_shape = np.sqrt(2) * 0.3         #the noise from the galaxy shapes on cosmic shear

##############################################################################################################
########################################### 1. PRELIMINARIES #################################################
##############################################################################################################

# Load packages, including CAMB
import sys, platform, os
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt

from scipy import constants, special, integrate, stats
from scipy.interpolate import CubicSpline, RegularGridInterpolator
from hankel import HankelTransform

camb_path = os.path.realpath(os.path.join(os.getcwd(),'..'))
sys.path.insert(0,camb_path)
import camb
from camb import model, initialpower

import pickle
from multiprocessing import Pool  

############################################ USEFUL FUNCTIONS ###########################################

### converting angles

def radtoarcmin(angle_rad):
    """
    This function converts an an angle expressed in radians
    into arcmins.
    """
    
    angle_arcmin = angle_rad * 60 * 180 / np.pi
    
    return angle_arcmin


def arcmintorad(angle_arcmin):
    """
    This function converts an an angle expressed in arcmins
    into radians.
    """
    
    angle_rad = angle_arcmin / (60 * 180 / np.pi)
    
    return angle_rad

################################### Correlation function bounds #####################################

Thetamin = arcmintorad(1)
Thetamax = np.pi
nTheta = 10000          #the maximum angular separation we consider

###################################### Monte Carlo Integrator #######################################

def monte_carlo_integrate(func, bounds, num_samples=nsamp, confidence=0.95):
    """
    Monte Carlo integration over a given domain with error estimation.
    
    Parameters:
    - func (callable): The function to integrate. NB it needs to accept an array as input
                       for integration over multiple dimensions.
    - bounds (list of tuples): Integration bounds [(a1, b1), (a2, b2), ...].
                                For 1D, use [(a, b)].
    - num_samples (int): Number of random samples to use (10^6 gives good result, but with largish errors)
    - confidence (float): Confidence level for the error estimate (default is 0.95) (this is something ChatGPT
                             recommended when I was figuring out how to estimate the error)
    
    Returns:
    - tuple: (float, float) Estimated value of the integral and its error.
    """
    
    # Determine the dimensionality and the volume of the integration domain
    dim = len(bounds)
    volumes = [b - a for a, b in bounds]
    total_volume = np.prod(volumes)
    rng = np.random.default_rng()
    
    # Generate random samples within the bounds
    samples = np.array([rng.uniform(low=a, high=b, size=num_samples) for a, b in bounds])
    
    # Evaluate the function at the random sample points
    values = func(samples)
    
    # Compute the Monte Carlo estimate of the integral
    integral = total_volume * np.mean(values)
    
    # Estimate the standard error
    variance = np.var(values, ddof=1)  # Sample variance
    std_error = total_volume * np.sqrt(variance / num_samples)
    
    # Compute the confidence interval using the normal distribution
    from scipy.stats import norm
    z = norm.ppf(1 - (1 - confidence) / 2)  # z-score for the confidence level
    error = z * std_error

    if error/integral > 0.1:
        print("WARNING: integration error = " + str(int(100 * error/integral)) + "%")
    
    return integral, error

####################################### Computing Weyl power spectrum #######################################################

# CAMB parameters
pars = camb.CAMBparams() #initialise the CAMBparams object, which contains all cosmological parameters and settings
pars.set_cosmology(H0=67.37, ombh2=0.0223, omch2=0.1198) #define the cosmological model
pars.InitPower.set_params(ns=0.965) #set the primordial power spectrum parameters
background = camb.get_background(pars) #compute the background cosmological evolution

zmax = 7     #the maximum redshift to which we compute the Weyl power spectrum
kmax = 5e2 #(inverse Mpc) the maximum wavenumber, ie. the smallest spatial scales to which we determine the power spectrum
extrap_kmax = 1e10 #the maximum k for extrapolation beyond kmax

#this gives us an interpolator which can be used to generate the Weyl power spectrum for any range of z and k
Weyl_power_spectra = camb.get_matter_power_interpolator(pars, zmax=zmax, kmax=kmax, zs=None,
hubble_units=False, k_hunit=False, var1=model.Transfer_Weyl, var2=model.Transfer_Weyl, extrap_kmax=extrap_kmax)

################################################ Euclid lenses ##############################################################

#reading in the forecasted sample of Euclid lenses
Euclid_lenses = np.loadtxt('lenses_Euclid.txt')
zd = Euclid_lenses[:, 0]
zs = Euclid_lenses[:, 1]

# convert into comoving distances (in Mpc)
chid = background.comoving_radial_distance(zd)
chis = background.comoving_radial_distance(zs)

######################################### Euclid galaxy distribution ##########################################################

#first, we define the parameters used to create the bins
binparams = {
           'redshifts': [0,0.4676,0.7194,0.9625,1.3319,3],
           'A': [3.0005,3.0003,3.0008,3.0014,3.0019],
           'alpha': 0.4710,
           'beta': 5.11843,
           'gamma': 0.7259
}

def nb(z, b):
    """
    This is the redshift distribution of galaxies expected from the Euclid survey for each redshift bin.
    Source: eq. (10) of https://arxiv.org/pdf/2010.07376.pdf and caption of fig. 3.

    z  : the redshift
    b  : the redshift bin
    """

    if b in [1,2,3,4,5]:

        A = binparams['A'][b-1]
        alpha = binparams['alpha']
        beta = binparams['beta']
        gamma = binparams['gamma']
        zzmin = binparams['redshifts'][b-1] 
        zzmax = binparams['redshifts'][b]
        
        if zzmin <= z < zzmax: #if the redshift is in the permitted redshift bin
            distrib = A * (z**alpha + z**(alpha*beta)) / (z**beta + gamma)
        else: 
            distrib = 0
            
        return distrib*np.heaviside(distrib,0)
    
    else:
        raise ValueError('Only bins 1 to 5 are supported.')

def normalisation(b):
    """
    This function computes the normalisation constant for the probability distribution of each redshift bin.

    b : the redshift bin
    """
    if b in [1,2,3,4,5]:

        A = binparams['A'][b-1]
        alpha = binparams['alpha']
        beta = binparams['beta']
        gamma = binparams['gamma']
        zzmin = binparams['redshifts'][b-1] 
        zzmax = binparams['redshifts'][b]
    
        def integrand(z):
            n = nb(z,b)
            return n
    
        Int, err = integrate.quad(integrand, zzmin, zzmax)
    
        return 1/Int
    
    else:
        raise ValueError('Only bins 1 to 5 are supported.')

def pb(z, b):

    value = nb(z,b)*normalisation(b)

    val = np.heaviside(value,0)
    
    return value

def find_bin(z):
    """
    This function finds the redshift bin to which a galaxy at redshift z would belong
    """

    found = False
    
    for i in range(len(binparams['A'])):
        if binparams['redshifts'][i] <= z and z < binparams['redshifts'][i+1]:
            bb = i+1
            found = True 

    if not found:  #if the galaxy lies outside any bin
        print('bin not found')
    else:
        return bb

print('Finished 1. preliminaries and computing power spectrum')

##############################################################################################################################
########################################### 2. LOS AUTOCORRELATION FUNCTIONS #################################################
##############################################################################################################################

################################################ LOS Weight function #####################################################

# Interpolate to get a fast 1D weight function
W_LOS_mean_vec = np.vectorize(W_LOS_mean)
chi = np.linspace(0, max(chis), 100)
W = W_LOS_mean_vec(chi)
W_LOS_mean_intp = CubicSpline(chi, W)


# Interpolate to get a fast 1D weight function
WW_LOS_mean_vec = np.vectorize(WW_LOS_mean)
chi = np.linspace(0, max(chis), 100)
WW = WW_LOS_mean_vec(chi)
WW_rms = np.sqrt(WW)
WW_LOS_rms_intp = CubicSpline(chi, WW_rms)

lmax = 1e8
nl = 1000
chimax = max(chis)
ls, cl2, cl1, cl32 = get_cls_gamma_LOS(chimax, lmax, nl)
cl2_LOS_intp = CubicSpline(ls, cl2)
cl1_LOS_intp = CubicSpline(ls, cl1)
cl32_LOS_intp = CubicSpline(ls, cl32)

Theta, xi2_LOS_plus, xi2_LOS_minus = get_correlations(
    cl2_LOS_intp, Thetamin, Thetamax, nTheta=nTheta)
Theta, xi1_LOS_plus, xi1_LOS_minus = get_correlations(
    cl1_LOS_intp, Thetamin, Thetamax, nTheta=nTheta)
Theta, xi32_LOS_plus, xi32_LOS_minus = get_correlations(
    cl32_LOS_intp, Thetamin, Thetamax, nTheta=nTheta)

Theta_arcmin = radtoarcmin(Theta)

xi2_LOS_plus_intp = CubicSpline(Theta, xi2_LOS_plus)
xi1_LOS_plus_intp = CubicSpline(Theta, xi1_LOS_plus)
xi32_LOS_plus_intp = CubicSpline(Theta, xi32_LOS_plus)
xi2_LOS_minus_intp = CubicSpline(Theta, xi2_LOS_minus)
xi1_LOS_minus_intp = CubicSpline(Theta, xi1_LOS_minus)
xi32_LOS_minus_intp = CubicSpline(Theta, xi32_LOS_minus)

print('Finished 2. LOS autocorrelation functions')

##############################################################################################################################
########################################## 3. SHAPE AUTOCORRELATION FUNCTIONS ################################################
##############################################################################################################################

# Interpolate to get fast 1D weight functions

W_os_mean_intp = []

for d in [1,2,3,4,5]:
    W_os_mean_vec = np.vectorize(W_os_mean)
    chi = np.linspace(1e-3, max(chis), 100)
    W = W_os_mean_vec(chi, d)
    W_os_mean_intp.append(CubicSpline(chi, W))

# Interpolate to get fast 1D weight functions

WW_os_rms_intp = []

for d in [1,2,3,4,5]:
    WW_os_mean_vec = np.vectorize(WW_os_mean)
    chi = np.linspace(1e-5, max(chis), 100)
    WW = WW_os_mean_vec(chi, d)
    WW_rms = np.sqrt(WW)
    WW_os_rms_intp.append(CubicSpline(chi, WW_rms))


ls_list = []
cl2_eps_intp = []
cl1_eps_intp = []

lmax = 1e8
nl = 1000
chimax = max(chis)

for d1 in [1,2,3,4,5]: #loop through b

    cl2_eps_intp.append([])

    for d2 in [1,2,3,4,5]: #loop through b'
        
        ls, cl2, cl1 = get_cl_gamma(d1, d2, chimax, lmax, nl) #each time we recalculate everything (a bit inefficient)
        
        cl2_eps_intp[d1-1].append(CubicSpline(ls, cl2))

    ls_list.append(ls)
    cl1_eps_intp.append(CubicSpline(ls, cl1))

Theta_list = []
xi2_eps_plus_list = []
xi2_eps_minus_list = []
xi1_eps_plus_list = []
xi1_eps_minus_list = []

for d1 in [1,2,3,4,5]:
        
    xi2_eps_plus_list.append([])
    xi2_eps_minus_list.append([])
    
    Theta, xi1_eps_plus, xi1_eps_minus = get_correlations(
        cl1_eps_intp[d1-1], Thetamin, Thetamax, nTheta)

    for d2 in [1,2,3,4,5]:
        Theta, xi2_eps_plus, xi2_eps_minus = get_correlations(
            cl2_eps_intp[d1-1][d2-1], Thetamin, Thetamax, nTheta)
        
        xi2_eps_plus_list[d1-1].append(xi2_eps_plus)
        xi2_eps_minus_list[d1-1].append(xi2_eps_minus)
    
    Theta_list.append(Theta)
    xi1_eps_plus_list.append(xi1_eps_plus)
    xi1_eps_minus_list.append(xi1_eps_minus)

xi2_eps_plus_intp = []
xi1_eps_plus_intp = []
xi2_eps_minus_intp = []
xi1_eps_minus_intp = []

for d1 in [1,2,3,4,5]:
    
    xi1_eps_plus_intp.append(CubicSpline(Theta_list[d1-1], xi1_eps_plus_list[d1-1]))
    xi1_eps_minus_intp.append(CubicSpline(Theta_list[d1-1], xi1_eps_minus_list[d1-1]))
    xi2_eps_plus_intp.append([])
    xi2_eps_minus_intp.append([])

    for d2 in [1,2,3,4,5]:
        xi2_eps_plus_intp[d1-1].append(CubicSpline(Theta_list[d1-1], xi2_eps_plus_list[d1-1][d2-1]))
        xi2_eps_minus_intp[d1-1].append(CubicSpline(Theta_list[d1-1], xi2_eps_minus_list[d1-1][d2-1]))
        
print('Finished 3. shape autocorrelation functions')

##############################################################################################################################
######################################## 4. POSITION AUTOCORRELATION FUNCTIONS ###############################################
##############################################################################################################################

##############################################################################################################################
############################################ 5. MIXED CORRELATION FUNCTIONS ##################################################
##############################################################################################################################

lmax = 1e8
nl = 1000
chimax = max(chis)

ls_list = []
cl2LOSos_intp_list = []
cl32LOSos2_intp_list = []
cl32LOS2os_intp_list = []
cl32LOS2os2_intp_list = []

for d1 in [1,2,3,4,5]:
    
    cl32LOSos2_intp_list.append([])
    cl32LOS2os2_intp_list.append([])

    for d2 in [1,2,3,4,5]:
    
        ls, cl2LOSos, cl32LOSos2, cl32LOS2os, cl32LOS2os2 = get_cls_mixed(d1, d2, chimax, lmax, nl)
        
        cl32LOSos2_intp_list[d1-1].append(CubicSpline(ls, cl32LOSos2))
        cl32LOS2os2_intp_list[d1-1].append(CubicSpline(ls, cl32LOS2os2))
        
    ls_list.append(ls)
    cl2LOSos_intp_list.append(CubicSpline(ls, cl2LOSos))
    cl32LOS2os_intp_list.append(CubicSpline(ls, cl32LOS2os))

Theta_list = []

xi2_LOS_eps_plus_list = []
xi2_LOS_eps_minus_list = []

xi32_LOS_eps2_plus_list = []
xi32_LOS_eps2_minus_list = []

xi32_LOS2_eps_plus_list = []
xi32_LOS2_eps_minus_list = []

xi32_LOS2_eps2_plus_list = []
xi32_LOS2_eps2_minus_list = []

for d1 in [1,2,3,4,5]:
    
    Theta, xi2_LOS_eps_plus, xi2_LOS_eps_minus = get_correlations(
        cl2LOSos_intp_list[d1-1], Thetamin, Thetamax, nTheta)
    
    Theta, xi32_LOS2_eps_plus, xi32_LOS2_eps_minus = get_correlations(
        cl32LOS2os_intp_list[d1-1], Thetamin, Thetamax, nTheta)

    xi32_LOS_eps2_plus_list.append([])
    xi32_LOS_eps2_minus_list.append([])

    xi32_LOS2_eps2_plus_list.append([])
    xi32_LOS2_eps2_minus_list.append([])
    
    for d2 in [1,2,3,4,5]:
    
        Theta, xi32_LOS_eps2_plus, xi32_LOS_eps2_minus = get_correlations(
            cl32LOSos2_intp_list[d1-1][d2-1], Thetamin, Thetamax, nTheta)
    
        xi32_LOS_eps2_plus_list[d1-1].append(xi32_LOS_eps2_plus)
        xi32_LOS_eps2_minus_list[d1-1].append(xi32_LOS_eps2_minus)
    
        Theta, xi32_LOS2_eps2_plus, xi32_LOS2_eps2_minus = get_correlations(
            cl32LOS2os2_intp_list[d1-1][d2-1], Thetamin, Thetamax, nTheta)
    
        xi32_LOS2_eps2_plus_list[d1-1].append(xi32_LOS2_eps2_plus)
        xi32_LOS2_eps2_minus_list[d1-1].append(xi32_LOS2_eps2_minus)

    Theta_list.append(Theta)
    
    xi2_LOS_eps_plus_list.append(xi2_LOS_eps_plus)
    xi2_LOS_eps_minus_list.append(xi2_LOS_eps_minus)
    
    xi32_LOS2_eps_plus_list.append(xi32_LOS2_eps_plus)
    xi32_LOS2_eps_minus_list.append(xi32_LOS2_eps_minus)

xi2_LOS_eps_plus_intp = []
xi2_LOS_eps_minus_intp = []

xi32_LOS_eps2_plus_intp = []
xi32_LOS_eps2_minus_intp = []

xi32_LOS2_eps_plus_intp = []
xi32_LOS2_eps_minus_intp = []

xi32_LOS2_eps2_plus_intp = []
xi32_LOS2_eps2_minus_intp = []

for d1 in [1,2,3,4,5]:
    
    xi2_LOS_eps_plus_intp.append(CubicSpline(Theta_list[d1-1], xi2_LOS_eps_plus_list[d1-1]))
    xi2_LOS_eps_minus_intp.append(CubicSpline(Theta_list[d1-1], xi2_LOS_eps_minus_list[d1-1]))
    
    xi32_LOS2_eps_plus_intp.append(CubicSpline(Theta_list[d1-1], xi32_LOS2_eps_plus_list[d1-1]))
    xi32_LOS2_eps_minus_intp.append(CubicSpline(Theta_list[d1-1], xi32_LOS2_eps_minus_list[d1-1]))
    
    xi32_LOS_eps2_plus_intp.append([])
    xi32_LOS_eps2_minus_intp.append([])

    for d2 in [1,2,3,4,5]:
        
        xi32_LOS_eps2_plus_intp[d1-1].append(CubicSpline(Theta_list[d1-1], xi32_LOS_eps2_plus_list[d1-1][d2-1]))
        xi32_LOS_eps2_minus_intp[d1-1].append(CubicSpline(Theta_list[d1-1], xi32_LOS_eps2_minus_list[d1-1][d2-1]))
        
        xi32_LOS2_eps2_plus_intp.append(CubicSpline(Theta_list[d1-1], xi32_LOS2_eps2_plus_list[d1-1][d2-1]))
        xi32_LOS2_eps2_minus_intp.append(CubicSpline(Theta_list[d1-1], xi32_LOS2_eps2_minus_list[d1-1][d2-1]))

print('Finished 4. mixed correlation functions')

################################################ GENERATING COVARIANCE MATRICES ##############################################################

def process_pair(args):
    b1, distributions, sigma_noise, sigma_shape = args

    # Generate ncov
    ncov_LOSeps, ncov_LOSeps_pp, ncov_LOSeps_mm, ncov_LOSeps_pm, ncov_LOSeps_mp = generate_ncov(distributions, sigma_noise, sigma_shape, b1)
    NCOV = {'full': ncov_LOSeps,
            'pp': ncov_LOSeps_pp,
            'mm': ncov_LOSeps_mm,
            'pm': ncov_LOSeps_pm,
            'mp': ncov_LOSeps_mp}

    print('generated noise covariance (b1 = ' +str(b1) + ')')
    
    # Generate ccov
    ccov_LOSeps, ccov_LOSeps_pp, ccov_LOSeps_mm, ccov_LOSeps_pm, ccov_LOSeps_mp = generate_ccov(distributions, b1)
    CCOV = {'full': ccov_LOSeps,
            'pp': ccov_LOSeps_pp,
            'mm': ccov_LOSeps_mm,
            'pm': ccov_LOSeps_pm,
            'mp': ccov_LOSeps_mp}

    print('generated cosmic covariance (b1 = ' +str(b1) + ')')
    
    # Generate scov
    scov_LOSeps, scov_LOSeps_pp, scov_LOSeps_mm, scov_LOSeps_pm, scov_LOSeps_mp = generate_scov(distributions, b1)
    SCOV = {'full': scov_LOSeps,
            'pp': scov_LOSeps_pp,
            'mm': scov_LOSeps_mm,
            'pm': scov_LOSeps_pm,
            'mp': scov_LOSeps_mp}

    print('generated sparsity covariance (b1 = ' +str(b1) + ')')
    
    # Save NCOV
    try:
        with open(f"Matrices/LLLe/ncov{b1}", "wb") as f:
            pickle.dump(NCOV, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as ex:
        print(f"Error during pickling NCOV for b1={b1}: {ex}")

    # Save CCOV
    try:
        with open(f"Matrices/LLLe/ccov{b1}", "wb") as f:
            pickle.dump(CCOV, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as ex:
        print(f"Error during pickling CCOV for b1={b1}: {ex}")
    
    # Save SCOV
    try:
        with open(f"Matrices/LLLe/scov{b1}", "wb") as f:
            pickle.dump(SCOV, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as ex:
        print(f"Error during pickling SCOV for b1={b1}: {ex}")

def main():
    # Create output directory if it doesn't exist
    os.makedirs("Matrices/LeLe", exist_ok=True)
    
    # Define parameter ranges and constants
    b1_values = [1, 2, 3, 4, 5]

    # Prepare arguments for parallel processing
    args_list = [(b1, distributions, sigma_noise, sigma_shape) for b1 in b1_values]

    # Run in parallel using multiprocessing
    with Pool() as pool:
        pool.map(process_pair, args_list)

if __name__ == "__main__":
    main()

