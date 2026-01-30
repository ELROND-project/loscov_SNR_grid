import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from config import *

from functions.useful_functions import *

def optimise_bins(correlation_function, correlation_type, antiderivative, b, SNR_goal = 20, Nbin_max = 10, SNR_min = 1.5, Thetamax_distribution = Thetamax_dist):

    Thetamax_distribution = arcmintorad(Thetamax_distribution)
    
    redshift_distributions = load_file(f"data/redshift_distributions")
    
    def snr(theta_1, theta_2):

        numerator = np.abs(antiderivative(theta_2) - antiderivative(theta_1))
        denom_sq = theta_2**2 - theta_1**2
        
        if denom_sq <= 0:
            return 0

        denominator = np.sqrt(denom_sq)
        
        return numerator / denominator

    def SNR(theta):
        return snr(0,theta)
        
    theta_optimal, SNR_max = find_maximum(SNR, 0, Thetamax_distribution)

    return [0, theta_optimal]         #return a single bin

class Angular_Distributions:
    """
    This class produces anything useful related to the statistics of lenses and galaxies,
    and their binning in angular separation
    """
    
    def __init__(self, binscheme=None, sky_coverage=sky_coverage, Nbin_a=None, Thetamax=Thetamax_dist):
        """
        Arguments:
        - Nlens         : number of lenses we can use
        - Ngal          : number of galaxies we can use
        - sky_coverage  : area of the survey footprint, in deg2 
        - Nbin_a         : number of bins of angular separation
        - b1            : the first angular separation bin
        - b2            : the second angular separation bin
        - Thetamax      : maximum angular separation, in arcmin
        - binscheme     : list, min and max angular separations of each angular separation bin
        
        All the angular attributes will be expressed in rad.
        """
        
        # Lens number and density
        self.Omegatot = sky_coverage * (np.pi / 180)**2 # in rad2
        self.number = Nobjects
        self.density = Nobjects / self.Omegatot #the density of lenses or galaxies in an angular bin

        # Binning
        if isinstance(binscheme, int): 
            
            Omega_arcmin2 = np.pi * Thetamax**2 / Nbin_a  #the angular size of each bin
            self.Nbina = Nbin_a
            self.Omega = Omega_arcmin2 * (np.pi / 180 / 60)**2  # converting the angular size of a bin to rad2
            self.Omegas = self.Omega * np.ones(Nbin_a) # in rad2 
            
            # the list of Omegas is in case we want to have different bin sizes
            self.limits = np.sqrt(self.Omega / np.pi * np.arange(Nbin_a + 1)) # in rad        
            
            # "Centres" of the bins, taken as the mean separation of objects within the angular bin
            self.Thetas = 2/3 * (self.limits[1:]**3  - self.limits[:-1]**3 ) / (self.limits[1:]**2  - self.limits[:-1]**2 ) 
            
        elif isinstance(binscheme, list):
                
            Omegas = []  #the angular size of each bin, in rad^2
            limits = []
            limits.append(binscheme[0])    #the minimum angular separation considered
                
            Nbin_a = len(binscheme) - 1
            self.Nbina = Nbin_a 
                
            for i in range(Nbin_a):
                omega = np.pi * (binscheme[i+1]**2-binscheme[i]**2) #the angular size of that bin, pi x (R_outer^2 - r_inner^2) 
                Omegas.append(omega)
                limits.append(binscheme[i+1])
                
            self.Omegas = np.array(Omegas)
                
            # the list of Omegas is in case we want to have different bin sizes
            self.limits = np.array(limits)        
                
            # "Centres" of the bins, taken as the median separation
            self.Thetas = np.sqrt((self.limits[1:]**2 + self.limits[:-1]**2) / 2) # in rad
        
        else:
            print("Error: no binscheme defined")        

    
    def compute_binned_correlation(self, correlation):
        """
        This function computes the expected signal for a correlation function with this binning
        Note, this is only relevant for autocorrelation functions.
        """
        
        xi_bins  = []
        
        Nbin   = self.Nbina
        rs     = self.limits
        Omegas = self.Omegas

        def integrand(r):
            f = 2 * np.pi * r * correlation(r)
            return f
        
        for a in range(Nbin):
            print(f'rs={rs}')
            print(f'Nbin={Nbin}')
            print(f'integrand={integrand}')
            integral, err = monte_carlo_integrate(integrand, [(rs[a], rs[a+1])])
            print(f'integral={integral}')
            integral /= Omegas[a]
            xi_bins.append(integral)

        return xi_bins


def generate_binned_correlation(correlation, b1):
    """Handles generation of binned correlation functions based on cov_matrix type."""
    correlation_data = {}

    get_item('LL_plus','LL_minus')
    get_item('LE_plus','LE_minus')
    get_item('LP')
    get_item('angular_distributions')

    correlation_mapping = {
        "LL": lambda: {
            "plus_correlation": angular_distributions['LL_plus'].compute_binned_correlation(LL_plus),
            "minus_correlation": angular_distributions['LL_minus'].compute_binned_correlation(LL_minus),
        },
        "LE": lambda: {
            "plus_correlation": angular_distributions['LE_plus'][b1].compute_binned_correlation(LE_plus[b1]),
            "minus_correlation": angular_distributions['LE_minus'][b1].compute_binned_correlation(LE_minus[b1]),
        },
        "LP": lambda: {
            "correlation": angular_distributions['LP'][b1].compute_binned_correlation(LP[b1]),
        }
    }

    if correlation in correlation_mapping:
        correlation_data = correlation_mapping[correlation]()
    
    return correlation_data
