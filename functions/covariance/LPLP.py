import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from config import *

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from redshift_distributions import *
from functions.angular_distributions import * 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from useful_functions import *



get_item('LLp','LLx','LP', 'PP', 'redshift_distributions', 'L0')

################################################## LPLP cosmic covariance ##############################################################

def generate_ccov_LPLP(angular_distribution):
    """
    Computes the contribution of cosmic variance in the covariance matrix
    of the LOS shear - galaxy position correlation functions.
    """
    
    Nbin       = angular_distribution.Nbina      #the number of angular bins for LP (redshift bin B)
    Omegas     = angular_distribution.Omegas     #\Omega_a in the math - the solid angle of bin a (in rad^2) (redshift bin B)
    rs         = angular_distribution.limits     #the angular bin limits for LP (in rad) (redshift bin B)

    B = 0 
    D = B #same redshift bin
    
    # Define the integrands
    
    def integrand(params):
        
        psi_b, psi_kd, r_b, r_kd, r_k = params
        
        x_bd = r_kd * np.cos(psi_kd) - r_b * np.cos(psi_b) + r_k
        y_bd = r_kd * np.sin(psi_kd) - r_b * np.sin(psi_b)
        
        r_bd = np.sqrt( x_bd**2 + y_bd**2 ) 
        psi_bd = np.arctan2(y_bd, x_bd)
        
        f = ( (LLp(r_k) * cos2(psi_b) * cos2(psi_kd)
            + LLx(r_k) * sin2(psi_b) * sin2(psi_kd))
            * PP[B][D](r_bd)
            + LP[D](r_bd) * LP[B](r_k) * cos2(psi_bd-psi_b) * cos2(psi_kd)
            )
        
        f *= 2 * np.pi * r_k * r_b * r_kd
        
        return f
    
    def integral_bins(integrand, alpha, beta):
        
        ranges = [(0, 2*np.pi), (0, 2*np.pi),
                  (rs[alpha], rs[alpha+1]), (rs[beta], rs[beta+1]), (0, r2_max)]
        
        integral, err = monte_carlo_integrate(integrand, ranges, Csamp)
        
        # normalisation of differential elements
        integral /= (Omegatot * Omegas[alpha] * Omegas[beta]) 
        err /= (Omegatot * Omegas[alpha] * Omegas[beta]) 
        return integral, err
    
    ccov, err = integral_bins(integrand, 0, 0)
    
    return ccov, err

def LPLP_ccov_v_theta(theta):
    
    angular_distribution = Angular_Distributions(binscheme=[0,theta], Nbin_a=1, Thetamax=None)
    ccov, err = generate_ccov_LPLP(angular_distribution)

    return ccov

################################################## LPLP noise/sparsity covariance #############################################################

def LPLP_ncov_v_theta(theta):
    """
    Computes the contribution of noise and sparsity variance in the 
    covariance matrix of the LOS shear - galaxy position correlation functions.
    """
    
    angular_distribution = Angular_Distributions(binscheme=[0,theta], Nbin_a=1, Thetamax=None)
    
    B = 0
    D = B #same redshift bin
    
    Nbin       = angular_distribution.Nbina      #the number of angular bins
    Omegas     = angular_distribution.Omegas     #\Omega_a in the math - the solid angle of bin a (in rad^2)
    rs         = angular_distribution.limits     #the angular bin limits (in rad)
    
    redshift_distribution = redshift_distributions['P']

    G_B    = redshift_distribution.get_ngal(B)         #G_B in the math - the number of galaxies in redshift bin B
    
    # Define the integrands
    
    def integrand_P(params):
        
        r_i, r_k, psi_k = params
    
        y_ik = r_k*np.sin(psi_k)
        x_ik = r_k*np.cos(psi_k) - r_i
        
        r_ik = np.sqrt( y_ik**2 + x_ik**2 ) 
        psi_ik = np.arctan2(y_ik, x_ik)
        
        f = ( LLp(r_ik) * cos2(psi_ik) * cos2(psi_ik - psi_k) 
            + LLx(r_ik) * sin2(psi_ik) * sin2(psi_ik - psi_k) )

        f *= 2 * np.pi * r_i * r_k
                              
        return f
    
    def integrand_L(params):
        
        r_i, r_k, psi_k = params
    
        y_ik = r_k*np.sin(psi_k)
        x_ik = r_k*np.cos(psi_k) - r_i
        
        r_ik = np.sqrt( y_ik**2 + x_ik**2 ) 
        psi_ik = np.arctan2(y_ik, x_ik)
        
        f = (1/2) * PP[B][D](r_ik) * cos2(psi_k)

        f *= 2 * np.pi * r_i * r_k
                              
        return f
    
    def integral_bins(integrand, alpha, beta):
        
        ranges = [(rs[alpha], rs[alpha+1]), (rs[beta], rs[beta+1]), (0, 2*np.pi)]
        
        integral, err = monte_carlo_integrate(integrand, ranges, Nsamp)
        
        # normalisation of differential elements
        integral /= (Omegas[alpha] * Omegas[beta]) 
        err     /= (Omegas[alpha] * Omegas[beta]) 
        
        return integral, err

    integrand = [integrand_L, integrand_P]
         
    intt, err = integral_bins(integrand, 0, 0)
             
    return intt

def generate_ncov_LPLP(sigma_L, Nlens, angular_distribution):
    """
    Computes the contribution of noise and sparsity variance in the 
    covariance matrix of the LOS shear - galaxy position correlation functions.
    """

    get_item('LPLP_int')

    B = 0
    
    Nbin       = angular_distribution.Nbina      #the number of angular bins
    Omegas     = angular_distribution.Omegas     #\Omega_a in the math - the solid angle of bin a (in rad^2)
    rs         = angular_distribution.limits     #the angular bin limits (in rad)
    
    redshift_distribution = redshift_distributions['P']

    G_B    = redshift_distribution.get_ngal(B)         #G_B in the math - the number of galaxies in redshift bin B
    
    ncov = (sigma_L**2/Nlens) * LPLP_int[0](rs[1])     
    # nerr = (sigma_L**2/Nlens) * err[0]
             
    scov = (L0/Nlens) * LPLP_int[0](rs[1])     
    # serr = (L0/Nlens) * err[0]

    ncov += (1/G_B) * LPLP_int[1](rs[1])  
    scov += (1/G_B) * LPLP_int[1](rs[1])  

    ncov += (1/2) * (sigma_L**2 / (Nlens*G_B) ) * (Omegatot/Omegas[0])
    scov += (1/2) * (L0 / (Nlens*G_B) ) * (Omegatot/Omegas[0])
    
    return [ncov, scov]#, [nerr, serr]