import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from config import *

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from redshift_distributions import *
from functions.angular_distributions import * 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from useful_functions import *

get_item('LLp', 'LLx', 'L0')

################################################## LLLL cosmic covariance ##############################################################

def generate_ccov_LLLL(angular_distribution):
    """
    Computes the contribution of cosmic variance in the covariance matrix
    of the LOS shear correlation functions.
    """
  
    Nbin       = angular_distribution.Nbina      #the number of angular bins for LL 
    Omegas     = angular_distribution.Omegas     #\Omega_a in the math - the solid angle of bin a (in rad^2)
    rs         = angular_distribution.limits     #the angular bin limits for LL (in rad)
    
    # Define the integrands
    
    def integrand_pp(params):
        
        psi_j, psi_kl, r_j, r_kl, r_k = params
    
        x_jl = r_kl * np.cos(psi_kl) - r_j * np.cos(psi_j) + r_k
        y_jl = r_kl * np.sin(psi_kl) - r_j * np.sin(psi_j)
        
        r_jl = np.sqrt( x_jl**2 + y_jl**2 ) 
        psi_jl = np.arctan2(y_jl, x_jl)

        f = ( ( LLp(r_k) * cos2(psi_j) * cos2(psi_kl)
              + LLx(r_k) * sin2(psi_j) * sin2(psi_kl) )
        * ( LLp(r_jl) * cos2(psi_jl - psi_j) * cos2(psi_jl - psi_kl)
              + LLx(r_jl) * sin2(psi_jl - psi_j) * sin2(psi_jl - psi_kl) ) )

        f *= 2 * np.pi * r_k * r_j * r_kl
        
        return f
    
    def integrand_px(params):
        
        psi_j, psi_kl, r_j, r_kl, r_k = params
    
        x_jl = r_kl * np.cos(psi_kl) - r_j * np.cos(psi_j) + r_k
        y_jl = r_kl * np.sin(psi_kl) - r_j * np.sin(psi_j)
        
        r_jl = np.sqrt( x_jl**2 + y_jl**2 ) 
        psi_jl = np.arctan2(y_jl, x_jl)

        f = - ( ( LLp(r_k) * cos2(psi_j) * sin2(psi_kl)
              - LLx(r_k) * sin2(psi_j) * cos2(psi_kl) ) * ( LLp(r_jl) * cos2(psi_jl - psi_j) * sin2(psi_jl - psi_kl)
              - LLx(r_jl) * sin2(psi_jl - psi_j) * cos2(psi_jl - psi_kl) ) )

        f *= 2 * np.pi * r_k * r_j * r_kl
        
        return f
    
    def integrand_xp(params):
        
        psi_j, psi_kl, r_j, r_kl, r_k = params
    
        x_jl = r_kl * np.cos(psi_kl) - r_j * np.cos(psi_j) + r_k
        y_jl = r_kl * np.sin(psi_kl) - r_j * np.sin(psi_j)
        
        r_jl = np.sqrt( x_jl**2 + y_jl**2 ) 
        psi_jl = np.arctan2(y_jl, x_jl)

        f = - ( ( LLp(r_k) * sin2(psi_j) * cos2(psi_kl)
              - LLx(r_k) * cos2(psi_j) * sin2(psi_kl) ) * ( LLp(r_jl) * sin2(psi_jl - psi_j) * cos2(psi_jl - psi_kl)
              - LLx(r_jl) * cos2(psi_jl - psi_j) * sin2(psi_jl - psi_kl) ) )

        f *= 2 * np.pi * r_k * r_j * r_kl
        
        return f
    
    def integrand_xx(params):
        
        psi_j, psi_kl, r_j, r_kl, r_k = params
    
        x_jl = r_kl * np.cos(psi_kl) - r_j * np.cos(psi_j) + r_k
        y_jl = r_kl * np.sin(psi_kl) - r_j * np.sin(psi_j)
        
        r_jl = np.sqrt( x_jl**2 + y_jl**2 ) 
        psi_jl = np.arctan2(y_jl, x_jl)

        f = ( ( LLp(r_k) * sin2(psi_j) * sin2(psi_kl)
              + LLx(r_k) * cos2(psi_j) * cos2(psi_kl) ) * ( LLp(r_jl) * sin2(psi_jl - psi_j) * sin2(psi_jl - psi_kl)
              + LLx(r_jl) * cos2(psi_jl - psi_j) * cos2(psi_jl - psi_kl) ) )

        f *= 2 * np.pi * r_k * r_j * r_kl
        
        return f
    
    def integral_bins(integrand, alpha, beta):
        
        ranges = [(0, 2*np.pi), (0, 2*np.pi),
                  (rs[alpha], rs[alpha+1]), (rs[beta], rs[beta+1]), (0, r2_max)]
        
        integral, err = monte_carlo_integrate(integrand, ranges, Csamp)
        
        # normalisation of differential elements
        integral *= 2/(Omegatot * Omegas[alpha] * Omegas[beta]) 
        err *= 2/(Omegatot * Omegas[alpha] * Omegas[beta])
        
        return integral, err
                     
    ccov_pp, err_pp = integral_bins(integrand_pp, 0, 0)
    ccov_px, err_px = integral_bins(integrand_px, 0, 0)
    ccov_xp, err_xp = integral_bins(integrand_xp, 0, 0)
    ccov_xx, err_xx = integral_bins(integrand_xx, 0, 0)

    err = np.sqrt(err_pp**2 + err_px**2 + err_xp**2 + err_xx**2)

    ccovpp = ccov_pp + ccov_px + ccov_xp + ccov_xx
    ccovmm = ccov_pp - ccov_px - ccov_xp + ccov_xx
    
    return ccovpp, ccovmm, err

def LLLL_ccov_v_theta(theta):
    
    angular_distribution = Angular_Distributions(binscheme=[0,theta], Nbin_a=1, Thetamax=None)
    ccovpp, ccovmm, err = generate_ccov_LLLL(angular_distribution)

    return ccovpp, ccovmm

################################################## 6.2.2 LeLe noise/sparsity covariance #############################################################

def LLLL_ncov_v_theta(theta):
    
    angular_distribution = Angular_Distributions(binscheme=[0,theta], Nbin_a=1, Thetamax=None)
        
    Nbin       = angular_distribution.Nbina      #the number of angular bins for LL
    Omegas     = angular_distribution.Omegas     #\Omega_a in the math - the solid angle of bin a (in rad^2)
    rs         = angular_distribution.limits     #the angular bin limits for LL (in rad)

    def integrand_pp(params):
        
        r_j, r_l, psi_l = params
    
        y_jl = r_l*np.sin(psi_l)
        x_jl = r_l*np.cos(psi_l) - r_j
        
        r_jl = np.sqrt( y_jl**2 + x_jl**2 ) 
        psi_jl = np.arctan2(y_jl, x_jl)
        
        f = cos2(psi_l) * ( LLp(r_jl) * cos2(psi_jl) * cos2(psi_jl - psi_l) + LLx(r_jl) * sin2(psi_jl) * sin2(psi_jl-psi_l) )

        f *= 2 * np.pi * r_j * r_l
                              
        return f
    
    def integrand_px(params):
        
        r_j, r_l, psi_l = params
    
        y_jl = r_l*np.sin(psi_l)
        x_jl = r_l*np.cos(psi_l) - r_j
        
        r_jl = np.sqrt( y_jl**2 + x_jl**2 ) 
        psi_jl = np.arctan2(y_jl, x_jl)
        
        f = - sin2(psi_l) * ( LLp(r_jl) * cos2(psi_jl) * sin2(psi_jl - psi_l) - LLx(r_jl) * sin2(psi_jl) * cos2(psi_jl-psi_l) )

        f *= 2 * np.pi * r_j * r_l
                              
        return f
    
    def integrand_xp(params):
        
        r_j, r_l, psi_l = params
    
        y_jl = r_l*np.sin(psi_l)
        x_jl = r_l*np.cos(psi_l) - r_j
        
        r_jl = np.sqrt( y_jl**2 + x_jl**2 ) 
        psi_jl = np.arctan2(y_jl, x_jl)
        
        f = sin2(psi_l) * ( LLp(r_jl) * sin2(psi_jl) * cos2(psi_jl - psi_l) - LLx(r_jl) * cos2(psi_jl) * sin2(psi_jl-psi_l) )

        f *= 2 * np.pi * r_j * r_l
                              
        return f
    
    def integrand_xx(params):
        
        r_j, r_l, psi_l = params
    
        y_jl = r_l*np.sin(psi_l)
        x_jl = r_l*np.cos(psi_l) - r_j
        
        r_jl = np.sqrt( y_jl**2 + x_jl**2 ) 
        psi_jl = np.arctan2(y_jl, x_jl)
        
        f = cos2(psi_l) * ( LLp(r_jl) * sin2(psi_jl) * sin2(psi_jl - psi_l) + LLx(r_jl) * cos2(psi_jl) * cos2(psi_jl-psi_l) )

        f *= 2 * np.pi * r_j * r_l
                              
        return f
    
    def integral_bins(integrand, alpha, beta):
        
        ranges = [(rs[alpha], rs[alpha+1]), (rs[beta], rs[beta+1]), (0, 2*np.pi)]
        
        integral, err = monte_carlo_integrate(integrand, ranges, Nsamp)
        
        # normalisation of differential elements
        integral *= 2/(Omegas[alpha] * Omegas[beta]) 
        err      *= 2/(Omegas[alpha] * Omegas[beta]) 
        
        return integral, err
                     
    int_pp, err_pp = integral_bins(integrand_pp, 0, 0)
    int_px, err_px = integral_bins(integrand_px, 0, 0)
    int_xp, err_xp = integral_bins(integrand_xp, 0, 0)
    int_xx, err_xx = integral_bins(integrand_xx, 0, 0)

    return int_pp, int_px, int_xp, int_xx

def generate_ncov_LLLL(sigma_L, Nlens, angular_distribution):
    """
    Computes the contribution of noise and sparsity variance in the 
    covariance matrix of the LOS shear correlation functions.
    """
        
    Nbin       = angular_distribution.Nbina      #the number of angular bins for LL
    Omegas     = angular_distribution.Omegas     #\Omega_a in the math - the solid angle of bin a (in rad^2)
    rs         = angular_distribution.limits     #the angular bin limits for LL (in rad)

    get_item('LLLL_int_pp','LLLL_int_px','LLLL_int_xp','LLLL_int_xx')
                         
    ncov_pp = (sigma_L**2/Nlens) * LLLL_int_pp(rs[1])
    # nerr_pp[alpha, beta] = (sigma_L**2/Nlens) * err_pp
             
    scov_pp = (L0/Nlens) * LLLL_int_pp(rs[1])
    # serr_pp = (L0/Nlens) * err_pp
             
    ncov_px = (sigma_L**2/Nlens) * LLLL_int_px(rs[1])
    # nerr_px = (sigma_L**2/Nlens) * err_px
             
    scov_px = (L0/Nlens) * LLLL_int_px(rs[1])
    # serr_px = (L0/Nlens) * err_px
             
    ncov_xp = (sigma_L**2/Nlens) * LLLL_int_xp(rs[1])
    # nerr_xp = (sigma_L**2/Nlens) * err_xp
             
    scov_xp = (L0/Nlens) * LLLL_int_xp(rs[1])
    # serr_xp = (L0/Nlens) * err_xp
             
    ncov_xx = (sigma_L**2/Nlens) * LLLL_int_xx(rs[1])
    # nerr_xx = (sigma_L**2/Nlens) * err_xx
             
    scov_xx = (L0/Nlens) * LLLL_int_xx(rs[1])
    # serr_xx = (L0/Nlens) * err_xx
                
    cterm_n = (1/2)* ( (sigma_L**4+2*L0*sigma_L**2)/(Nlens**2) ) * Omegatot / Omegas[0]
    cterm_s = (1/2) * ( (L0/Nlens)**2 ) * Omegatot / Omegas[0]
    
    ncov_pp += cterm_n
    ncov_xx += cterm_n
    
    scov_pp += cterm_s
    scov_xx += cterm_s

    # nerr = np.sqrt(nerr_pp**2 + nerr_px**2 + nerr_xp**2 + nerr_xx**2)
    # serr = np.sqrt(serr_pp**2 + serr_px**2 + serr_xp**2 + serr_xx**2)

    ncovpp = ncov_pp + ncov_px + ncov_xp + ncov_xx    
    scovpp = scov_pp + scov_px + scov_xp + scov_xx

    ncovmm = ncov_pp - ncov_px - ncov_xp + ncov_xx
    scovmm = scov_pp - scov_px - scov_xp + scov_xx

    ncov = [ncovpp, ncovmm]
    scov = [scovpp, scovmm]
    
    return [ncov, scov]#, [nerr, serr]