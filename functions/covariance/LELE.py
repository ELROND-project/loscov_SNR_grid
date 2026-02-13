import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from config import *

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from redshift_distributions import *
from functions.angular_distributions import * 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from useful_functions import *

get_item('LLp','LLx', 'LEp', 'LEx', 'EEp', 'EEx', 'redshift_distributions', 'L0', 'E0')

################################################## LELE cosmic covariance ##############################################################

def generate_ccov_LELE(angular_distribution):
    """
    Computes the contribution of cosmic variance in the covariance matrix
    of the LOS shear - galaxy shape correlation functions.
    """
    
    B = 0
    D = B #only one redshift bin
    
    Nbin       = angular_distribution.Nbina      #the number of angular bins for LE 
    Omegas     = angular_distribution.Omegas     #\Omega_a in the math - the solid angle of bin a (in rad^2)
    rs         = angular_distribution.limits     #the angular bin limits for LE (in rad)

    # Define the integrands
    
    def integrand_pp(params):
        
        psi_b, psi_kd, r_b, r_kd, r_k = params
    
        x_bd = r_kd * np.cos(psi_kd) - r_b * np.cos(psi_b) + r_k
        y_bd = r_kd * np.sin(psi_kd) - r_b * np.sin(psi_b)
        
        r_bd = np.sqrt( x_bd**2 + y_bd**2 ) 
        psi_bd = np.arctan2(y_bd, x_bd)

        f = ( ( LLp(r_k) * cos2(psi_b) * cos2(psi_kd)
              + LLx(r_k) * sin2(psi_b) * sin2(psi_kd) )
        * ( EEp[B][D](r_bd) * cos2(psi_bd - psi_b) * cos2(psi_bd - psi_kd)
           + EEx[B][D](r_bd) * sin2(psi_bd - psi_b) * sin2(psi_bd - psi_kd) )
        + ( LEp[D](r_k) * cos2(psi_b) * cos2(psi_kd)
           + LEx[D](r_k) * sin2(psi_b) * sin2(psi_kd) )
        * ( LEp[B](r_bd) * cos2(psi_bd - psi_b) * cos2(psi_bd - psi_kd)
           + LEx[B](r_bd) * sin2(psi_bd - psi_b) * sin2(psi_bd - psi_kd) )
            )
        
        f *= 2 * np.pi * r_k * r_b * r_kd
        
        return f
    
    def integrand_px(params):
        
        psi_b, psi_kd, r_b, r_kd, r_k = params
    
        x_bd = r_kd * np.cos(psi_kd) - r_b * np.cos(psi_b) + r_k
        y_bd = r_kd * np.sin(psi_kd) - r_b * np.sin(psi_b)
        
        r_bd = np.sqrt( x_bd**2 + y_bd**2 ) 
        psi_bd = np.arctan2(y_bd, x_bd)

        f = - ( ( LLp(r_k) * cos2(psi_b) * sin2(psi_kd)
              - LLx(r_k) * sin2(psi_b) * cos2(psi_kd) )
        * ( EEp[B][D](r_bd) * cos2(psi_bd - psi_b) * sin2(psi_bd - psi_kd)
           - EEx[B][D](r_bd) * sin2(psi_bd - psi_b) * cos2(psi_bd - psi_kd) )
        + ( LEp[D](r_k) * cos2(psi_b) * sin2(psi_kd)
           - LEx[D](r_k) * sin2(psi_b) * cos2(psi_kd) )
            * ( LEp[B](r_bd) * cos2(psi_bd - psi_b) * sin2(psi_bd - psi_kd)
              - LEx[B](r_bd) * sin2(psi_bd - psi_b) * cos2(psi_bd - psi_kd) )
            )
        
        f *= 2 * np.pi * r_k * r_b * r_kd
        
        return f
    
    def integrand_xp(params):
        
        psi_b, psi_kd, r_b, r_kd, r_k = params
    
        x_bd = r_kd * np.cos(psi_kd) - r_b * np.cos(psi_b) + r_k
        y_bd = r_kd * np.sin(psi_kd) - r_b * np.sin(psi_b)
        
        r_bd = np.sqrt( x_bd**2 + y_bd**2 ) 
        psi_bd = np.arctan2(y_bd, x_bd)

        f = - ( ( LLp(r_k) * sin2(psi_b) * cos2(psi_kd)
              - LLx(r_k) * cos2(psi_b) * sin2(psi_kd) )
        * ( EEp[B][D](r_bd) * sin2(psi_bd - psi_b) * cos2(psi_bd - psi_kd)
           - EEx[B][D](r_bd) * cos2(psi_bd - psi_b) * sin2(psi_bd - psi_kd) )
        + ( LEp[D](r_k) * sin2(psi_b) * cos2(psi_kd)
           - LEx[D](r_k) * cos2(psi_b) * sin2(psi_kd) )
            * ( LEp[B](r_bd) * sin2(psi_bd - psi_b) * cos2(psi_bd - psi_kd)
              - LEx[B](r_bd) * cos2(psi_bd - psi_b) * sin2(psi_bd - psi_kd) )
            )
        
        f *= 2 * np.pi * r_k * r_b * r_kd
        
        return f
    
    def integrand_xx(params):
        
        psi_b, psi_kd, r_b, r_kd, r_k = params
    
        x_bd = r_kd * np.cos(psi_kd) - r_b * np.cos(psi_b) + r_k
        y_bd = r_kd * np.sin(psi_kd) - r_b * np.sin(psi_b)
        
        r_bd = np.sqrt( x_bd**2 + y_bd**2 ) 
        psi_bd = np.arctan2(y_bd, x_bd)

        f = ( ( LLp(r_k) * sin2(psi_b) * sin2(psi_kd)
              + LLx(r_k) * cos2(psi_b) * cos2(psi_kd) )
            * ( EEp[B][D](r_bd) * sin2(psi_bd - psi_b) * sin2(psi_bd - psi_kd)
             + EEx[B][D](r_bd) * cos2(psi_bd - psi_b) * cos2(psi_bd - psi_kd) )
            + ( LEp[D](r_k) * sin2(psi_b) * sin2(psi_kd)
              + LEx[D](r_k) * cos2(psi_b) * cos2(psi_kd) )
            * ( LEp[B](r_bd) * sin2(psi_bd - psi_b) * sin2(psi_bd - psi_kd)
              + LEx[B](r_bd) * cos2(psi_bd - psi_b) * cos2(psi_bd - psi_kd) )
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
            
    ccov_pp, err_pp = integral_bins(integrand_pp, 0, 0)
    ccov_px, err_px = integral_bins(integrand_px, 0, 0)
    ccov_xp, err_xp = integral_bins(integrand_xp, 0, 0)
    ccov_xx, err_xx = integral_bins(integrand_xx, 0, 0)

    err = np.sqrt(err_pp**2 + err_px**2 + err_xp**2 + err_xx**2)

    ccovpp = ccov_pp + ccov_px + ccov_xp + ccov_xx
    ccovmm = ccov_pp - ccov_px - ccov_xp + ccov_xx
    
    return ccovpp, ccovmm, err

def LELE_ccov_v_theta(theta):
    
    angular_distribution = Angular_Distributions(binscheme=[0,theta], Nbin_a=1, Thetamax=None)
    ccovpp, ccovmm, err = generate_ccov_LELE(angular_distribution)

    return ccovpp, ccovmm

################################################## LELE noise/sparsity covariance #############################################################

def LELE_ncov_v_theta(theta):
    """
    Computes the contribution of noise and sparsity variance in the covariance matrix
    of the LOS shear - galaxy shape correlation functions.
    """
    
    B = 0
    D = B #only one redshift bin

    angular_distribution = Angular_Distributions(binscheme=[0,theta], Nbin_a=1, Thetamax=None)
    D = B #same redshift bin
    
    Nbin       = angular_distribution.Nbina      #the number of angular bins for LE (sign1) 
    Omegas     = angular_distribution.Omegas     #\Omega_a in the math - the solid angle of bin a (in rad^2) (sign1)
    rs         = angular_distribution.limits     #the angular bin limits for LE (in rad) (sign1)
    
    redshift_distribution = redshift_distributions['E']
    
    G_B    = redshift_distribution.get_ngal(B)         #G_B in the math - the number of galaxies in redshift bin B
    
    # Define the integrands
    
    def integrand_pp_L(params):
        
        r_b, r_d, psi_d = params
    
        y_bd = r_d*np.sin(psi_d)
        x_bd = r_d*np.cos(psi_d) - r_b
        
        r_bd = np.sqrt( y_bd**2 + x_bd**2 ) 
        psi_bd = np.arctan2(y_bd, x_bd)
        
        f = ( cos2(psi_d) 
            * ( EEp[B][D](r_bd) * cos2(psi_bd) * cos2(psi_bd - psi_d) 
              + EEx[B][D](r_bd) * sin2(psi_bd) * sin2(psi_bd - psi_d) 
            ) )

        f *= np.pi * r_b * r_d #the usual 2 disappears because of the factor 1/2 out the front
                              
        return f
    
    def integrand_pp_E(params):
        
        r_b, r_d, psi_d = params
    
        y_bd = r_d*np.sin(psi_d)
        x_bd = r_d*np.cos(psi_d) - r_b
        
        r_bd = np.sqrt( y_bd**2 + x_bd**2 ) 
        psi_bd = np.arctan2(y_bd, x_bd)
        
        f = ( cos2(psi_d) 
            * ( LLp(r_bd) * cos2(psi_bd) * cos2(psi_bd - psi_d)  
              + LLx(r_bd) * sin2(psi_bd) * sin2(psi_bd - psi_d)  
            ) )

        f *= np.pi * r_b * r_d #the usual 2 disappears because of the factor 1/2 out the front
                              
        return f
    
    def integrand_px_L(params):
        
        r_b, r_d, psi_d = params
    
        y_bd = r_d*np.sin(psi_d)
        x_bd = r_d*np.cos(psi_d) - r_b
        
        r_bd = np.sqrt( y_bd**2 + x_bd**2 ) 
        psi_bd = np.arctan2(y_bd, x_bd)
        
        f = - ( sin2(psi_d) 
            * ( EEp[B][D](r_bd) * cos2(psi_bd) * sin2(psi_bd - psi_d)  
              - EEx[B][D](r_bd) * sin2(psi_bd) * cos2(psi_bd - psi_d)  
            ) )

        f *= np.pi * r_b * r_d #the usual 2 disappears because of the factor 1/2 out the front
                              
        return f
    
    def integrand_px_E(params):
        
        r_b, r_d, psi_d = params
    
        y_bd = r_d*np.sin(psi_d)
        x_bd = r_d*np.cos(psi_d) - r_b
        
        r_bd = np.sqrt( y_bd**2 + x_bd**2 ) 
        psi_bd = np.arctan2(y_bd, x_bd)
        
        f = - ( sin2(psi_d) 
            * ( LLp(r_bd) * cos2(psi_bd) * sin2(psi_bd - psi_d) 
              - LLx(r_bd) * sin2(psi_bd) * cos2(psi_bd - psi_d) 
            ) )

        f *= np.pi * r_b * r_d #the usual 2 disappears because of the factor 1/2 out the front
                              
        return f
    
    def integrand_xp_L(params):
        
        r_b, r_d, psi_d = params
    
        y_bd = r_d*np.sin(psi_d)
        x_bd = r_d*np.cos(psi_d) - r_b
        
        r_bd = np.sqrt( y_bd**2 + x_bd**2 ) 
        psi_bd = np.arctan2(y_bd, x_bd)
        
        f = ( sin2(psi_d) 
            * ( EEp[B][D](r_bd) * sin2(psi_bd) * cos2(psi_bd - psi_d)  
              - EEx[B][D](r_bd) * cos2(psi_bd) * sin2(psi_bd - psi_d)  
            ) )

        f *= np.pi * r_b * r_d #the usual 2 disappears because of the factor 1/2 out the front
                              
        return f
    
    def integrand_xp_E(params):
        
        r_b, r_d, psi_d = params
    
        y_bd = r_d*np.sin(psi_d)
        x_bd = r_d*np.cos(psi_d) - r_b
        
        r_bd = np.sqrt( y_bd**2 + x_bd**2 ) 
        psi_bd = np.arctan2(y_bd, x_bd)
        
        f = ( sin2(psi_d) 
            * ( LLp(r_bd) * sin2(psi_bd) * cos2(psi_bd - psi_d)   
              - LLx(r_bd) * cos2(psi_bd) * sin2(psi_bd - psi_d) 
            ) )

        f *= np.pi * r_b * r_d #the usual 2 disappears because of the factor 1/2 out the front
                              
        return f
    
    def integrand_xx_L(params):
        
        r_b, r_d, psi_d = params
    
        y_bd = r_d*np.sin(psi_d)
        x_bd = r_d*np.cos(psi_d) - r_b
        
        r_bd = np.sqrt( y_bd**2 + x_bd**2 ) 
        psi_bd = np.arctan2(y_bd, x_bd)
        
        f = ( cos2(psi_d) 
            * ( EEp[B][D](r_bd) * sin2(psi_bd) * sin2(psi_bd - psi_d)  
              + EEx[B][D](r_bd) * cos2(psi_bd) * cos2(psi_bd - psi_d)  
            ) )

        f *= np.pi * r_b * r_d #the usual 2 disappears because of the factor 1/2 out the front
                              
        return f
    
    def integrand_xx_E(params):
        
        r_b, r_d, psi_d = params
    
        y_bd = r_d*np.sin(psi_d)
        x_bd = r_d*np.cos(psi_d) - r_b
        
        r_bd = np.sqrt( y_bd**2 + x_bd**2 ) 
        psi_bd = np.arctan2(y_bd, x_bd)
        
        f = ( cos2(psi_d) 
            * ( LLp(r_bd) * sin2(psi_bd) * sin2(psi_bd - psi_d) 
              + LLx(r_bd) * cos2(psi_bd) * cos2(psi_bd - psi_d) 
            ) )

        f *= np.pi * r_b * r_d #the usual 2 disappears because of the factor 1/2 out the front
                              
        return f
    
    
    def integral_bins(integrand, alpha, beta):
        
        ranges = [(rs[alpha], rs[alpha+1]), (rs[beta], rs[beta+1]), (0, 2*np.pi)]
        
        integral, err = monte_carlo_integrate(integrand, ranges, Nsamp)
        
        # normalisation of differential elements
        integral /= (Omegas[alpha] * Omegas[beta]) 
        err      /= (Omegas[alpha] * Omegas[beta]) 
        
        return integral, err
    
    integrand_pp = [integrand_pp_L, integrand_pp_E]
    integrand_px = [integrand_px_L, integrand_px_E]
    integrand_xp = [integrand_xp_L, integrand_xp_E]
    integrand_xx = [integrand_xx_L, integrand_xx_E]

    int_pp, err_pp = integral_bins(integrand_pp, 0, 0)
    int_px, err_px = integral_bins(integrand_px, 0, 0)
    int_xp, err_xp = integral_bins(integrand_xp, 0, 0)
    int_xx, err_xx = integral_bins(integrand_xx, 0, 0)

    return int_pp, int_px, int_xp, int_xx

def generate_ncov_LELE(sigma_L, Nlens, angular_distribution):
    """
    Computes the contribution of noise and sparsity variance in the covariance matrix
    of the LOS shear - galaxy shape correlation functions.
    
    B             : the galaxy redshift bin D (0 to Nbinz_E)
    """

    get_item('LELE_int_pp','LELE_int_px','LELE_int_xp','LELE_int_xx')

    B = 0
    D = B #same redshift bin
    
    Nbin       = angular_distribution.Nbina      #the number of angular bins for LE (sign1) 
    Omegas     = angular_distribution.Omegas     #\Omega_a in the math - the solid angle of bin a (in rad^2) (sign1)
    rs         = angular_distribution.limits     #the angular bin limits for LE (in rad) (sign1)
    
    redshift_distribution = redshift_distributions['E']
    
    G_B    = redshift_distribution.get_ngal(B)         #G_B in the math - the number of galaxies in redshift bin B
    
    # Define the integrands

    ncov_pp = (sigma_L**2/Nlens) * LELE_int_pp[0](rs[1])

    # nerr_pp = (sigma_L**2/Nlens) * err_pp[0]

    scov_pp = (L0/Nlens) * LELE_int_pp[0](rs[1])

    # serr_pp = (L0/Nlens) * err_pp[0]

    ncov_px = (sigma_L**2/Nlens) * LELE_int_px[0](rs[1])

    # nerr_px = (sigma_L**2/Nlens) * err_px[0]

    scov_px = (L0/Nlens) * LELE_int_px[0](rs[1])

    # serr_px = (L0/Nlens) * err_px[0]

    ncov_xp = (sigma_L**2/Nlens) * LELE_int_xp[0](rs[1])

    # nerr_xp = (sigma_L**2/Nlens) * err_xp[0]

    scov_xp = (L0/Nlens) * LELE_int_xp[0](rs[1])

    # serr_xp = (L0/Nlens) * err_xp[0]

    ncov_xx = (sigma_L**2/Nlens) * LELE_int_xx[0](rs[1])

    # nerr_xx = (sigma_L**2/Nlens) * err_xx[0]

    scov_xx = (L0/Nlens) * LELE_int_xx[0](rs[1])

    # serr_xx = (L0/Nlens) * err_xx[0]

    ncov_pp += (sigma_E**2/G_B) * LELE_int_pp[1](rs[1]) 

    # nerr_pp = (
    #          np.sqrt( nerr_pp**2
    #             + ( (sigma_E**2/G_B) * err_pp[1])**2 )
    #           ) 

    scov_pp += (E0[B]/G_B) * LELE_int_pp[1](rs[1]) 

    # serr_pp = (
    #           np.sqrt(  serr_pp**2
    #                   + ( (E0[B]/G_B) * err_pp[1])**2 )
    #           )

    ncov_px += (sigma_E**2/G_B) * LELE_int_px[1](rs[1]) 

    # nerr_px = (
    #           np.sqrt(  nerr_px**2
    #                   + ( (sigma_E**2/G_B) * err_px[1])**2 )
    #           )

    scov_px += (E0[B]/G_B) * LELE_int_px[1](rs[1]) 

    # serr_px = (
    #           np.sqrt(  serr_px**2
    #                   + ( (E0[B]/G_B) * err_px[1])**2 )
    #           )

    ncov_xp += (sigma_E**2/G_B) * LELE_int_xp[1](rs[1]) 

    # nerr_xp = (
    #           np.sqrt(  nerr_xp**2
    #                   + ( (sigma_E**2/G_B) * err_xp[1])**2 )
    #           )

    scov_xp += (E0[B]/G_B) * LELE_int_xp[1](rs[1]) 

    # serr_xp = (
    #           np.sqrt(  serr_xp**2
    #                   + ( (E0[B]/G_B) * err_xp[1])**2 )
    #           )

    ncov_xx += (sigma_E**2/G_B) * LELE_int_xx[1](rs[1]) 

    # nerr_xx = (
    #           np.sqrt(  nerr_xx**2
    #                   + ( (sigma_E**2/G_B) * err_xx[1])**2 )
    #           )

    scov_xx += (E0[B]/G_B) * LELE_int_xx[1](rs[1]) 

    # serr_xx = (
    #           np.sqrt(  serr_xx**2
    #                   + ( (E0[B]/G_B) * err_xx[1])**2 )
    #           )

    cterm_n = ( (1/4) * (1/Nlens) * (1/G_B) 
              * ( sigma_L**2 * sigma_E**2
                + sigma_L**2 * E0[B]
                + L0 * sigma_E**2 ) 
               * Omegatot / Omegas[0] )
    
    cterm_s = ( (1/4) * (1/Nlens) * (1/G_B) 
              * L0 * E0[B] 
               * Omegatot / Omegas[0])

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

