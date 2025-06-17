import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from config import *

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from redshift_distributions import *

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from useful_functions import *

get_item('LLp','LLx','LP', 'angular_distributions', 'redshift_distributions', 'L0', 'E0')

################################################## LLLP cosmic covariance ##############################################################

def generate_ccov_LLLP(D):
    """
    Computes the contribution of cosmic variance in the covariance matrix
    of the LOS shear - LOS shear cross LOS shear - galaxy position correlation functions.
    
    D             : the galaxy redshift bin D (0 to Nbinz_P)
    """

    def generate_matrices(sign):    
    
        angular_distribution1 = angular_distributions[f'LL_{sign}']
        angular_distribution2 = angular_distributions['LP'][D]
        
    
        Nbin1       = angular_distribution1.Nbina      #the number of angular bins for LL 
        Omegas1     = angular_distribution1.Omegas     #\Omega_a in the math - the solid angle of bin a (in rad^2)
        rs1         = angular_distribution1.limits     #the angular bin limits for LL (in rad)
        
        Nbin2       = angular_distribution2.Nbina      #the number of angular bins for Lp 
        Omegas2     = angular_distribution2.Omegas     #\Omega_a' in the math - the solid angle of bin a' (in rad^2)
        rs2         = angular_distribution2.limits     #the angular bin limits for Lp (in rad)
        
        # Initialise the blocks
        ccov_p = np.zeros((Nbin1, Nbin2))
        ccov_x = np.zeros((Nbin1, Nbin2))
        
        # Define the integrands (complete from here)
        
        def integrand_p(params):
            
            psi_j, psi_kd, r_j, r_kd, r_k = params
        
            y_kj = r_j*np.sin(psi_j)
            x_kj = r_j*np.cos(psi_j) - r_k
            
            r_kj = np.sqrt( y_kj**2 + x_kj**2 ) 
            psi_kj = np.arctan2(y_kj, x_kj)
            
            r_jd = cos_law_side(r_kd, r_kj, (psi_kd-psi_kj))
            psi_jd = cos_law_angle(r_kd, r_jd, r_kj) + psi_kd
    
            f = ( LP[D](r_jd) * cos2(psi_jd - psi_j)
                * ( LLp(r_k) * cos2(psi_j) * cos2(psi_kd)
                  + LLx(r_k) * sin2(psi_j) * sin2(psi_kd)
                  ) )
    
            f *= 2 * np.pi * r_k * r_j * r_kd
            
            return f
        
        def integrand_x(params):
            
            psi_j, psi_kd, r_j, r_kd, r_k = params
        
            y_kj = r_j*np.sin(psi_j)
            x_kj = r_j*np.cos(psi_j) - r_k
            
            r_kj = np.sqrt( y_kj**2 + x_kj**2 ) 
            psi_kj = np.arctan2(y_kj, x_kj)
            
            r_jd = cos_law_side(r_kd, r_kj, (psi_kd-psi_kj))
            psi_jd = cos_law_angle(r_kd, r_jd, r_kj) + psi_kd
    
            f = ( LP[D](r_jd) * cos2(psi_jd - psi_j)
                * ( LLp(r_k) * sin2(psi_j) * cos2(psi_kd)
                  - LLx(r_k) * cos2(psi_j) * sin2(psi_kd)
                  ) )
    
            f *= 2 * np.pi * r_k * r_j * r_kd
            
            return f
        
        def integral_bins(integrand, alpha, beta):
            
            ranges = [(0, 2*np.pi), (0, 2*np.pi),
                      (rs1[alpha], rs1[alpha+1]), (rs2[beta], rs2[beta+1]), (0, r2_max)]
            
            integral, err = monte_carlo_integrate(integrand, ranges, Csamp)
            
            # normalisation of differential elements
            integral *= 2/(Omegatot * Omegas1[alpha] * Omegas2[beta]) 
            err *= 2/(Omegatot * Omegas1[alpha] * Omegas2[beta]) 
            return integral, err
        
        for alpha in range(Nbin1):
            for beta in range(Nbin2): 
                         
                ccov_p[alpha, beta], err_p = integral_bins(integrand_p, alpha, beta)
                ccov_x[alpha, beta], err_x = integral_bins(integrand_x, alpha, beta)
    
                test_err(err_p, ccov_p[alpha, beta], f'LLLP ccov plus redshifts {D} angular bins {alpha, beta}')
                test_err(err_x, ccov_x[alpha, beta], f'LLLP ccov times redshifts {D} angular bins {alpha, beta}')

        return ccov_p, ccov_x

    #plus

    ccov_p, ccov_x = generate_matrices('plus')
    
    ccovp = ccov_p + ccov_x
    
    #minus

    ccov_p, ccov_x = generate_matrices('minus')
    
    ccovm = ccov_p - ccov_x
    ccov = np.block([[ccovm],
                     [ccovp]])
    
    return ccov

################################################## LeLe noise/sparsity covariance #############################################################

def generate_ncov_LLLP(D):
    """
    Computes the contribution of noise and sparsity variance in the 
    covariance matrix of the LOS shear - galaxy position correlation functions.
    
    D             : the galaxy redshift bin D (0 to 4)
    """

    def generate_matrices(sign):    
    
        angular_distribution1 = angular_distributions[f'LL_{sign}']
        angular_distribution2 = angular_distributions['LP'][D]
    
        Nbin1       = angular_distribution1.Nbina      #the number of angular bins for LL (sign)
        Omegas1     = angular_distribution1.Omegas     #\Omega_a in the math - the solid angle of bin a (in rad^2)
        rs1         = angular_distribution1.limits     #the angular bin limits for LL (in rad)
        
        Nbin2       = angular_distribution2.Nbina      #the number of angular bins for LP 
        Omegas2     = angular_distribution2.Omegas     #\Omega_a' in the math - the solid angle of bin a' (in rad^2)
        rs2         = angular_distribution2.limits     #the angular bin limits for LP (in rad)
        
        # Initialise the blocks
        ncov_p = np.zeros((Nbin1, Nbin2))
        ncov_x = np.zeros((Nbin1, Nbin2))
        
        scov_p = np.zeros((Nbin1, Nbin2))
        scov_x = np.zeros((Nbin1, Nbin2))
        
        # Define the integrands
        
        def integrand_p(params):
            
            r_i, r_d, psi_d = params
        
            y_id = r_d*np.sin(psi_d)
            x_id = r_d*np.cos(psi_d) - r_i
            
            r_id = np.sqrt( y_id**2 + x_id**2 ) 
            psi_id = np.arctan2(y_id, x_id)
            
            f = LP[D](r_id) * cos2(psi_d) * cos2(psi_id) 
    
            f *= 2 * np.pi * r_i * r_d
                                  
            return f
        
        def integrand_x(params):
            
            r_i, r_d, psi_d = params
        
            y_id = r_d*np.sin(psi_d)
            x_id = r_d*np.cos(psi_d) - r_i
            
            r_id = np.sqrt( y_id**2 + x_id**2 ) 
            psi_id = np.arctan2(y_id, x_id)
            
            f = LP[D](r_id) * sin2(psi_d) * sin2(psi_id) 
    
            f *= 2 * np.pi * r_i * r_d
                                  
            return f
        
        def integral_bins(integrand, alpha, beta):
            
            ranges = [(rs1[alpha], rs1[alpha+1]), (rs2[beta], rs2[beta+1]), (0, 2*np.pi)]
            
            integral, err = monte_carlo_integrate(integrand, ranges, Nsamp)
            
            # normalisation of differential elements
            integral /= Omegas1[alpha] * Omegas2[beta]
            err      /= Omegas1[alpha] * Omegas2[beta]
            
            return integral, err
        
        for alpha in range(Nbin1):
            for beta in range(Nbin2): 
    
                int_p, err_p = integral_bins(integrand_p, alpha, beta)
                int_x, err_x = integral_bins(integrand_x, alpha, beta)
                         
                ncov_p[alpha, beta] = (sigma_L**2/Nlens) * int_p
                nerr_p = (sigma_L**2/Nlens) * err_p
                         
                scov_p[alpha, beta] = (L0/Nlens) * int_p
                serr_p = (L0/Nlens) * err_p
                
                ncov_x[alpha, beta] = (sigma_L**2/Nlens) * int_x
                nerr_x = (sigma_L**2/Nlens) * err_x
                         
                scov_x[alpha, beta] = (L0/Nlens) * int_x
                serr_x = (L0/Nlens) * err_x
    
                test_err(nerr_p, ncov_p[alpha, beta], f'LLLP ncov plus redshift {D} angular bins {alpha, beta}')
                test_err(nerr_x, ncov_x[alpha, beta], f'LLLP ncov times redshift {D} angular bins {alpha, beta}')
    
                test_err(serr_p, scov_p[alpha, beta], f'LLLP scov plus redshift {D} angular bins {alpha, beta}')
                test_err(serr_x, scov_x[alpha, beta], f'LLLP scov times redshift {D} angular bins {alpha, beta}')

        return ncov_p, ncov_x, scov_p, scov_x

    #plus

    ncov_p, ncov_x, scov_p, scov_x = generate_matrices('plus')
    
    ncovp = ncov_p + ncov_x
    scovp = scov_p + scov_x

    #minus

    ncov_p, ncov_x, scov_p, scov_x = generate_matrices('minus')
    
    ncovm = ncov_p - ncov_x
    scovm = scov_p - scov_x
    
    ncov = np.block([[ncovm],
                     [ncovp]])
    
    scov = np.block([[scovm],
                     [scovp]])
    
    return [ncov, scov]