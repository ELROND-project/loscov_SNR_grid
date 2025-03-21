##############################################################################################################################
################################################ 1. PRELIMINARIES ############################################################
##############################################################################################################################

from config import *                                #all constants, defined in the config.py file

################################################## 1.1 Parameters ############################################################
##############################################################################################################################

from functions.useful_functions import *            #useful functions, defined in the functions/useful_functions.py file

if compute_correlations:

    Thetamin = arcmintorad(Thetamin_arcmin)  #minimum theta from which we calculate correlation functions (in radians)
    
    ####################################### 1.2 Euclid lenses and galaxies ##################################################
    #########################################################################################################################
    
    #reading in the forecasted sample of Euclid lenses
    Euclid_lenses = np.loadtxt('lenses_Euclid.txt')
    zd = Euclid_lenses[:, 0]
    zs = Euclid_lenses[:, 1]
    
    # convert into comoving distances (in Mpc)
    chid = background.comoving_radial_distance(zd)
    chis = background.comoving_radial_distance(zs)
    
    chimax_lens = max(chis)

    zmax_gal = max(binparams['redshifts']) 

    chimax_gal = background.comoving_radial_distance(zmax_gal)

    chimax = max(chimax_lens,chimax_gal) 
    
    #place these variables in the global dictionary
    add_dict(chimax, chid, chis, zd, zs)

    ##############################################################################################################################
    ############################################# 2. AUTOCORRELATION FUNCTIONS ###################################################
    ##############################################################################################################################
    
    from functions.correlations.get_correlations import *
    
    ######################################################## 2.1 Shear ###########################################################
    ##############################################################################################################################
    
    
    from functions.correlations.shear import *
    
    ################################################# 2.1.1 weight functions ######################################################
    
    # Interpolate to get a fast 1D weight function
    W_LOS_mean_vec = np.vectorize(W_LOS_mean)
    chi = np.linspace(0, chimax, 100)
    W = W_LOS_mean_vec(chi)
    W_LOS_mean_intp = CubicSpline(chi, W)
    
    # Interpolate to get a fast 1D weight function
    WW_LOS_mean_vec = np.vectorize(WW_LOS_mean)
    chi = np.linspace(0, chimax, 100)
    WW = WW_LOS_mean_vec(chi)
    WW_rms = np.sqrt(WW)
    WW_LOS_rms_intp = CubicSpline(chi, WW_rms)
    
    add_dict(W_LOS_mean_intp, WW_LOS_rms_intp)
    
    ######################################################## 2.1.2 cls ###########################################################
    
    ls, cl2, cl1, cl32 = get_cls_gamma_LOS(chimax, lmax, nl)
    cl2_LOS_intp = CubicSpline(ls, cl2)
    cl1_LOS_intp = CubicSpline(ls, cl1)
    cl32_LOS_intp = CubicSpline(ls, cl32)
    
    ############################################# 2.1.3 correlation functions ####################################################
    
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
    
    print('Finished 2.1 LOS autocorrelation functions')
    
    add_dict(xi2_LOS_plus_intp, xi1_LOS_plus_intp, xi32_LOS_plus_intp, xi2_LOS_minus_intp, xi1_LOS_minus_intp, xi32_LOS_minus_intp)
    
    ######################################################## 2.2 Shape ###########################################################
    ##############################################################################################################################
    
    
    from functions.correlations.shape import *
    
    ################################################# 2.2.1 weight functions ######################################################
    
    # Interpolate to get fast 1D weight functions
    
    W_os_mean_intp = []
    
    for b in range(5):
        W_os_mean_vec = np.vectorize(W_os_mean)
        chi = np.linspace(1e-3, chimax, 100)
        W = W_os_mean_vec(chi, b)
        W_os_mean_intp.append(CubicSpline(chi, W))
    
    # Interpolate to get fast 1D weight functions
    
    WW_os_rms_intp = []
    
    for b in range(5):
        WW_os_mean_vec = np.vectorize(WW_os_mean)
        chi = np.linspace(1e-5, chimax, 100)                    #maybe parameterise these?
        WW = WW_os_mean_vec(chi, b)
        WW_rms = np.sqrt(WW)                             #potential for confusion - WW_rms is actually order W
        WW_os_rms_intp.append(CubicSpline(chi, WW_rms))
    
    add_dict(W_os_mean_intp, WW_os_rms_intp)
    
    ######################################################## 2.2.2 cls ###########################################################
    
    ls_list = []
    cl2_eps_intp = []
    cl1_eps_intp = []
    
    for b1 in range(5): #loop through b
    
        cl2_eps_intp.append([])
    
        for b2 in range(5): #loop through b'
            
            ls, cl2, cl1 = get_cl_gamma(b1, b2, chimax, lmax, nl) #each time we recalculate everything (a bit inefficient)
            
            cl2_eps_intp[b1].append(CubicSpline(ls, cl2))
    
        ls_list.append(ls)
        cl1_eps_intp.append(CubicSpline(ls, cl1))
    
    ############################################# 2.2.3 correlation functions ####################################################
    
    Theta_list = []
    xi2_eps_plus_list = []
    xi2_eps_minus_list = []
    xi1_eps_plus_list = []
    xi1_eps_minus_list = []
    
    for b1 in range(5):
            
        xi2_eps_plus_list.append([])
        xi2_eps_minus_list.append([])
        
        Theta, xi1_eps_plus, xi1_eps_minus = get_correlations(
            cl1_eps_intp[b1], Thetamin, Thetamax, nTheta)
    
        for b2 in range(5):
            Theta, xi2_eps_plus, xi2_eps_minus = get_correlations(
                cl2_eps_intp[b1][b2], Thetamin, Thetamax, nTheta)
            
            xi2_eps_plus_list[b1].append(xi2_eps_plus)
            xi2_eps_minus_list[b1].append(xi2_eps_minus)
        
        Theta_list.append(Theta)
        xi1_eps_plus_list.append(xi1_eps_plus)
        xi1_eps_minus_list.append(xi1_eps_minus)
    
    xi2_eps_plus_intp = []
    xi1_eps_plus_intp = []
    xi2_eps_minus_intp = []
    xi1_eps_minus_intp = []
    
    for b1 in range(5):
        
        xi1_eps_plus_intp.append(CubicSpline(Theta_list[b1], xi1_eps_plus_list[b1]))
        xi1_eps_minus_intp.append(CubicSpline(Theta_list[b1], xi1_eps_minus_list[b1]))
        xi2_eps_plus_intp.append([])
        xi2_eps_minus_intp.append([])
    
        for b2 in range(5):
            xi2_eps_plus_intp[b1].append(CubicSpline(Theta_list[b1], xi2_eps_plus_list[b1][b2]))
            xi2_eps_minus_intp[b1].append(CubicSpline(Theta_list[b1], xi2_eps_minus_list[b1][b2]))
            
    print('Finished 2.2 shape autocorrelation functions')
    
    add_dict(xi2_eps_plus_intp,xi1_eps_plus_intp,xi2_eps_minus_intp,xi1_eps_minus_intp)
    
    ######################################################## 2.3 Position ########################################################
    ##############################################################################################################################

    
    from functions.correlations.position import *
    
    ################################################# 2.3.1 weight functions ######################################################
    
    # Interpolate to get fast 1D weight functions
    
    W_d_mean_intp = []
    
    for b in range(5):
        W_d_mean_vec = np.vectorize(W_d)
        chi = np.linspace(1e-3, chimax, 100)
        W = W_d_mean_vec(chi, b)
        W_d_mean_intp.append(CubicSpline(chi, W))
    
    # Interpolate to get fast 1D weight functions
    
    WW_d_rms_intp = []
    
    for b in range(5):
        WW_d_mean_vec = np.vectorize(WW_d)
        chi = np.linspace(1e-5, chimax, 100)                    #maybe parameterise these?
        WW = WW_d_mean_vec(chi, b)
        WW_rms = np.sqrt(WW)
        WW_d_rms_intp.append(CubicSpline(chi, WW_rms))

    W_d_intp = W_d_mean_intp  #redundant, fix this
    add_dict(W_d_mean_intp, WW_d_rms_intp)
    add_dict(W_d_intp)
    
    ######################################################## 2.3.2 cls ###########################################################
    
    ls_list = []
    cl1_d_intp = []
    cl2_d_intp = []
    cl32_d_intp = []
    
    for b1 in range(5): #loop through b
    
        cl2_d_intp.append([])
        cl32_d_intp.append([])
        
        for b2 in range(5): #loop through b'
            
            ls, cl2, cl1, cl32 = get_cl_d(b1, b2, chimax, lmax, nl) #each time we recalculate everything (a bit inefficient)
            
            cl2_d_intp[b1].append(CubicSpline(ls, cl2))
            cl32_d_intp[b1].append(CubicSpline(ls, cl32))     #almost certainly wrong
        
        ls_list.append(ls)
        cl1_d_intp.append(CubicSpline(ls, cl1)) 

    #Note - because of the weight function, all of these will be zero unless b1 == b2 
    
    ############################################# 2.3.3 correlation functions ####################################################
    
    Theta_list = []
    xi2_d_list = []
    xi1_d_list = []
    xi32_d_list = []
    
    for b1 in range(5):
            
        xi2_d_list.append([])
        xi32_d_list.append([])
            
        Theta, xi1_d = get_DD_correlations(
            cl1_d_intp[b1], Thetamin, Thetamax, nTheta)
        
        Theta_list.append(Theta)
        xi1_d_list.append(xi1_d)
    
        for b2 in range(5):
            Theta, xi2_d = get_DD_correlations(
                cl2_d_intp[b1][b2], Thetamin, Thetamax, nTheta)
            
            xi2_d_list[b1].append(xi2_d)
            
            Theta, xi32_d = get_DD_correlations(
                cl32_d_intp[b1][b2], Thetamin, Thetamax, nTheta)
            
            xi32_d_list[b1].append(xi32_d)
    
    xi2_d_intp = []
    xi1_d_intp = []
    xi32_d_intp = []
    
    for b1 in range(5):
        
        xi2_d_intp.append([])
        xi32_d_intp.append([])
        xi1_d_intp.append(CubicSpline(Theta_list[b1], xi1_d_list[b1]))
    
        for b2 in range(5):
            xi2_d_intp[b1].append(CubicSpline(Theta_list[b1], xi2_d_list[b1][b2]))
            xi32_d_intp[b1].append(CubicSpline(Theta_list[b1], xi32_d_list[b1][b2]))
            
    print('Finished 2.3 position autocorrelation functions')
    
    add_dict(xi2_d_intp, xi1_d_intp, xi32_d_intp)
    
    ##############################################################################################################################
    ############################################ 3. MIXED CORRELATION FUNCTIONS ##################################################
    ##############################################################################################################################
    
    ##################################################### 3.1 shear shape ########################################################
    ##############################################################################################################################
    
    from functions.correlations.shear_shape import *
    
    ls_list = []
    cl2LOSos_intp_list = []
    cl32LOSos2_intp_list = []
    cl32LOS2os_intp_list = []
    cl1LOSos_intp_list = []
    
    for b in range(5):

        ls, cl2LOSos, cl32LOSos2, cl32LOS2os, cl1LOSos = get_cls_mixed_LOS_os(b, chimax, lmax, nl)

        ls_list.append(ls)

        cl2LOSos_intp_list.append(CubicSpline(ls, cl2LOSos))
        cl32LOSos2_intp_list.append(CubicSpline(ls, cl32LOSos2))
        cl32LOS2os_intp_list.append(CubicSpline(ls, cl32LOS2os))
        cl1LOSos_intp_list.append(CubicSpline(ls, cl1LOSos))
    
    Theta_list = []
    
    xi2_LOS_eps_plus_list = []
    xi2_LOS_eps_minus_list = []
    
    xi32_LOS_eps2_plus_list = []
    xi32_LOS_eps2_minus_list = []
    
    xi32_LOS2_eps_plus_list = []
    xi32_LOS2_eps_minus_list = []
    
    xi1_LOS_eps_plus_list = []
    xi1_LOS_eps_minus_list = []
    
    for b1 in range(5):
        
        Theta, xi2_LOS_eps_plus, xi2_LOS_eps_minus = get_correlations(
            cl2LOSos_intp_list[b1], Thetamin, Thetamax, nTheta)
        
        xi2_LOS_eps_plus_list.append(xi2_LOS_eps_plus)
        xi2_LOS_eps_minus_list.append(xi2_LOS_eps_minus)
        
        Theta, xi32_LOS_eps2_plus, xi32_LOS_eps2_minus = get_correlations(
            cl32LOSos2_intp_list[b1], Thetamin, Thetamax, nTheta)
        
        xi32_LOS_eps2_plus_list.append(xi32_LOS_eps2_plus)
        xi32_LOS_eps2_minus_list.append(xi32_LOS_eps2_minus)
        
        Theta, xi32_LOS2_eps_plus, xi32_LOS2_eps_minus = get_correlations(
            cl32LOS2os_intp_list[b1], Thetamin, Thetamax, nTheta)
        
        xi32_LOS2_eps_plus_list.append(xi32_LOS2_eps_plus)
        xi32_LOS2_eps_minus_list.append(xi32_LOS2_eps_minus)
        
        Theta, xi1_LOS_eps_plus, xi1_LOS_eps_minus = get_correlations(
            cl1LOSos_intp_list[b1], Thetamin, Thetamax, nTheta)
        
        xi1_LOS_eps_plus_list.append(xi1_LOS_eps_plus)
        xi1_LOS_eps_minus_list.append(xi1_LOS_eps_minus)
    
        Theta_list.append(Theta)
    
    xi2_LOS_eps_plus_intp = []
    xi2_LOS_eps_minus_intp = []
    
    xi32_LOS_eps2_plus_intp = []
    xi32_LOS_eps2_minus_intp = []
    
    xi32_LOS2_eps_plus_intp = []
    xi32_LOS2_eps_minus_intp = []
    
    xi1_LOS_eps_plus_intp = []
    xi1_LOS_eps_minus_intp = []
    
    for b1 in range(5):
        
        xi2_LOS_eps_plus_intp.append(CubicSpline(Theta_list[b1], xi2_LOS_eps_plus_list[b1]))
        xi2_LOS_eps_minus_intp.append(CubicSpline(Theta_list[b1], xi2_LOS_eps_minus_list[b1]))
            
        xi32_LOS_eps2_plus_intp.append(CubicSpline(Theta_list[b1], xi32_LOS_eps2_plus_list[b1]))
        xi32_LOS_eps2_minus_intp.append(CubicSpline(Theta_list[b1], xi32_LOS_eps2_minus_list[b1]))
        
        xi32_LOS2_eps_plus_intp.append(CubicSpline(Theta_list[b1], xi32_LOS2_eps_plus_list[b1]))
        xi32_LOS2_eps_minus_intp.append(CubicSpline(Theta_list[b1], xi32_LOS2_eps_minus_list[b1]))
            
        xi1_LOS_eps_plus_intp.append(CubicSpline(Theta_list[b1], xi1_LOS_eps_plus_list[b1]))
        xi1_LOS_eps_minus_intp.append(CubicSpline(Theta_list[b1], xi1_LOS_eps_minus_list[b1]))
    
    print('Finished 3.1 shear shape correlation functions')
    
    add_dict(xi2_LOS_eps_plus_intp, xi2_LOS_eps_minus_intp, xi32_LOS_eps2_plus_intp, xi32_LOS_eps2_minus_intp, xi32_LOS2_eps_plus_intp, xi32_LOS2_eps_minus_intp, xi1_LOS_eps_plus_intp, xi1_LOS_eps_minus_intp)
    
    #################################################### 3.2 shear position ######################################################
    ##############################################################################################################################
    
    from functions.correlations.shear_position import *
    
    ls_list = []
    cl2LOSd_intp_list = []
    cl32LOSd2_intp_list = []
    cl32LOS2d_intp_list = []
    cl1LOSd_intp_list = []
    
    for b in range(5):
        
        ls, cl2LOSd, cl32LOSd2, cl32LOS2d, cl1LOSd = get_cls_mixed_LOS_d(b, chimax, lmax, nl)
            
        ls_list.append(ls)
        cl2LOSd_intp_list.append(CubicSpline(ls, cl2LOSd))
        cl32LOS2d_intp_list.append(CubicSpline(ls, cl32LOS2d))
        cl32LOSd2_intp_list.append(CubicSpline(ls, cl32LOSd2))
        cl1LOSd_intp_list.append(CubicSpline(ls, cl1LOSd))
    
    Theta_list = []
    
    xi2_LOS_d_list = []
    
    xi32_LOS_d2_list = []
    
    xi32_LOS2_d_list = []
    
    xi1_LOS_d_list = []
    
    for b in range(5):
        
        Theta, xi2_LOS_d = get_gD_correlations(
            cl2LOSd_intp_list[b], Thetamin, Thetamax, nTheta)
        
        xi2_LOS_d_list.append(xi2_LOS_d)
        
        Theta, xi32_LOS2_d = get_gD_correlations(
            cl32LOS2d_intp_list[b], Thetamin, Thetamax, nTheta)
        
        xi32_LOS2_d_list.append(xi32_LOS2_d)
        
        Theta, xi32_LOS_d2 = get_gD_correlations(
            cl32LOSd2_intp_list[b], Thetamin, Thetamax, nTheta)
        
        xi32_LOS_d2_list.append(xi32_LOS_d2)
        
        Theta, xi1_LOS_d = get_gD_correlations(
            cl1LOSd_intp_list[b], Thetamin, Thetamax, nTheta)
        
        xi1_LOS_d_list.append(xi1_LOS_d)
    
        Theta_list.append(Theta)
    
    xi2_LOS_d_intp = []
    
    xi32_LOS_d2_intp = []
    
    xi32_LOS2_d_intp = []
    
    xi1_LOS_d_intp = []
    
    for b in range(5):
        
        xi2_LOS_d_intp.append(CubicSpline(Theta_list[b], xi2_LOS_d_list[b]))
        
        xi32_LOS2_d_intp.append(CubicSpline(Theta_list[b], xi32_LOS2_d_list[b]))
            
        xi32_LOS_d2_intp.append(CubicSpline(Theta_list[b], xi32_LOS_d2_list[b]))
            
        xi1_LOS_d_intp.append(CubicSpline(Theta_list[b], xi1_LOS_d_list[b]))
    
    print('Finished 3.2 shear position correlation functions')
    
    add_dict(xi2_LOS_d_intp, xi32_LOS_d2_intp, xi32_LOS2_d_intp, xi1_LOS_d_intp)
    
    #################################################### 3.3 shape position ######################################################
    ##############################################################################################################################
    
    from functions.correlations.position_shape import *
    
    ls_list = []
    cl2dos_intp_list = []
    cl32dos2_intp_list = []
    cl32d2os_intp_list = []
    cl1dos_intp_list = []
    
    for b1 in range(5):
        
        cl2dos_intp_list.append([])
        cl32dos2_intp_list.append([])
        cl32d2os_intp_list.append([])
        cl1dos_intp_list.append([])
    
        for b2 in range(5):
        
            ls, cl2dos, cl32dos2, cl32d2os, cl1dos = get_cls_mixed_d_os(b1, b2, chimax, lmax, nl)
            
            cl2dos_intp_list[b1].append(CubicSpline(ls, cl2dos))
            cl32dos2_intp_list[b1].append(CubicSpline(ls, cl32dos2))
            cl32d2os_intp_list[b1].append(CubicSpline(ls, cl32d2os))
            cl1dos_intp_list[b1].append(CubicSpline(ls, cl1dos))
            
        ls_list.append(ls)
    
    Theta_list = []
    
    xi2_d_eps_list = []
    
    xi32_d_eps2_list = []
    
    xi32_d2_eps_list = []
    
    xi1_d_eps_list = []
    
    for b1 in range(5):
        
        xi2_d_eps_list.append([])
        
        xi32_d_eps2_list.append([])
    
        xi32_d2_eps_list.append([])
    
        xi1_d_eps_list.append([])
        
        for b2 in range(5):
        
            Theta, xi2_d_eps_plus = get_gD_correlations(
                cl2dos_intp_list[b1][b2], Thetamin, Thetamax, nTheta)
        
            xi2_d_eps_list[b1].append(xi2_d_eps_plus)
        
            Theta, xi32_d_eps2_plus = get_gD_correlations(
                cl32dos2_intp_list[b1][b2], Thetamin, Thetamax, nTheta)
        
            xi32_d_eps2_list[b1].append(xi32_d_eps2_plus)
        
            Theta, xi32_d2_eps_plus = get_gD_correlations(
                cl32d2os_intp_list[b1][b2], Thetamin, Thetamax, nTheta)
        
            xi32_d2_eps_list[b1].append(xi32_d2_eps_plus)
        
            Theta, xi1_d_eps_plus = get_gD_correlations(
                cl1dos_intp_list[b1][b2], Thetamin, Thetamax, nTheta)
        
            xi1_d_eps_list[b1].append(xi1_d_eps_plus)
    
        Theta_list.append(Theta)
    
    xi2_d_eps_intp = []
    
    xi32_d_eps2_intp = []
    
    xi32_d2_eps_intp = []
    
    xi1_d_eps_intp = []
    
    for b1 in range(5):
        
        xi2_d_eps_intp.append([])
        xi32_d2_eps_intp.append([])
        xi32_d_eps2_intp.append([])
        xi1_d_eps_intp.append([])
    
        for b2 in range(5):
            
            xi2_d_eps_intp[b1].append(CubicSpline(Theta_list[b1], xi2_d_eps_list[b1][b2]))
        
            xi32_d_eps2_intp[b1].append(CubicSpline(Theta_list[b1], xi32_d_eps2_list[b1][b2]))
            
            xi32_d2_eps_intp[b1].append(CubicSpline(Theta_list[b1], xi32_d2_eps_list[b1][b2]))
            
            xi1_d_eps_intp[b1].append(CubicSpline(Theta_list[b1], xi1_d_eps_list[b1][b2]))
            
    print('Finished 3.3 shape position correlation functions')
    
    add_dict(xi2_d_eps_intp, xi32_d2_eps_intp, xi32_d_eps2_intp, xi1_d_eps_intp)
        
    save_pickle(global_dict, 'correlations', f"Saved all correlations")

else:
    
    load_correlations(filename="correlations")

##############################################################################################################################
######################################### 4. GENERATING COVARIANCE MATRICES ##################################################
##############################################################################################################################


from functions.distributions_and_correlations import *

from functions.covariance.LLLL_cov import *
from functions.covariance.LeLe_cov import *
from functions.covariance.LLLe_cov import *
from functions.covariance.LpLp_cov import *
from functions.covariance.LLLp_cov import *
from functions.covariance.LpLe_cov import *

###################################### 4.1 Defining the distributions classes ################################################
##############################################################################################################################

distribution_LL = Distributions(Nlens, binscheme=binscheme_LL, Nbina=Nbina_LL, Thetamax=Thetamax_LL) 
distribution_Le = Distributions(NGal, binscheme=binscheme_Le, Nbina=Nbina_Le, Thetamax=Thetamax_Le)
distribution_Lp = Distributions(NGal, binscheme=binscheme_Lp, Nbina=Nbina_Lp, Thetamax=Thetamax_Lp)

distributions = {"LL": distribution_LL,
                "Le": distribution_Le,
                "Lp": distribution_Lp}

#distributions = Distributions(Nlens=Nlens, sky_coverage=sky_coverage, Nbina=Nbina, Thetamax=Thetamax_dist)      #old implementation

########################################## 4.2 The process pair function #####################################################
##############################################################################################################################

def compute_covariance_piece(args):
    """Computes a specific covariance component."""
    b1, b2, distributions, sigma_noise, sigma_shape, cov_matrix, cov_type = args

    func = globals().get(f"generate_{cov_type}_{cov_matrix}")
    if func is None:
        print(f"Error: Covariance function {cov_type}_{cov_matrix} not found.")
        return (b1, b2, cov_matrix, cov_type, None)

    if cov_type == "ncov":
        result = func(distributions, sigma_noise, sigma_shape, b1, b2, use_approx)
    else:
        result = func(distributions, b1, b2, use_approx)

    return (b1, b2, cov_matrix, cov_type, {
        "full": result[0],
        "pp": result[1],
        "mm": result[2],
        "pm": result[3],
        "mp": result[4]
    })

def process_pair(args):
    """Handles correlation function computations and file saving."""
    b1, b2, distributions, sigma_noise, sigma_shape, cov_matrix = args

    # Create output directory if it doesn't exist
    output_dir = f"{matrices_folder}/{cov_matrix}"
    os.makedirs(output_dir, exist_ok=True)

    # Generate binned correlation functions
    correlation_data = generate_binned_correlation(distributions, cov_matrix, b1, b2)

    if correlation_data:
        for suffix, data in correlation_data.items():
            filename = f"{output_dir}/{suffix}{b1}" if b1 is not None else f"{output_dir}/{suffix}"
            save_pickle(data, filename, f"{suffix} for b1={b1}, b2={b2}, cov_matrix={cov_matrix}")

def main():
    """Main function to dispatch parallel jobs."""
    # Define parameter ranges
    b1_values = [0, 1, 2, 3, 4]
    b2_values = [0, 1, 2, 3, 4]
    
    cov_matrices_full = ['LpLe', 'LeLe', 'LpLp']   # Needs (b1, b2)
    cov_matrices_b1 = ['LLLp', 'LLLe']             # Needs only b1
    cov_matrices_no_b = ['LLLL']                   # No (b1, b2)

    # Prepare argument lists
    pair_args = []
    cov_args = []
    cov_types = ["ncov", "ccov", "scov"]

    # Full (b1, b2) iteration
    for b1, b2 in product(b1_values, b2_values):
        for cov_matrix in cov_matrices_full:
            pair_args.append((b1, b2, distributions, sigma_noise, sigma_shape, cov_matrix))
            for cov_type in cov_types:
                cov_args.append((b1, b2, distributions, sigma_noise, sigma_shape, cov_matrix, cov_type))

    # Single b1 iteration
    for cov_matrix, b1 in product(cov_matrices_b1, b1_values):
        pair_args.append((b1, None, distributions, sigma_noise, sigma_shape, cov_matrix))
        for cov_type in cov_types:
            cov_args.append((b1, None, distributions, sigma_noise, sigma_shape, cov_matrix, cov_type))

    # No b1, b2 iteration (only one call)
    for cov_matrix in cov_matrices_no_b:
        pair_args.append((None, None, distributions, sigma_noise, sigma_shape, cov_matrix))
        for cov_type in cov_types:
            cov_args.append((None, None, distributions, sigma_noise, sigma_shape, cov_matrix, cov_type))

    # Run all jobs in parallel
    with Pool(processes=cpu_count()) as pool:
        # Process correlation functions first
        pool.map(process_pair, pair_args)

        # Compute covariance matrices and save immediately
        for result in pool.imap_unordered(compute_covariance_piece, cov_args):
            if result is None:
                continue  # Skip failed computations

            b1, b2, cov_matrix, cov_type, data = result
            output_dir = f"{matrices_folder}/{cov_matrix}"
            os.makedirs(output_dir, exist_ok=True)
            
            filename = f"{output_dir}/{cov_type}"
            if b1 is not None:
                filename += f"{b1}"
            if b2 is not None:
                filename += f"{b2}"
            
            save_pickle(data, filename, f"{cov_type} for b1={b1}, b2={b2}, cov_matrix={cov_matrix}")

if __name__ == "__main__":
    main()

print("")
print("Finito! :)")