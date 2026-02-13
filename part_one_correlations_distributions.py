##############################################################################################################################
#################################################### 1. IMPORTS ##############################################################
##############################################################################################################################

from config import *                                #all constants, defined in the config.py file
from functions.useful_functions import *            #useful functions, defined in the functions/useful_functions.py file

##############################################################################################################################
############################################ 2  REDSHIFT DISTRIBUTIONS #######################################################
##############################################################################################################################

from functions.redshift_distributions import *      #redshift distributions class

redshift_distribution_E = Redshift_Distributions(NGal, binscheme = binscheme_E, Nbinz = Nbinz_E, zmax_dist = zmax_E)
redshift_distribution_P = Redshift_Distributions(NGal, binscheme = binscheme_P, Nbinz = Nbinz_P, zmax_dist = zmax_P)

redshift_distributions = {"E" : redshift_distribution_E,
                          "P" : redshift_distribution_P}

save_pickle(redshift_distributions, f'data/redshift_distributions', f"Saved redshift distributions")
add_dict(redshift_distributions)

print(f"Finished 2. Redshift Distributions.")

##############################################################################################################################
################################################# 3. CORRELATIONS ############################################################
##############################################################################################################################

if not os.path.exists(f"correlations_NE={Nbinz_E}_NP={Nbinz_P}{correlation_notes}"):
    compute_correlations = True

if compute_correlations:
    
    ####################################### 3.1 Euclid lenses and galaxies ##################################################
    #########################################################################################################################
    
    #reading in the forecasted sample of Euclid lenses
    Euclid_lenses = np.loadtxt('lenses_Euclid.txt')
    zd_Euclid = Euclid_lenses[:, 0]
    zs_Euclid = Euclid_lenses[:, 1]
    
    # convert into comoving distances (in Mpc)
    chid_Euclid = background.comoving_radial_distance(zd_Euclid)
    chis_Euclid = background.comoving_radial_distance(zs_Euclid)
    
    chimax_L = max(chis_Euclid)

    chimax_E = background.comoving_radial_distance(zmax_E)
    chimax_P = background.comoving_radial_distance(zmax_P)

    chimax = max(chimax_L,chimax_E,chimax_P) 
    
    #place these variables in the global dictionary
    add_dict(chimax, chid_Euclid, chis_Euclid, zd_Euclid, zs_Euclid)

    ##############################################################################################################################
    ############################################ 3.2 AUTOCORRELATION FUNCTIONS ###################################################
    ##############################################################################################################################
    
    from functions.correlations.get_correlations import *
    
    ######################################################### 3.2.1 LL ###########################################################
    ##############################################################################################################################
    
    from functions.correlations.LL import *
    
    ################################################# 3.2.1.1 weight functions ######################################################
    
    # Interpolate to get a fast 1D weight function
    Q_LOS_mean_vec = np.vectorize(Q_LOS_mean)
    chi = np.linspace(chimin, chimax, 100)
    Q = Q_LOS_mean_vec(chi)
    Q_LOS_mean_intp = CubicSpline(chi, Q)
    
    # Interpolate to get a fast 1D weight function
    QQ_LOS_mean_vec = np.vectorize(QQ_LOS_mean)
    chi = np.linspace(chimin, chimax, 100)
    QQ = QQ_LOS_mean_vec(chi)
    QQ_rms = np.sqrt(QQ)
    QQ_LOS_rms_intp = CubicSpline(chi, QQ_rms)
    
    add_dict(Q_LOS_mean_intp, QQ_LOS_rms_intp)
    
    ######################################################## 3.2.1.2 cls ###########################################################
    
    ls, cl2, cl1 = get_cl_L(chimax, lmax, nl)
    cl2_LOS_intp = CubicSpline(ls, cl2)
    cl1_LOS_intp = CubicSpline(ls, cl1)
    
    ############################################# 3.2.1.3 correlation functions ####################################################
    
    Theta, LLp, LLx, LL_plus, LL_minus = get_correlations(
        cl2_LOS_intp, Thetamin, Thetamax, nTheta=nTheta)
    
    LLp = CubicSpline(Theta, LLp)
    LLx = CubicSpline(Theta, LLx)
    
    LL_plus = CubicSpline(Theta, LL_plus)
    LL_minus = CubicSpline(Theta, LL_minus)
    
    Theta, L0p, L0x, L0_plus, L0_minus = get_correlations(
        cl1_LOS_intp, Thetamin, Thetamax, nTheta=nTheta)
    
    L0_plus = CubicSpline(Theta, L0_plus)

    L0 = L0_plus(0)

    LL_plus_primitive = compute_antiderivative(LL_plus, Thetamax_LL_plus)
    LL_minus_primitive = compute_antiderivative(LL_minus, Thetamax_LL_minus)
    
    print('Finished 3.2.1 LL autocorrelation functions')
    
    add_dict(LLp, LLx, L0, LL_plus, LL_minus, LL_plus_primitive, LL_minus_primitive)
    
    ######################################################## 3.2.2 EE ###########################################################
    #############################################################################################################################
    
    from functions.correlations.EE import *
    
    ################################################# 3.2.2.1 weight functions ######################################################
    
    # Interpolate to get fast 1D weight functions
    
    Q_os_mean_intp = []
    
    for b in range(Nbinz_E):
        Q_os_mean_vec = np.vectorize(Q_os_mean)
        chi = np.linspace(chimin, chimax, 100)
        Q = Q_os_mean_vec(chi, b)
        Q_os_mean_intp.append(CubicSpline(chi, Q))
    
    # Interpolate to get fast 1D weight functions
    
    QQ_os_rms_intp = []
    
    for b in range(Nbinz_E):
        QQ_os_mean_vec = np.vectorize(QQ_os_mean)
        chi = np.linspace(chimin, chimax, 100)                    #maybe parameterise these?
        QQ = QQ_os_mean_vec(chi, b)
        QQ_rms = np.sqrt(QQ)                             #potential for confusion - WW_rms is actually order W
        QQ_os_rms_intp.append(CubicSpline(chi, QQ_rms))
    
    add_dict(Q_os_mean_intp, QQ_os_rms_intp)
    
    ######################################################## 3.2.2.2 cls ###########################################################
    
    ls_list = []
    cl2_eps_intp = []
    cl1_eps_intp = []
    
    for b1 in range(Nbinz_E): #loop through b
    
        cl2_eps_intp.append([])
    
        for b2 in range(Nbinz_E): #loop through b'
            
            ls, cl2, cl1 = get_cl_E(b1, b2, chimax, lmax, nl) #each time we recalculate everything (a bit inefficient)
            
            cl2_eps_intp[b1].append(CubicSpline(ls, cl2))

            if b1 == b2:
                ls_list.append(ls)
                cl1_eps_intp.append(CubicSpline(ls, cl1))
    
    ############################################# 3.2.2.3 correlation functions ####################################################
    
    Theta_list = []
    EEp_list = []
    EEx_list = []
    E0_list = []
    EE_plus_list = []
    EE_minus_list = []
    
    for b1 in range(Nbinz_E):
            
        EEp_list.append([])
        EEx_list.append([])
            
        EE_plus_list.append([])
        EE_minus_list.append([])
            
        Theta, E0p, E0x, E0_plus, E0_minus = get_correlations(
            cl1_eps_intp[b1], Thetamin, Thetamax, nTheta)
            
        E0_list.append(E0_plus)
    
        for b2 in range(Nbinz_E):
            Theta, EEp, EEx, EE_plus, EE_minus = get_correlations(
                cl2_eps_intp[b1][b2], Thetamin, Thetamax, nTheta)
            
            EEp_list[b1].append(EEp)
            EEx_list[b1].append(EEx)
            
            EE_plus_list[b1].append(EE_plus)
            EE_minus_list[b1].append(EE_minus)
        
        Theta_list.append(Theta)
    
    EEp = []
    EEx = []
    
    EE_plus = []
    EE_minus = []
    
    E0 = []
    
    for b1 in range(Nbinz_E):
        
        EEp.append([])
        EEx.append([])
        
        EE_plus.append([])
        EE_minus.append([])
        
        E0.append(CubicSpline(Theta_list[b1], E0_list[b1])(0))
    
        for b2 in range(Nbinz_E):
            EEp[b1].append(CubicSpline(Theta_list[b1], EEp_list[b1][b2]))
            EEx[b1].append(CubicSpline(Theta_list[b1], EEx_list[b1][b2]))
            
            EE_plus[b1].append(CubicSpline(Theta_list[b1], EE_plus_list[b1][b2]))
            EE_minus[b1].append(CubicSpline(Theta_list[b1], EE_minus_list[b1][b2]))
            
    print('Finished 3.2.2 EE autocorrelation functions')
    
    add_dict(EEp, EEx, E0, EE_plus, EE_minus)

    ######################################################### 3.2.3 PP ###########################################################
    ##############################################################################################################################
    
    from functions.correlations.PP import *
    
    ################################################# 3.2.3.1 weight functions ######################################################
    
    # Interpolate to get fast 1D weight functions
    
    Q_d_mean_intp = []
    
    for b in range(Nbinz_P):
        Q_d_mean_vec = np.vectorize(Q_d)
        chi = np.linspace(chimin, chimax, 100)
        Q = Q_d_mean_vec(chi, b)
        Q_d_mean_intp.append(CubicSpline(chi, Q))
    
    # Interpolate to get fast 1D weight functions
    
    QQ_d_rms_intp = []
    
    for b in range(Nbinz_P):
        QQ_d_mean_vec = np.vectorize(QQ_d)
        chi = np.linspace(1e-5, chimax, 100)                    #maybe parameterise these?
        QQ = QQ_d_mean_vec(chi, b)
        QQ_rms = np.sqrt(QQ)
        QQ_d_rms_intp.append(CubicSpline(chi, QQ_rms))

    Q_d_intp = Q_d_mean_intp  #redundant, fix this
    add_dict(Q_d_mean_intp, QQ_d_rms_intp)
    add_dict(Q_d_intp)
    
    ######################################################## 3.2.3.2 cls ###########################################################
    
    ls_list = []
    cl2_d_intp = []
    
    for b1 in range(Nbinz_P): #loop through b
    
        cl2_d_intp.append([])
        
        for b2 in range(Nbinz_P): #loop through b'
            
            ls, cl2 = get_cl_P(b1, b2, chimax, lmax, nl) #each time we recalculate everything (a bit inefficient)
            
            cl2_d_intp[b1].append(CubicSpline(ls, cl2))

            if b1 == b2:
                ls_list.append(ls)

    #Note - because of the weight function, all of these will be zero unless b1 == b2 
    
    ############################################# 3.2.3.3 correlation functions ####################################################
    
    Theta_list = []
    PP_list = []
    
    for b1 in range(Nbinz_P):
            
        PP_list.append([])
    
        for b2 in range(Nbinz_P):
            
            Theta, PP = get_DD_correlations(
                cl2_d_intp[b1][b2], Thetamin, Thetamax, nTheta)
            
            PP_list[b1].append(PP)
        
        Theta_list.append(Theta)
    
    PP = []
    
    for b1 in range(Nbinz_P):
        
        PP.append([])
    
        for b2 in range(Nbinz_P):
            PP[b1].append(CubicSpline(Theta_list[b1], PP_list[b1][b2]))
            
    print('Finished 3.2.3 PP autocorrelation functions')
    
    add_dict(PP)
    
    ##############################################################################################################################
    ############################################ 3.3 MIXED CORRELATION FUNCTIONS #################################################
    ##############################################################################################################################
    
    ######################################################## 3.3.1 LE ############################################################
    ##############################################################################################################################
    
    from functions.correlations.LE import *
    
    ls_list = []
    cl2LOSos_intp_list = []
    
    for b in range(Nbinz_E):

        ls, cl2LOSos = get_cls_mixed_LE(b, chimax, lmax, nl)

        ls_list.append(ls)

        cl2LOSos_intp_list.append(CubicSpline(ls, cl2LOSos))
    
    Theta_list = []
    
    LEp_list = []
    LEx_list = []

    LE_plus_list = []
    LE_minus_list = []
    
    for b1 in range(Nbinz_E):
        
        Theta, LEp, LEx, LE_plus, LE_minus = get_correlations(
            cl2LOSos_intp_list[b1], Thetamin, Thetamax, nTheta)
        
        LEp_list.append(LEp)
        LEx_list.append(LEx)
        
        LE_plus_list.append(LE_plus)
        LE_minus_list.append(LE_minus)
    
        Theta_list.append(Theta)
    
    LEp = []
    LEx = []
    
    LE_plus = []
    LE_minus = []
    
    LE_plus_primitive = []
    LE_minus_primitive = []
    
    for b1 in range(Nbinz_E):
        
        LEp.append(CubicSpline(Theta_list[b1], LEp_list[b1]))
        LEx.append(CubicSpline(Theta_list[b1], LEx_list[b1]))
        
        LE_plus.append(CubicSpline(Theta_list[b1], LE_plus_list[b1]))
        LE_minus.append(CubicSpline(Theta_list[b1], LE_minus_list[b1]))
        
        LE_plus_primitive.append(compute_antiderivative(LE_plus[b1], Thetamax_LE_plus))
        LE_minus_primitive.append(compute_antiderivative(LE_minus[b1], Thetamax_LE_minus))
    
    print('Finished 3.3.1 LE correlation functions')
    
    add_dict(LEp, LEx, LE_plus, LE_minus, LE_plus_primitive, LE_minus_primitive)
    
    ######################################################### 3.3.2 LP ###########################################################
    ##############################################################################################################################
    
    from functions.correlations.LP import *
    
    ls_list = []
    cl2LOSd_intp_list = []
    
    for b in range(Nbinz_P):
        
        ls, cl2LOSd = get_cls_mixed_LP(b, chimax, lmax, nl)
            
        ls_list.append(ls)
        cl2LOSd_intp_list.append(CubicSpline(ls, cl2LOSd))
    
    Theta_list = []
    
    LP_list = []
    
    for b in range(Nbinz_P):
        
        Theta, LP = get_gD_correlations(
            cl2LOSd_intp_list[b], Thetamin, Thetamax, nTheta)
        
        LP_list.append(LP)
    
        Theta_list.append(Theta)
    
    LP = []
    
    LP_primitive = []
    
    for b in range(Nbinz_P):
        
        LP.append(CubicSpline(Theta_list[b], LP_list[b]))
        
        LP_primitive.append(compute_antiderivative(LP[b], Thetamax_LP))
    
    print('Finished 3.3.2 LP correlation functions')
    
    add_dict(LP, LP_primitive)
    
######################################################### 3.3.3 EP ##########################################################
##############################################################################################################################

    from functions.correlations.EP import *
    
    ls_list = []
    cl2dos_intp_list = []
    
    for b1 in range(Nbinz_E):
        
        cl2dos_intp_list.append([])
    
        for b2 in range(Nbinz_P):
        
            ls, cl2dos = get_cls_mixed_EP(b1, b2, chimax, lmax, nl)
            
            cl2dos_intp_list[b1].append(CubicSpline(ls, cl2dos))
            
        ls_list.append(ls)
    
    Theta_list = []
    
    EP_list = []
    
    for b1 in range(Nbinz_E):
        
        EP_list.append([])
        
        for b2 in range(Nbinz_P):
        
            Theta, EP = get_gD_correlations(
                cl2dos_intp_list[b1][b2], Thetamin, Thetamax, nTheta)
        
            EP_list[b1].append(EP)
    
        Theta_list.append(Theta)
    
    EP = []
    
    for b1 in range(Nbinz_E):
        
        EP.append([])
    
        for b2 in range(Nbinz_P):
            
            EP[b1].append(CubicSpline(Theta_list[b1], EP_list[b1][b2]))
            
    print('Finished 3.3.3 shape position correlation functions')
    
    add_dict(EP)
        
    save_pickle(global_dict, f"correlations_NE={Nbinz_E}_NP={Nbinz_P}{correlation_notes}", f"Saved all correlations")

##############################################################################################################################
############################################## 4. Preparing for part 2  ######################################################
##############################################################################################################################

############################################### 4.1 Creating params.txt ######################################################
##############################################################################################################################

# Define parameter ranges
sigma_L_values = np.logspace(sigL_lower, sigL_upper, sigL_n) 
Nlens_values = np.logspace(Nlens_lower, Nlens_upper, Nlens_n, dtype=int) 

# Create all combinations
with open('params.txt', 'w') as f:
    for sigma_L in sigma_L_values:
        for Nlens in Nlens_values:
            f.write(f"{sigma_L:.6e} {Nlens}\n") 

total = len(sigma_L_values) * len(Nlens_values)
print(f"Total combinations: {total}")

############################################### 4.2 Creating tasks.txt ######################################################
##############################################################################################################################

# For parallelising step 2
with open('tasks.txt', 'w') as f:
    f.write(f"LLLL_ccov\n") 
    f.write(f"LELE_ccov\n") 
    f.write(f"LPLP_ccov\n") 
    f.write(f"LLLL_ncov\n") 
    f.write(f"LELE_ncov\n") 
    f.write(f"LPLP_ncov\n") 