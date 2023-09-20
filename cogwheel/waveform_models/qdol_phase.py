"""
Importing this module will override the code of
``waveform.compute_hplus_hcross`` and print a warning that this was
done.
Useful for implementing custom waveform models.
"""

import warnings
from cogwheel import waveform
from cogwheel.waveform import Approximant
import numpy as np

###################################################################################
#################### Tidal waveform model params and functions ####################
###################################################################################

###### Wf model params, copied from ripple.constants.py and ripple.typing.py ######

EulerGamma = 0.577215664901532860606512090082402431
MSUN = 1.988409902147041637325262574352366540e30  # solar mass in kg
G = 6.67430e-11  # m^3 / kg / s^2 # Newton's gravitational constant
C = 299792458.0  # m / s # speed of light
PI = 3.141592653589793238462643383279502884
gt = G * MSUN / (C ** 3.0) # G MSUN / C^3 in seconds
m_per_Mpc = 3.085677581491367278913937957796471611e22 # meters per Mpc

###################################################################################
###################################################################################

def phase_qdol(f, theta, fix_bh_superradiance=True):
    """
    
    Computes the phase of the TaylorF2 waveform + finite-size effect parameters. 
    Spin-induced moments and spin-dependent dissipation up to 3.5PN;
    Spin-independent dissipation at 4PN;
    Love number at 5PN.
    Sets time and phase of coealence to be zero.
    
    Parameters
    -----------
    f (array): Frequency at which to output the phase [Hz]
    theta: Source Intrinsic parameters
    
    Mc : Chrip Mass [Msun]
    eta : m1*m2/(m1+m2)**2
    chi_1, chi_2 : Component spins in the orbital angular momentum direction
    kappa1, kappa2 : Component spin-induced quadrupoles 
    h1s1, h2s1 : Component linear-in-spin dissipation numbers
    lambda1, lambda2 : Component spin-induced octupoles
    h1s3, h2s3 : Component cubic-in-spin dissipation numbers 
    h1s0, h2s0 : Component spin-independent dissipation numbers
    l1, l2 : Component Love numbers

    Returns
    -------
    phase (array): GW phase as a function of frequency
    
    """
    
    Mc, eta, chi_1, chi_2, kappa1, kappa2, h1s1, h2s1, lambda1, lambda2, \
    h1s3,h2s3, h1s0,h2s0, l1, l2 = theta
        
    # Mass parameters
    M = Mc / (eta ** (3 / 5))
    m1 = (M + np.sqrt(M**2 - 4 * (eta * M**2))) / 2
    m2 = (M - np.sqrt(M**2 - 4 * (eta * M**2))) / 2
    delta = np.sqrt(1.0 - 4.0 * eta)

    # Convenient variables
    # vlso doesn't matter for 2.5PN and 4PN constant log(v/vlso) terms (absorbed into overall phase and time)
    # but can in principle matter for log(v/vlso)**2 terms starting at 4PN
    vlso = 1.0 
    v = (PI * M * (f + 1e-100) * gt) ** (1.0 / 3.0)
    v2 = v * v
    v3 = v2 * v
    v4 = v2 * v2
    v5 = v4 * v
    v6 = v3 * v3
    v7 = v3 * v4
    v8 = v4 * v4
    v9 = v4 * v5
    v10 = v5 * v5
    v12 = v10 * v2
    eta2 = eta**2
    eta3 = eta**3
    eta4 = eta**4

    # # Black hole finite-size values
    # kappa1 = 1.0
    # kappa2 = 1.0
    # h1s1 = -8.0 / 45.0 * (1 + 3*chi_1**2)
    # h2s2 = -8.0 / 45.0 * (1 + 3*chi_2**2)
    # lambda1 = 1.0
    # lambda2 = 1.0
    # h1s3 = 2.0 / 3.0
    # h2s3 = 2.0 / 3.0
    # h1s0 = 16.0 / 45.0 # small spin limit
    # h2s0 = 16.0 / 45.0 # small spin limit
    # l1 = 0
    # l2 = 0

    # Symmetric/antisymmetric spins and spin-induced multipoles
    chi_s = 0.5 * (chi_1 + chi_2)
    chi_a = 0.5 * (chi_1 - chi_2)
    k_s = 0.5 * (kappa1 + kappa2)
    k_a = 0.5 * (kappa1 - kappa2)
    lambda_s = 0.5 * (lambda1 + lambda2)
    lambda_a = 0.5 * (lambda1 - lambda2)
    
    # Effective Love number parameters
    l_tilde = 16.0 * ((m1 + 12 * m2) * m1**4 * l1 + (m2 + 12 * m1) * m2**4 * l2) / (13.0 * M**5.0)
    
    
    ######## Effective dissipation number parameters ########
    # 2.5PN Dissipation (Linear spin)
    hs = (h1s1*m1**3 + h2s1*m2**3) / M**3
    ha = (h1s1*m1**3 - h2s1*m2**3) / M**3
    
    # 3.5PN Dissipation (Cubic spin)
    hs3 = (h1s3*m1**3 + h2s3*m2**3) / M**3
    ha3 = (h1s3*m1**3 - h2s3*m2**3) / M**3
    
    # 4PN Dissipation (LO non-spinning) 
    # Here one could impose the superradiance condition (m Omega_H - omega), which is valid at LO in spin. 
    # Good approximation if the component spins are small.
    if fix_bh_superradiance:
        # factor of -2 strictly only applies for Schw black holes
        h0 = (-2.0 * h1s1 * m1 ** 4 - 2.0 * h2s1 * m2 ** 4) / M**4 
    else:
        h0 = (h1s0 * m1 ** 4 + h2s0 * m2 ** 4) / M**4
    
    
    ######################### ----- Non spinning part of the waveform ----- #########################
    

    ############ Tidal Love Numbers (Schwarzschild) ############
    
    psi_TLN_5PN = - (39.0 * l_tilde / 2.0) * v10
    
    ############ Tidal Dissipation Numbers (Schwarzschild) ############
    
    psi_TDN_NS_4PN = 25.0 / 2.0 * h0 * v8 * (1.0 - 3.0 * np.log(v / vlso))
    
    
    ######################### ----- (Aligned) Spin part of the waveform ----- #########################        

    ############ Background GR point particle ############ 
    

    # 2PN SS
    psi_S_2PN = (-(5.0 / 8.0)
                 * (+ 80.0 * delta * k_a + 80.0 * (1.0 - 2.0 * eta) * k_s)
                 * (chi_s**2))
    psi_S_2PN -= ((5.0 / 8.0)
                  * (+ 80.0 * delta * k_a + 80.0 * (1.0 - 2.0 * eta) * k_s)
                  * (chi_a**2))
    psi_S_2PN -= ((5.0 / 4.0)
                   * (+ 80.0 * delta * k_s + 80.0 * (1.0 - 2.0 * eta) * k_a)
                   * chi_s * chi_a)
    psi_S_2PN *= v4
    
    
    # 3PN SS
    psi_S_3PN = ((26015.0 / 14.0 - 88510.0 * eta / 21.0 - 480.0 * eta2) * k_a
                  + delta * (-1344475.0 / 1008.0 + 745.0 * eta / 18.0 + (26015.0 / 14.0 - 1495.0 * eta / 3.0) * k_s)
                  ) * chi_s * chi_a
    psi_S_3PN += (+ (26015.0 / 28.0 - 44255.0 * eta / 21.0 - 240.0 * eta2) * k_s
                  + delta * (26015.0 / 28.0 - 1495.0 * eta / 6.0) * k_a
                  ) * chi_s ** 2
    psi_S_3PN += (+ (26015.0 / 28.0 - 44255.0 * eta / 21.0 - 240.0 * eta2) * k_s
                  + delta * (26015.0 / 28.0 - 1495.0 * eta / 6.0) * k_a
                  ) * chi_a ** 2
    psi_S_3PN *= v6
    
    
    
    # 3.5PN SS
    psi_S_35PN =(- 400.0 * PI * delta * k_a - 400.0 * PI * k_s
       + eta * (+ 800.0 * PI * k_s)
    ) * (chi_a) ** 2
    psi_S_35PN +=(- 800.0 * PI * k_a + 1600.0 * PI * eta * k_a - 800.0 * PI * delta * k_s
    ) * (chi_a * chi_s)
    psi_S_35PN +=( - 400.0 * PI * delta * k_a - 400.0 * PI * k_s
        + eta * (+ 800.0 * PI * k_s)
    ) * (chi_s) ** 2
    
    
    # 3.5PN SSS
    psi_S_35PN += (+ (3110.0 / 3.0 - 10250.0 * eta / 3.0 + 40.0 * eta2) * k_s
                   - 440.0 * (1.0 - 3.0 * eta) * lambda_s
                   + delta
                   * ((3110.0 / 3.0 - 4030.0 * eta / 3.0) * k_a - 440.0 * (1.0 - eta) * lambda_a)
                   ) * chi_s ** 3
    psi_S_35PN += ((3110.0 / 3.0 - 8470.0 * eta / 3.0) * k_a
                   - 440.0 * (1.0 - 3.0 * eta) * lambda_a
                   + delta
                   * (+ (3110.0 / 3.0 - 750.0 * eta) * k_s
                       - 440.0 * (1 - eta) * lambda_s)
                    ) * chi_a ** 3
    psi_S_35PN += ((3110.0 - 28970.0 * eta / 3.0 + 80.0 * eta2) * k_a
                    - 1320.0 * (1.0 - 3.0 * eta) * lambda_a
                    + delta
                    * (+ (3110.0 - 10310.0 * eta / 3.0) * k_s
                        - 1320.0 * (1.0 - eta) * lambda_s)
                   ) * (chi_s**2 * chi_a)
    psi_S_35PN += (+ (3110.0 - 27190.0 * eta / 3.0 + 40.0 * eta2) * k_s
                   - 1320.0 * (1.0 - 3 * eta) * lambda_s
                   + delta
                   * ((3110.0 - 8530.0 * eta / 3.0) * k_a - 1320.0 * (1.0 - eta) * lambda_a)
                   ) * (chi_a**2 * chi_s)
    psi_S_35PN *= v7
    
    
    ############ Tidal Dissipation Numbers (Kerr) ############
    # The following assumes E/B duality for Kerr

    # 2.5PN linear-in-spin dissipation
    psi_TDN_S_25PN = (25.0*hs*chi_s/4.0 + 25.0*ha* chi_a/4.0) * v5 * (1 + 3.0 * np.log(v / vlso))
    
    # 3.5PN linear- and cubic-in-spin dissipation 
    psi_TDN_S_35PN = (115575.0 / 448.0 * ha + 1425.0 / 16 * eta * ha + 675.0 / 32.0 * delta * hs
                      ) * chi_a
    psi_TDN_S_35PN += (115575.0 / 448.0 * hs + 1425.0 / 16 * eta * hs + 675.0 / 32.0 * delta * ha
                       ) * chi_s 
    psi_TDN_S_35PN += (225.0 / 8.0 * ha3) * chi_a**3 
    psi_TDN_S_35PN += (225.0 / 8.0 * hs3) * chi_s**3
    psi_TDN_S_35PN += (675.0 / 8.0 * ha3) * chi_a * chi_s**2 
    psi_TDN_S_35PN += (675.0 / 8.0 * hs3) * chi_a**2 * chi_s 
    psi_TDN_S_35PN *= v7
    
    ######################### ----- Summing up all phases ----- #########################   
    
    # Point particle + spin-induced moments
    psi_S = psi_S_2PN + psi_S_3PN + psi_S_35PN
    
    # Tidal effects
    psi_TDN = psi_TDN_S_25PN + psi_TDN_S_35PN + psi_TDN_NS_4PN
    psi_TLN = psi_TLN_5PN
    
    
    return 3.0 / 128.0 / eta / v5 * (psi_S + psi_TDN + psi_TLN)



def compute_f22_peak(M, eta, chi_EOB):
    """
    Computes the cut in frequency domain for the inspiral-only waveform, which is also named as "tape frequency" in the test GR paper: 2203.13937.
    f_{22}^{tape} = \alpha * f_{22}^{peak}. f_{22}^{peak} is given in Eq.(A8) of 1611.03703  
    
    Returns:
    f_cut (number): Frequency domain cut
    """
    p0TPL = 0.562679
    p1TPL = -0.087062
    p2TPL = 0.001743
    p3TPL = 25.850378
    p3EQ = 10.262073
    p4TPL = 25.819795
    p4EQ = 7.629922
    
    A3 = p3EQ + 4.0 * (p3EQ - p3TPL) * (eta - 1.0/4.0)
    A4 = p4EQ + 4.0 * (p4EQ - p4TPL) * (eta - 1.0/4.0)
    
    
    f22_peak = (p0TPL + (p1TPL + p2TPL * chi_EOB) * np.log(A3 - A4 * chi_EOB)) / (G * MSUN / C**3) / (2.0 * PI) / M
    
    return f22_peak




def gen_h0_qdol_phase(f, theta_more, f_ref):
    """
    
    Computes the TaylorF2 waveform + finite-size effects. 
    
    Parameters
    -----------
    f (array): Frequency at which to output the phase [Hz]
    theta_more: Intrinsic and extrinsic parameters (excluding inclination, iota). 
                Intrinsic parameters are the same as theta in phase_qdol
    f_ref: Reference frequency
                
    Deff: Effective luminosity distance [Mpc]
    tc: coalescence time
    phic: coalescence phase

    Returns
    -------
    Strain (array): GW frequency-domain strain as a function of frequency
    
    """
    
    # theta_more ordering: Mc, eta, chi_1, chi_2, kappa1, kappa2, h1s1, h2s2, 
    # lambda1, lambda2, h1s3, h2s3, h1s0, h2s0, l1, l2, Deff, tc, phic
    Mc, eta, chi_1, chi_2, _, _, _, _, _, _, _, _, _, _, _, _, Deff, tc, phic = theta_more
    M = Mc / (eta ** (3 / 5))
    pre = 3.6686934875530996e-19  # (GN*Msun/c^3)^(5/6)/Hz^(7/6)*c/Mpc/sec
    chi_s = 0.5 * (chi_1 + chi_2)
    chi_a = 0.5 * (chi_1 - chi_2)
    delta = np.sqrt(1.0 - 4.0 * eta)
    chi_EOB = chi_s + chi_a * delta / (1.0 - 2.0 * eta)   
    Mchirp = M * eta**0.6
    alpha = 0.35

    
    TDN_Phi = phase_qdol(f, theta_more[:-3])
    
    f_tape = compute_f22_peak(M, eta, chi_EOB) * alpha
    TDN_Phi_tape = TDN_Phi[f>f_tape][0]
    TDN_Phi[f>f_tape] = TDN_Phi_tape
    

    
    TDN_Phi_ref = phase_qdol(f_ref, theta_more[:-3])
    TDN_Phi -=  TDN_Phi_ref


    return TDN_Phi
