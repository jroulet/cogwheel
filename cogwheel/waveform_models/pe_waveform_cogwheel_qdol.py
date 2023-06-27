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

# waveform.APPROXIMANTS['TaylorF2Love'] = Approximant(tides=True)
# waveform.APPROXIMANTS['TaylorF2Heat'] = Approximant(tides=False) # TODO: tides=True?

# TODO: Interface this script with loveNum_waveform_merger (and Ripple's JAX version?)

###################################################################################
#################### Tidal waveform model params and functions ####################
###################################################################################


###### Wf model params, copied from ripple.constants.py and ripple.typing.py ######

"""
Various constants, all in SI units.
"""

EulerGamma = 0.577215664901532860606512090082402431

MSUN = 1.988409902147041637325262574352366540e30  # kg
"""Solar mass"""

G = 6.67430e-11  # m^3 / kg / s^2
"""Newton's gravitational constant"""

C = 299792458.0  # m / s
"""Speed of light"""

"""Pi"""
PI = 3.141592653589793238462643383279502884

gt = G * MSUN / (C ** 3.0)
"""
G MSUN / C^3 in seconds
"""

m_per_Mpc = 3.085677581491367278913937957796471611e22
"""
Meters per Mpc.
"""

# theta ordering: Mc, eta, chi_1, chi_2, kappa1, kappa2, h1s1, h2s2, lambda1, lambda2, h1s3, h2s3, h1s0,h2s0, l1, l2, dL, tc, phic, iota

###################################################################################
###################################################################################

def Phif3hPN_TLN(f, theta, fix_bh_superradiance=True):
    """
    
    Computes the phase of the TaylorF2 waveform + finite-size effect parameters. 
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
    vlso = 1.0 # Doesn't matter for 2.5PN and 4PN constant log(v/vlso) terms, but matters for 4PN log(v/vlso)**2
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

    # # These are black hole finite-size values
    # kappa1 = 1.0
    # kappa2 = 1.0
    # h1s1 = 1 + 3*chi_1**2 ## TODO_HS: -8/45 Factor?
    # h2s2 = 1 + 3*chi_2**2 ## TODO_HS: -8/45 Factor?
    # lambda1 = 1.0
    # lambda2 = 1.0
    # h1s3 = 2/3
    # h2s3 = 2/3
    # h1s0 = TODO_HS: Add
    # h2s0 = TODO_HS: Add
    # l1 = 0
    # l2 = 0

    ######## Symmetric/antisymmetric spins and spin-induced multipoles ########
    chi_s = 0.5 * (chi_1 + chi_2)
    chi_a = 0.5 * (chi_1 - chi_2)
    k_s = 0.5 * (kappa1 + kappa2)
    k_a = 0.5 * (kappa1 - kappa2)
    lambda_s = 0.5 * (lambda1 + lambda2)
    lambda_a = 0.5 * (lambda1 - lambda2)
        
    ######## Effective dissipation number parameters ########
    # 2.5PN Dissipation (Linear spin)
    hs = (h1s1*m1**3 + h2s1*m2**3) / M**3
    ha = (h1s1*m1**3 - h2s1*m2**3) / M**3
    
    # 3.5PN Dissipation (Cubic spin)
    hs3 = (h1s3*m1**3 + h2s3*m2**3) / M**3
    ha3 = (h1s3*m1**3 - h2s3*m2**3) / M**3
    
    # Intermediate 3.5PN Dissipation Terms (not independent variables)
    hts = (h1s1*m1**4 + h2s1*m2**4) / M**4
    hta = (h1s1*m1**4 - h2s1*m2**4) / M**4
    hps = (h1s1*m1**3 * m2**2 + h2s1*m2**3 * m1**2) / M**5
    hpa = (h1s1*m1**3 * m2**2 - h2s1*m2**3 * m1**2) / M**5
    
    # 4PN Dissipation (LO non-spinning) 
    # Here one could impose the superradiance condition (m Omega_H - omega), which is valid at LO in spin. 
    # Good approximation if the component spins are small.
    if fix_bh_superradiance:
        # factor of -2 strictly only applies for Schw black holes
        ht0 = (-2.0 * h1s1 * m1 ** 4 - 2.0 * h2s1 * m2 ** 4) / M**4 
    else:
        ht0 = (h1s0 * m1 ** 4 + h2s0 * m2 ** 4) / M**4
    
    ######## Effective Love number parameters ########
    l_tilde = 16.0 * ((m1 + 12 * m2) * m1**4 * l1 + (m2 + 12 * m1) * m2**4 * l2) / (13.0 * M**5.0)
    # TODO_HS: Add delta_l_tilde definition
    
    
    
    ######################### ----- Non spinning part of the waveform ----- #########################
    
    ############ ---- Background GR point particle ---- ############
    
    psi_NS_0PN = 1.0
    psi_NS_1PN = (3715.0 / 756.0 + 55.0 * eta / 9.0) * v2
    psi_NS_15PN = - 16.0 * PI * v3
    psi_NS_2PN = (15293365.0 / 508032.0 + 27145.0 * eta / 504.0 + 3085.0 * eta2 / 72.0) * v4
    psi_NS_25PN = PI * (38645.0 / 756.0 - 65.0 * eta / 9.0) * (1 + 3.0 * np.log(v / vlso)) * v5
    
    psi_NS_3PN = (11583231236531.0 / 4694215680.0
                  - 640.0 * PI**2 / 3.0
                  - 6848.0 * EulerGamma / 21.0
                  + (2255.0 * PI**2 / 12.0 - 15737765635.0 / 3048192.0) * eta
                  + 76055.0 / 1728.0 * eta2 
                  - 127825.0 / 1296.0 * eta3 
                  - 6848.0 * np.log(4.0 * v) / 21.0
                  ) * v6
    
    psi_NS_35PN = PI * (77096675.0 / 254016.0 + 378515.0 * eta / 1512.0 - 74045.0 * eta2 / 756.0) * v7
        
    psi_NS_4PN = (- 90490.0 * PI**2 / 567.0 
                  - 36812.0 * EulerGamma / 189.0 
                  + 2550713843998885153.0 / 830425530654720.0 
                  - 26325.0 / 196.0 * np.log(3) 
                  - 1011020.0 / 3969.0 * np.log(2) 
                  + (- 680712846248317.0 / 126743823360.0 
                     - 3911888.0 * EulerGamma / 3969.0 
                     + 109295.0 * PI**2 / 672.0 
                     - 9964112.0 / 3969.0 * np.log(2) 
                     + 26325.0 / 49.0 * np.log(3)
                    ) * eta
                  + (7510073635.0 / 9144576.0 - 11275.0 / 432.0 * PI**2) * eta2
                  + 1292395.0 / 36288.0 * eta3 
                  - 5975.0 / 288.0 * eta4 
                  ) * v8 * (1.0 - 3.0 * np.log(v / vlso))
    
    psi_NS_4PN_log2 = (18406.0 / 63.0  + 1955944.0 / 1323.0 * eta) * v8 * np.log(v / vlso)**2

    ############ ---- Tidal Love Numbers (Schwarzschild) ---- ############
    
    psi_TLN_5PN = - (39.0 * l_tilde / 2.0) * v10
    # psi_TLN_6PN = (-3115.0 * l_tilde / 64.0 + 6595.0 * delta_l_tilde / 364.0) * v12
    
    ############ ---- Tidal Dissipation Numbers (Schwarzschild) ---- ############
    
    ## TODO_HS: Double check
    psi_diss_NS_4PN = 25.0 / 2.0 * ht0 * v8 * (1.0 - 3.0 * np.log(v / vlso))
    
    
    
    ######################### ----- (Aligned) Spin part of the waveform ----- #########################        

    ############ ---- Background GR point particle ---- ############ 
    
    # 1.5PN SO
    psi_S_15PN = ((113.0 / 3.0 - 76.0 * eta / 3.0) * chi_s + 113.0 * delta * chi_a / 3.0) * v3

    # 2PN SS
    psi_S_2PN = (-(5.0 / 8.0)
                 * (1.0 + 156.0 * eta + 80.0 * delta * k_a + 80.0 * (1.0 - 2.0 * eta) * k_s)
                 * (chi_s**2))
    psi_S_2PN -= ((5.0 / 8.0)
                  * (1.0 - 160.0 * eta + 80.0 * delta * k_a + 80.0 * (1.0 - 2.0 * eta) * k_s)
                  * (chi_a**2))
    psi_S_2PN -= ((5.0 / 4.0)
                   * (delta + 80.0 * delta * k_s + 80.0 * (1.0 - 2.0 * eta) * k_a)
                   * chi_s * chi_a)
    psi_S_2PN *= v4
    
    # 2.5PN SO
    psi_S_25PN = (-(732985.0 / 2268.0 - 24260.0 * eta / 81.0 - 340.0 * eta2 / 9.0) * chi_s
                  - (732985.0 / 2268.0 + 140.0 * eta / 9.0) * delta * chi_a
                  ) * v5 * (1 + 3.0 * np.log(v / vlso))
    
    # 3PN SO + SS
    psi_S_3PN = (2270.0 / 3.0 - 520.0 * eta) * PI * chi_s + (2270.0 * PI / 3.0) * delta * chi_a
    psi_S_3PN += ((26015.0 / 14.0 - 88510.0 * eta / 21.0 - 480.0 * eta2) * k_a
                  + delta * (-1344475.0 / 1008.0 + 745.0 * eta / 18.0 + (26015.0 / 14.0 - 1495.0 * eta / 3.0) * k_s)
                  ) * chi_s * chi_a
    psi_S_3PN += (-1344475.0 / 2016.0
                  + 829705.0 * eta / 504.0
                  + 3415.0 * eta2 / 9.0
                  + (26015.0 / 28.0 - 44255.0 * eta / 21.0 - 240.0 * eta2) * k_s
                  + delta * (26015.0 / 28.0 - 1495.0 * eta / 6.0) * k_a
                  ) * chi_s ** 2
    psi_S_3PN += (-1344475.0 / 2016.0
                  + 267815.0 * eta / 252.0
                  - 240.0 * eta2
                  + (26015.0 / 28.0 - 44255.0 * eta / 21.0 - 240.0 * eta2) * k_s
                  + delta * (26015.0 / 28.0 - 1495.0 * eta / 6.0) * k_a
                  ) * chi_a ** 2
    psi_S_3PN *= v6
    
    # 3.5PN SO
    psi_S_35PN = (- 25150083775.0 / 3048192.0
                  + 10566655595.0 * eta / 762048.0
                  - 1042165 * eta2 / 3024.0
                  + 5345.0 * eta3 / 36.0
                  ) * chi_s
    psi_S_35PN += ((-25150083775.0 / 3048192.0 + 26804935.0 * eta / 6048.0 - 1985.0 * eta2 / 48.0)
                    * delta * chi_a)
    
    # 3.5PN SS ## TODO_HS: Check
    psi_S_35PN +=(
       - 15.0 * PI / 2.0 - 400.0 * PI * delta * k_a - 400.0 * PI * k_s
       + eta * (800.0 * PI + 800.0 * PI * k_s)
    ) * (chi_a) ** 2
    psi_S_35PN +=(
        -15.0 * PI * delta - 800.0 * PI * k_a + 1600.0 * PI * eta * k_a - 800.0 * PI * delta * k_s
    ) * (chi_a * chi_s)
    psi_S_35PN +=(
        - 15.0 * PI / 2.0 - 400.0 * PI * delta * k_a - 400.0 * PI * k_s
        + eta * (- 770.0 * PI + 800.0 * PI * k_s)
    ) * (chi_s) ** 2
    
    
    # 3.5PN SSS
    psi_S_35PN += (265.0 / 24.0
                   + 4035.0 * eta / 2.0
                   - 20.0 * eta2 / 3.0
                   + (3110.0 / 3.0 - 10250.0 * eta / 3.0 + 40.0 * eta2) * k_s
                   - 440.0 * (1.0 - 3.0 * eta) * lambda_s
                   + delta
                   * ((3110.0 / 3.0 - 4030.0 * eta / 3.0) * k_a - 440.0 * (1.0 - eta) * lambda_a)
                   ) * chi_s ** 3
    psi_S_35PN += ((3110.0 / 3.0 - 8470.0 * eta / 3.0) * k_a
                   - 440.0 * (1.0 - 3.0 * eta) * lambda_a
                   + delta
                   * (265.0 / 24.0
                       - 2070.0 * eta
                       + (3110.0 / 3.0 - 750.0 * eta) * k_s
                       - 440.0 * (1 - eta) * lambda_s)
                    ) * chi_a ** 3
    psi_S_35PN += ((3110.0 - 28970.0 * eta / 3.0 + 80.0 * eta2) * k_a
                    - 1320.0 * (1.0 - 3.0 * eta) * lambda_a
                    + delta
                    * (265.0 / 8.0
                        + 12055.0 * eta / 6.0
                        + (3110.0 - 10310.0 * eta / 3.0) * k_s
                        - 1320.0 * (1.0 - eta) * lambda_s)
                   ) * (chi_s**2 * chi_a)
    psi_S_35PN += (265.0 / 8.0
                   - 6500.0 * eta / 3.0
                   + 40.0 * eta2
                   + (3110.0 - 27190.0 * eta / 3.0 + 40.0 * eta2) * k_s
                   - 1320.0 * (1.0 - 3 * eta) * lambda_s
                   + delta
                   * ((3110.0 - 8530.0 * eta / 3.0) * k_a - 1320.0 * (1.0 - eta) * lambda_a)
                   ) * (chi_a**2 * chi_s)
    psi_S_35PN *= v7
    
    
    

    ############ ---- Tidal Dissipation Numbers (Kerr) ---- ############

    # 2.5PN linear-in-spin dissipation
    # TODO_HS: Check
    psi_diss_S_25PN = (25.0*hs*chi_s/4.0 + 25.0*ha* chi_a/4.0) * v5 * (1 + 3.0 * np.log(v / vlso))
    
    # 3.5PN linear- and cubic-in-spin dissipation
    # TODO_HS: Check
    psi_diss_S_35PN = (
        112425.0 / 448.0 * ha - 225.0 / 16.0 * hpa + 225.0 / 8.0 * hta + 75 * ha * eta
    ) * (chi_a)
    psi_diss_S_35PN += (
        225.0 / 8.0 * ha3
    ) * (chi_a) ** 3 
    psi_diss_S_35PN += (
        112425.0 / 448.0 * hs - 225.0 / 16.0 * hps + 225.0 / 8.0 * hts + 75 * hs * eta   
    ) * (chi_s) 
    psi_diss_S_35PN += (
        225.0 / 8.0 * hs3   
    ) * (chi_s) ** 3
    psi_diss_S_35PN *= v7
    
    
    
    ######################### ----- Summing up all phases ----- #########################   
    
    # Point particle + spin-induced moments
    psi_NS = (psi_NS_0PN + psi_NS_1PN + psi_NS_15PN + psi_NS_2PN + psi_NS_25PN + psi_NS_3PN
              + psi_NS_35PN + psi_NS_4PN + psi_NS_4PN_log2)
    # psi_S_25PN_log removed from original code as it's now included in psi_S_25PN 
    psi_S = psi_S_15PN + psi_S_2PN + psi_S_25PN + psi_S_3PN + psi_S_35PN
    
    # Tidal effects
    psi_diss = psi_diss_S_25PN + psi_diss_S_35PN + psi_diss_NS_4PN
    psi_TLN = psi_TLN_5PN  # + psi_TLN_6PN

    # TODO_HS: Note that when called with hf3hPN_TLN, we need to include - sign for correct time domain direction
    return 3.0 / 128.0 / eta / v5 * (psi_NS + psi_S + psi_diss + psi_TLN )
      


def Amp_merger(f, f_cutoff, Amp_Ins_end):
        
    # Amp_m = Amp_Ins_end * (1 - (1 / (1 + np.exp(-(f - 1.2 * f_cutoff)))))
    Amp_m = (
        (1.0 - (1.0 / (1.0 + np.exp(-(f - 1.2 * f_cutoff)))))
        * (Amp_Ins_end / f_cutoff ** (-7.0 / 6.0))
        * (f ** (-7.0 / 6.0))
    )

    return Amp_m


def gen_h0(f, theta, f_ref):
    """
    Computes the Taylor F2 Frequency domain strain waveform with non-standard spin induced quadrupole moment/tidal deformability for object two
    Note that this waveform assumes object 1 is a BH therefore uses the chi*M relation to find C
    Note that this waveform also assumes that object one is the more massive. Therefore the more massive object is always considered a BH
    Parameters:
    f (array): Frequency at which to output the phase [Hz]
    Mc (float): Chirp Mass [Msun]
    eta (float): m1*m2/(m1+m2)**2
    s1z (float): Spin of object 1 in z direction
    s2z (float): Spin of object 2 in z direction
    Deff (float): Distance [Mpc]

    Returns:
    Strain (array):
    """
    
    
    Mc, eta, _, _, _, _, _, _, _, _, _, _, _, _, _, _, Deff, tc, phic = theta #chi_1, chi_2, kappa1, kappa2, h1s1, h2s1, lambda1,     lambda2, h1s3, h2s3, h1s0, h2s0, l1, l2
    M = Mc / (eta ** (3 / 5))
    pre = 3.6686934875530996e-19  # (GN*Msun/c^3)^(5/6)/Hz^(7/6)*c/Mpc/sec
    Mchirp = M * eta**0.6

    A0 = (
        Mchirp ** (5.0 / 6.0)
        / (f + 1e-30) ** (7.0 / 6.0)
        / Deff
        / PI ** (2.0 / 3.0)
        * np.sqrt(5.0 / 24.0)
    )
    
    Phi = Phif3hPN_TLN(f, theta[:-3])
    grad_phase = np.gradient(Phi, f)
    f_phasecutoff = f[np.argmax(grad_phase)]
    f_ISCO = 4.4e3 * (1 / M)  # Hz
    f_cutoff = np.min(np.array([f_ISCO, f_phasecutoff]))

    
    # TODO: Might have a tiny bug here. Need to double check and merge with loveNum_waveform_merger
    fc_ind = np.argmin(np.absolute(f-f_cutoff))
    t0 = grad_phase[fc_ind]
    
    
    # Lets call the amplitude and phase now
    Phi_ref = Phif3hPN_TLN(f_ref, theta[:-3])
    Phi -= t0 * (f - f_ref) + Phi_ref

    ext_phase_contrib = 2.0 * PI * f * tc - 2 * phic
    Phi += ext_phase_contrib
    Phi = Phi * np.heaviside(f_cutoff - f, 1.0)
    
    
    Amp_m = Amp_merger(f, f_cutoff, A0[fc_ind])  
    Amp = A0 * np.heaviside(f_cutoff - f, 0.5) + Amp_m * np.heaviside(
        f - f_cutoff, 0.5
    )
    
    return Amp * pre * np.exp(-1.0j * Phi) + 1e-30 # Add tiny value to avoid zero division in finding reference wf

def gen_taylorF2_qdol_polar(f, params, f_ref):
    """
    Generate PhenomD frequency domain waveform following 1508.07253.
    vars array contains both intrinsic and extrinsic variables
    theta = [Mchirp, eta, chi1, chi2, D, tc, phic]
    Mchirp: Chirp mass of the system [solar masses]
    eta: Symmetric mass ratio [between 0.0 and 0.25]
    chi1: Dimensionless aligned spin of the primary object [between -1 and 1]
    chi2: Dimensionless aligned spin of the secondary object [between -1 and 1]
    L1:
    L2:
    D: Luminosity distance to source [Mpc]
    tc: Time of coalesence. This only appears as an overall linear in f contribution to the phase
    phic: Phase of coalesence
    inclination: Inclination angle of the binary [between 0 and PI]

    f_ref: Reference frequency for the waveform

    Returns:
    --------
      hp (array): Strain of the plus polarization
      hc (array): Strain of the cross polarization
    """
    iota = params[-1]
    h0 = gen_h0(f, params[:-1], f_ref)

    hp = h0 * (1 / 2 * (1 + np.cos(iota) ** 2))
    hc = -1j * h0 * np.cos(iota)

    return hp, hc



##
################################################################################
######################## Function to be called by cogwheel ########################
###################################################################################

def custom_compute_hplus_hcross(f, par_dic, *_):
    """
	Function to replace ``waveform.compute_hplus_hcross`` with,
	use with caution.

    Return hplus, hcross evaluated at f.

    Parameters
    ----------
    f: 1d array of type float
        Frequency array in Hz

    par_dic: dict
        Source parameters. Needs to have these keys:
            * m1, m2: component masses (Msun)
            * d_luminosity: luminosity distance (Mpc)
            * iota: inclination (rad)
            * phi_ref: phase at reference frequency (rad)
            * f_ref: reference frequency (Hz)
        plus, optionally:
            * s1x_n, s1y_n, s1z, s2x_n, s2y_n, s2z: dimensionless spins
            * h1, h2: dimensionless dissipation constants

    Return
    ------
    hplus_hcross: complex array of shape ``(2, len(f))``.
    """

    # input_params ordering (lal convention): ['m1', 'm2', 's1z', 's2z', 'l1', 'l2', 'd_luminosity', 'phi_ref', 'iota', 'f_ref']
    # wf_params ordering: ['Mc', 'eta', 'chi_1', 'chi_2', 'kappa1', 'kappa2', 'h1s1', 'h2s1', 'lambda1', 'lambda2', 'h1s3','h2s3', 'h1s0', 'h2s0', 'l1', 'l2', 'Deff', 'tc', 'phic', 'iota', 'f_ref']
    
    eta = par_dic['m1']*par_dic['m2'] / (par_dic['m1']+par_dic['m2'])**2
    Mc = (par_dic['m1']+par_dic['m2'])*eta**(3/5)
    
    wf_params = np.array([Mc, eta, par_dic['s1z'], par_dic['s2z'], par_dic['kappa1'], par_dic['kappa2'], 
                          par_dic['h1s1'], par_dic['h2s1'], par_dic['lambda1'], par_dic['lambda2'],
                          par_dic['h1s3'], par_dic['h2s3'], par_dic['h1s0'], par_dic['h2s0'],par_dic['l1'], par_dic['l2'],
                          par_dic['d_luminosity'], 0, par_dic['phi_ref'], par_dic['iota']]) # tc in gen_taylorF2_qdol_polar set to zero
    f_ref = par_dic['f_ref'] 
    
    # Avoid divergence in amplitude at f=0
    f0_is_0 = f[0] == 0 
    if f0_is_0:
        f[0] = 0.0001 # Prevent divergence in amplitude
    
    hplus, hcross = gen_taylorF2_qdol_polar(f, wf_params, f_ref)
    hplus, hcross = np.array(hplus), np.array(hcross)
    hplus_hcross = np.stack([hplus, hcross])
    
    return hplus_hcross