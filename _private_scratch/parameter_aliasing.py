import numpy as np
from copy import deepcopy as dcopy


#     Parameter Aliases
# ----------------------------

MASS_PARKEYS = {
    'mchirp': ['mchirp', 'mc', 'mchirp_det', 'mc_det', 'chirp_mass', 'detector_chirp_mass'],
    'mtot': ['mtot', 'mt', 'mtot_det', 'mt_det', 'total_mass', 'detector_total_mass'],
    'q': ['q', 'mass_ratio'],
    'q1': ['q1', 'inverse_mass_ratio', 'qinv', 'q_inv', 'q_inverse', 'inverted_mass_ratio'],
    'lnq': ['lnq', 'logq', 'log_mass_ratio', 'log_q', 'ln_q'],
    'eta': ['eta', 'symmetric_mass_ratio'],
    'm1': ['m1', 'mass1', 'mass_1', 'm1_det', 'mass1_det', 'mass_1_det', 'detector_mass_1', 'detector_m1',
           'detector_mass1'],
    'm2': ['m2', 'mass2', 'mass_2', 'm2_det', 'mass2_det', 'mass_2_det', 'detector_mass_2', 'detector_m2',
           'detector_mass2'],
    'mchirp_source': ['mchirp_source', 'mchirp_src', 'source_mchirp', 'src_mchirp', 'mc_source', 'mc_src', 'source_mc',
                      'src_mc', 'source_chirp_mass', 'chirp_mass_src', 'chirp_mass_source'],
    'mtot_source': ['mtot_source', 'mtot_src', 'source_mtot', 'src_mtot', 'mt_source', 'mt_src', 'source_mt',
                    'src_mt', 'source_total_mass', 'total_mass_src', 'total_mass_source'],
    'm1_source': ['m1_source', 'm1_src', 'source_mass_1', 'source_mass1', 'source_m1', 'm1src', 'mass1src', 'srcmass1',
                  'mass1_src', 'mass1_source', 'mass_1_src', 'mass_1_source'],
    'm2_source': ['m2_source', 'm2_src', 'source_mass_2', 'source_mass2', 'source_m2', 'm2src', 'mass2src', 'srcmass2',
                  'mass2_src', 'mass2_source', 'mass_2_src', 'mass_2_source']
}

# NOTE: chip and chi_p separated because LVC chi_p is NOT same as IAS chip (closer to IAS chi_perp, but not exact?)
SPIN_PARKEYS = {
    's1x': ['s1x', 'spin1x', 'spin_1x', 'spin_1_x', 's1_x'],
    's1y': ['s1y', 'spin1y', 'spin_1y', 'spin_1_y', 's1_y'],
    's1z': ['s1z', 'spin1z', 'spin_1z', 'spin_1_z', 's1_z'],
    's2x': ['s2x', 'spin2x', 'spin_2x', 'spin_2_x', 's2_x'],
    's2y': ['s2y', 'spin2y', 'spin_2y', 'spin_2_y', 's2_y'],
    's2z': ['s2z', 'spin2z', 'spin_2z', 'spin_2_z', 's2_z'],
    'chieff': ['chieff', 'chi_eff', 'chi_effective', 'effective_spin'],
    'chiperp': ['chiperp', 'chi_perp'],
    'chip': ['chip', 'chi_p'], 'chia': ['chia'],
    's1': ['s1', 'spin1', 'spin_1', 's1a', 'spin1a', 'spin_1a', 'a_1'],
    's2': ['s2', 'spin2', 'spin_2', 's2a', 'spin2a', 'spin_2a', 'a_2'],
    's1theta': ['s1theta', 's1t', 'spin_1_tilt', 'tilt_1', 'spin1_tilt', 'tilt1', 'spin_1_theta', 'theta1',
                'spin1_theta', 'theta_s1', 'theta_spin1', 's1tilt', 'spin1tilt', 'tilt_s1', 'tilt_spin1'],
    's2theta': ['s2theta', 's2t', 'spin_2_tilt', 'tilt_2', 'spin2_tilt', 'tilt2', 'spin_2_theta', 'theta2',
                'spin2_theta', 'theta_s2', 'theta_spin2', 's2tilt', 'spin2tilt', 'tilt_s2', 'tilt_spin2'],
    's1costheta': ['s1costheta', 's1costheta', 's1costilt', 'spin_1_costheta', 'spin_1_costilt', 'costilt1',
                   'spin1_costilt', 'costilt1', 'costheta1', 'spin1_costheta', 'costheta_s1', 'costheta_spin1',
                   'spin1costilt', 'spin1costheta'],
    's2costheta': ['s2costheta', 's2costheta', 's2costilt', 'spin_2_costheta', 'spin_2_costilt', 'costilt2',
                   'spin2_costilt', 'costilt2', 'costheta2', 'spin2_costheta', 'costheta_s2', 'costheta_spin2',
                   'spin2costilt', 'spin2costheta'],
    's1phi': ['s1phi', 'spin1phi', 's1_phi', 'spin1_phi', 'spin_1_phi', 'phi1', 'phi_1', 'phi_s1', 'phi_spin1'],
    's2phi': ['s2phi', 'spin2phi', 's2_phi', 'spin2_phi', 'spin_2_phi', 'phi2', 'phi_2', 'phi_s2', 'phi_spin2'],
    'cumchidiff': ['cumchidiff'],
    'cums1r_s1z': ['cums1r_s1z'], 'cums2r_s2z': ['cums2r_s2z'], 'cums1z': ['cums1z'], 'cums2z': ['cums2z'],
    'phi12': ['phi12', 'phi_12'], 'phiJL': ['phiJL', 'phi_jl'],
    's1r': ['s1r', 'chi_1_in_plane'], 's2r': ['s2r', 'chi_2_in_plane'],
    's1phi_plus_vphi': ['s1phi_plus_vphi', 's1phiplusvphi'],
    's2phi_plus_vphi': ['s2phi_plus_vphi', 's2phiplusvphi'],
    's1phi_hat': ['s1phi_hat', 's1phihat'], 's2phi_hat': ['s2phi_hat', 's2phihat'],
    's1x_prime': ['s1x_prime', 's1xprime'], 's1y_prime': ['s1y_prime', 's1yprime'],
    's2x_prime': ['s2x_prime', 's2xprime'], 's2y_prime': ['s2y_prime', 's2yprime'],
    's1x_prime_rescale': ['s1x_prime_rescale', 's1xprimerescale'],
    's1y_prime_rescale': ['s1y_prime_rescale', 's1yprimerescale'],
    's2x_prime_rescale': ['s2x_prime_rescale', 's2xprimerescale'],
    's2y_prime_rescale': ['s2y_prime_rescale', 's2yprimerescale'],
    's1x_newsign': ['s1x_newsign', 's1xprime'], 's1y_newsign': ['s1y_newsign', 's1ynewsign'],
    's2x_newsign': ['s2x_newsign', 's2xprime'], 's2y_newsign': ['s2y_newsign', 's2ynewsign'],
}

# combine intrinsic parameters: masses, spins, tidal deformabilities, and eccentricity
INTRINSIC_PARKEYS = dcopy(MASS_PARKEYS)
INTRINSIC_PARKEYS.update(SPIN_PARKEYS)
INTRINSIC_PARKEYS.update({'l1': ['l1', 'l_1', 'lambda1', 'lambda_1', 'tidal_deformability_1'],
                          'l2': ['l2', 'l_2', 'lambda2', 'lambda_2', 'tidal_deformability_2'],
                          'e': ['e', 'ecc', 'eccentricity']})

DISTANCE_PARKEYS = {
    'd_luminosity': ['d_luminosity', 'DL', 'dL', 'DL_Mpc', 'dL_Mpc',
                     'luminosity_distance', 'luminosity_dist', 'distance'],
    'Dcomov': ['Dcomov', 'Dcom', 'Dcomoving', 'Dcomov_Mpc', 'comoving_distance', 'comoving_dist'],
    'Vcomov': ['Vcomov', 'Vcom', 'Vcomoving', 'Vcomov_Mpc3', 'comoving_volume', 'comoving_vol'],
    'z': ['z', 'redshift'],
    'DL_Gpc': ['DL_Gpc', 'dL_Gpc', 'luminosity_distance_Gpc'],
    'deff': ['deff', 'Deff', 'D_eff', 'D_effective', 'd_eff', 'd_effective'],
    'effective_distance': ['effective_distance'],
    'd_hat': ['d_hat', 'dhat']
}

ANGLE_PARKEYS = {
    'ra': ['ra', 'RA', 'right_ascension'],
    'dec': ['dec', 'DEC', 'declination'],
    'psi': ['psi', 'polarization_angle', 'polarization_phase'],
    'psi_minus_psidet': ['psi_minus_psidet'],
    'psi_hat': ['psi_hat', 'psihat'],
    'philigo_hat': ['philigo_hat', 'philigohat'],
    'philigo': ['philigo', 'phiLVC'],
    'thetaligo': ['thetaligo', 'thetaLVC'], 'costhetaligo': ['costhetaligo', 'costhetaLVC'],
    'vphi': ['vphi', 'phi_ref', 'phiRef', 'phase', 'coalescence_phase', 'orbital_phase'],
    'iota': ['iota', 'inclination', 'inclin', 'inclination_angle'],
    'cosiota': ['cosiota', 'cos_iota', 'cosinclination', 'cosine_inclination', 'inclination_cosine'],
    'psiplusvphi': ['psiplusvphi'], 'psiminusvphi': ['psiminusvphi'],
    'thetaJN': ['thetaJN', 'theta_jn', 'theta_JN'],
    'costhetaJN': ['costhetaJN', 'costheta_jn', 'costheta_JN']
}

TIME_PARKEYS = {
    'tc': ['tc', 't_c', 't_coa', 'coalescence_time', 't_coalescence', 't_merge', 't_merger'],
    't_geocenter': ['tgeo', 't_geo', 't_geocenter', 'geocent_time', 'geocenter_time', 'geo_time'],
    'tgps': ['tgps', 't_gps', 'gpstime', 'gps_time', 'tGPS', 't_GPS', 'GPStime', 'GPS_time']
}

# combine extrinsic parameters: angles distance, time
EXTRINSIC_PARKEYS = dcopy(ANGLE_PARKEYS)
EXTRINSIC_PARKEYS.update(DISTANCE_PARKEYS)
EXTRINSIC_PARKEYS.update(TIME_PARKEYS)

ANALYSIS_PARKEYS = {
    'lnl': ['lnl', 'lnL', 'deltalogL', 'deltalogl', 'loglr', 'log_likelihood', 'lnlike', 'loglike', 'loglikelihood'],
    'lnPrior': ['lnPrior', 'ln_prior', 'log_prior', 'lnprior', 'logprior'],
    'lnPosterior': ['lnPosterior', 'lnPost', 'log_posterior', 'log_post', 'lnL+lnPrior', 'lnPrior+lnL'],
    'cosmo_prior': ['cosmo_prior', 'cosmo_jacobian'], 'prior': ['prior'],
    'snr': ['snr', 'SNR', 'signal_to_noise', 'signal-to-noise', 'matched_filter_snr'],
    'snr2': ['snr2', 'SNR2', 'snr_squared', 'SNR_squared', 'snr_sq', 'SNR_sq', 'snrsq', 'SNRsq',
             'signal_to_noise_squared', 'signal-to-noise-squared'],
    'lnLmax': ['lnLmax', 'lnL_max', 'maximum_log_likelihood', 'max_log_likelihood',
               'maxl_loglr', 'max_lnL', 'max_lnlike', 'lnlike_max'],
    'lnLmarg': ['lnLmarg', 'lnL_marg', 'log_likelihood_marginalized'],
    'lnL_Lmax': ['lnL_Lmax', 'log_max_likelihood_ratio', 'delta_log_max_likelihood'],
    'f_ref': ['f_ref', 'fRef', 'reference_frequency'], 'f_lo': ['f_lo', 'f_low'], 'f_hi': ['f_hi', 'f_high'],
    'approximant': ['approximant', 'approx'], 'score': ['score'], 'index': ['index'], 'rank': ['rank'],
    'Fplus': ['Fplus', 'fplus'], 'Fcross': ['Fcross', 'fcross']
}
ANALYSIS_PARKEYS.update({f'lnl_{d[0]}': [f'lnl_{d[0]}', f'lnL_{d}', f'deltalogl{d}', f'logl{d}']
                         for d in ['H1', 'L1', 'V1']})
ANALYSIS_PARKEYS.update({f'snr_{d[0]}': [f'snr_{d[0]}', f'{d[0]}_matched_filter_snr', f'snr_{d}', f'{d}_matched_filter_snr']
                         for d in ['H1', 'L1', 'V1']})
ANALYSIS_PARKEYS.update({f'fplus_{d[0]}': [f'fplus_{d[0]}', f'Fplus_{d[0]}', f'fplus_{d}', f'Fplus_{d}']
                         for d in ['H1', 'L1', 'V1']})
ANALYSIS_PARKEYS.update({f'fcross_{d[0]}': [f'fcross_{d[0]}', f'Fcross_{d[0]}', f'fcross_{d}', f'Fcross_{d}']
                         for d in ['H1', 'L1', 'V1']})
ANALYSIS_PARKEYS.update({f'antenna_{d[0]}': [f'antenna_{d[0]}', f'Antenna_{d[0]}', f'antenna_{d}', f'Antenna_{d}']
                         for d in ['H1', 'L1', 'V1']})

# combine intrinsic and extrinsic
ALL_PARKEYS = dcopy(INTRINSIC_PARKEYS)
ALL_PARKEYS.update(EXTRINSIC_PARKEYS)
ALL_PARKEYS.update(ANALYSIS_PARKEYS)

# map taking every element of ALL_PARKEYS[k] back to k
PARKEY_MAP = {}
for k, alt_keys in ALL_PARKEYS.items():
    PARKEY_MAP.update({k_alt: k for k_alt in alt_keys})


def get_key(dic, key, map_to_all_keys=PARKEY_MAP):
    if key in dic:
        return key
    return dic[map_to_all_keys[key]]

def get_from_pdic(pdic, param_key, alt_val=None, map_to_all_keys=PARKEY_MAP):
    dkey = get_key(pdic, param_key, map_to_all_keys=map_to_all_keys)
    return (alt_val if dkey is None else pdic[dkey])

def get_param_samples(samples, key, alt_val=None, map_to_all_keys=PARKEY_MAP):
    skey = get_key(samples, key, map_to_all_keys=map_to_all_keys)
    return (alt_val if skey is None else np.asarray(samples[skey]))

def compare_pdics(d1, d2, common_only=True, use_all_keys=None, map_to_all_keys=PARKEY_MAP):
    testkeys = d1.keys()
    if use_all_keys in [None, False]:
        if common_only:
            # this option only looks at parameters appearing in both dicts
            testkeys = [k for k in d2.keys() if k in testkeys]
        elif len(testkeys) != len(d2.keys()):
            # this option checks for different numbers of keys
            return False
        # if same number of keys but not the same params, common_only=False will raise key error
        return {k: np.allclose(d1[k], d2[k]) for k in testkeys}
    else:
        if use_all_keys is True:
            use_all_keys = ALL_PARKEYS
        # see which parameters are contained in both
        d2keys = [get_dict_key(d2, k, all_keys=use_all_keys, map_to_all_keys=map_to_all_keys) for k in testkeys]
        if common_only:
            testkeys = [(k1, k2) for k1, k2 in zip(testkeys, d2keys) if k2 is not None]
        elif (len(testkeys) != len(d2.keys())):
            # this option checks for different numbers of keys
            return False
        else:
            # if same number of keys but not the same params, common_only=False will raise key error
            testkeys = zip(testkeys, d2keys)
        return {k1: np.allclose(d1[k1], d2[k2]) for k1, k2 in testkeys}



# STANDARD PARAMETER RANGES
# --------------------------

SPIN_MAG_LIM = 0.99

DEFAULT_RNG_DIC = {
    'mchirp': [3, 300], 'mchirp_source': [3, 300],
    'mtot': [5, 500], 'mtot_source': [5, 500],
    'q': [0.05, 1], 'q1': [1, 20],
    'lnq': [-2.9, 0], 'eta': [0.049, 0.25],
    'm1': [3, 400], 'm1_source': [3, 400],
    'm2': [3, 400], 'm1_source': [3, 400],
    's1x': [-SPIN_MAG_LIM, SPIN_MAG_LIM], 's1y': [-SPIN_MAG_LIM, SPIN_MAG_LIM], 's1z': [-SPIN_MAG_LIM, SPIN_MAG_LIM],
    's2x': [-SPIN_MAG_LIM, SPIN_MAG_LIM], 's2y': [-SPIN_MAG_LIM, SPIN_MAG_LIM], 's2z': [-SPIN_MAG_LIM, SPIN_MAG_LIM],
    'chieff': [-SPIN_MAG_LIM, SPIN_MAG_LIM],
    'chiperp': [1 - SPIN_MAG_LIM, SPIN_MAG_LIM],
    'chip': None, 'chia': None,
    'cumchidiff': [0, 1],
    's1': [0, SPIN_MAG_LIM], 's2': [0, SPIN_MAG_LIM],
    's1theta': [0, np.pi], 's2theta': [0, np.pi],
    's1costheta': [-1, 1], 's2costheta': [-1, 1],
    's1phi': [0, 2 * np.pi], 's2phi': [0, 2 * np.pi],
    'cums1r_s1z': [0, SPIN_MAG_LIM], 'cums2r_s2z': [0, SPIN_MAG_LIM],
    'cums1z': [1 - SPIN_MAG_LIM, SPIN_MAG_LIM], 'cums2z': [1 - SPIN_MAG_LIM, SPIN_MAG_LIM],
    'd_luminosity': [0.001, 10000], 'DL': [0.001, 10000], 'Dcomov': [0.001, 5000],
    'Vcomov': [1e-9, 1e12], 'z': [0, 10], 'DL_Gpc': [1e-6, 10], 'd_hat': [0.001, 500],
    'ra': [0, 2 * np.pi], 'dec': [-np.pi / 2, np.pi / 2], 'psi': [0, np.pi],
    'philigo': [0, 2 * np.pi], 'thetaligo': [0, np.pi], 'costhetaligo': [-1, 1],
    'vphi': [0, 2 * np.pi], 'iota': [0, np.pi], 'cosiota': [-1, 1],
    'deff': [.001, 10000], 'psiplusvphi': [0, 3 * np.pi], 'psiminusvphi': [-2 * np.pi, np.pi], 'tc': [-1, 1]
}



# Latex formatting
# ----------------
param_labels = {'mchirp': r'$\mathcal{{M}}^{\rm det}$',
                'q': r'$q$',
                'q1': r'$q^{-1}$',
                'chieff': r'$\chi_{{\rm eff}}$',
                'chip': r'$\chi_p$',
                'eta': r'$\eta$',
                'm1': r'$m_1$',
                'm2': r'$m_2$',
                'mtot': r'$M_{{\rm tot}}^{{\rm det}}$',
                'm1_source': r'$m_1^{{\rm src}}$',
                'm2_source': r'$m_2^{{\rm src}}$',
                'mtot_source': r'$M_{{\rm tot}}^{{\rm src}}$',
                'mchirp_source': r'$\mathcal{M}^{{\rm src}}$',
                'd_luminosity': r'$D_L$',
                'DL': r'$D_L$',
                'dL': r'$D_L$',
                'DL_Gpc': r'$D_L$',
                'Dcomov': r'$D_{\rm comoving}$',
                'Vcomov': r'$V_{\rm comoving}$',
                'z': r'$z$',
                's1': r'$|s_1|$',
                's2': r'$|s_2|$',
                'spin1': r'$|s_1|$',
                'spin2': r'$|s_2|$',
                's1theta': r'$\cos^{-1}( s_{1,z} / |s_1| )$',
                's2theta': r'$\cos^{-1}( s_{2,z} / |s_2| )$',
                's1costheta': r'$\hat{s}_1 \cdot \hat{L}$',
                's2costheta': r'$\hat{s}_2 \cdot \hat{L}$',
                's1r': r'$\sqrt{s_{1,x}^2 + s_{1,y}^2}$',
                's2r': r'$\sqrt{s_{2,x}^2 + s_{2,y}^2}$',
                's1phi': r'$\tan^{-1}( s_{1,y} / s_{1,x} )$',
                's2phi': r'$\tan^{-1}( s_{2,y} / s_{2,x} )$',
                's1x': r'$s_{1,x}$',
                's1y': r'$s_{1,y}$',
                's1z': r'$s_{1,z}$',
                's2x': r'$s_{2,x}$',
                's2y': r'$s_{2,y}$',
                's2z': r'$s_{2,z}$',
                'thetaJN': r'$\theta_{J N}$',
                'costhetaJN': r'$\hat{J} \cdot \hat{N}$',
                'phiJL': r'$\phi_{J L}$',
                'phi12': r'$\phi_{1 2}$',
                'RA': r'$\alpha$',
                'DEC': r'$\delta$',
                'ra': r'$\alpha$',
                'dec': r'$\delta$',
                'psi': r'$\psi$',
                'vphi': r'$\varphi$',
                'thetaligo': r'$\theta_{\rm LIGO}$',
                'philigo': r'$\phi_{\rm LIGO}$',
                'log10rate': r'$\log_{10} (R \rm Gpc^3yr)$',
                'costhetaligo': r'$\cos \theta_{\rm LIGO}$',
                'cosiota': r'$\cos \iota$',
                'iota': r'$\iota$',
                'lnq': r'$\ln q$',
                'deff': r'$D_{\rm eff}(D_L, \iota)$',
                'd_hat': r'$D_{\rm eff} / \mathcal{M}$',
                'chiperp': r'$\chi_\perp$',
                'psiplusvphi': r'$\psi + \varphi$',
                'psiminusvphi': r'$\psi - \varphi$',
                'tc': r'$t_c$',
                't_geocenter': r'$t_{\rm geocenter}$',
                'tgeo': r'$t_{\rm geocenter}$',
                'tgps': r'$t_{\rm GPS}$',
                'lnl': r'$\Delta \ln \mathcal{L}$',
                'lnl_H': r'$\Delta \ln \mathcal{L}_{H}$',
                'lnl_L': r'$\Delta \ln \mathcal{L}_{L}$',
                'lnl_V': r'$\Delta \ln \mathcal{L}_{V}$',
                'lnL': r'$\Delta \ln \mathcal{L}$',
                'lnL_H1': r'$\Delta \ln \mathcal{L}_{H1}$',
                'lnL_L1': r'$\Delta \ln \mathcal{L}_{L1}$',
                'lnL_V1': r'$\Delta \ln \mathcal{L}_{V1}$',
                'lnPrior': r'$\Delta \ln \Pi$',
                'lnPosterior': r'$\Delta \ln \mathcal{L} + \Delta \ln \Pi$',
                'lnLmax': r'$max_{\vec{\theta}} \Delta \ln \mathcal{L}(\vec{\theta} | d)$',
                'lnL_Lmax': r'$\ln( \mathcal{L} / \mathcal{L}_{\rm{max}} )$',
                'snr': r'$\rho$',
                'snr2': r'$\rho^2$',
                'Fplus': r'$F_+$',
                'Fcross': r'$F_{\times}$',
                'fplus_H': r'$F_+^{(H1)}(\alpha, \delta, \psi)$',
                'fplus_L': r'$F_+^{(L1)}(\alpha, \delta, \psi)$',
                'fplus_V': r'$F_+^{(V1)}(\alpha, \delta, \psi)$',
                'fcross_H': r'$F_{\times}^{(H1)}(\alpha, \delta, \psi)$',
                'fcross_L': r'$F_{\times}^{(L1)}(\alpha, \delta, \psi)$',
                'fcross_V': r'$F_{\times}^{(V1)}(\alpha, \delta, \psi)$',
                'antenna_H': r'H1: $F_+^2 + F_{\times}^2$',
                'antenna_L': r'L1: $F_+^2 + F_{\times}^2$',
                'antenna_V': r'V1: $F_+^2 + F_{\times}^2$',
                's1phi_plus_vphi': r'$\phi_{s1} + \varphi$',
                's2phi_plus_vphi': r'$\phi_{s2} + \varphi$',
                's1phi_hat': r'$\hat{\phi}_{s1}$',
                's2phi_hat': r'$\hat{\phi}_{s2}$',
                'psi_minus_psidet': r'$\psi - \psi_{\rm det}$',
                'psi_hat': r'$\hat{\psi}$',
                'philigo_hat': r'$\hat{\phi}_{\rm LIGO}$',
                'effective_distance': r'$D_{\rm eff}(D_L, \iota, F_+, F_{\times})$',
                's1x_prime': r"$s_{1,x'}$",
                's1y_prime': r"$s_{1,y'}$",
                's2x_prime': r"$s_{2,x'}$",
                's2y_prime': r"$s_{2,y'}$",
                's1x_prime_rescale': r"$\dfrac{s_{1,x'}}{\sqrt{1-s_{1,z}^2}}$",
                's1y_prime_rescale': r"$\dfrac{s_{1,y'}}{\sqrt{1-s_{1,z}^2}}$",
                's2x_prime_rescale': r"$\dfrac{s_{2,x'}}{\sqrt{1-s_{2,z}^2}}$",
                's2y_prime_rescale': r"$\dfrac{s_{2,y'}}{\sqrt{1-s_{2,z}^2}}$",
                's1x_newsign': r"$s_{1,x}$ $\cdot$ sign[cos($\iota$)]",
                's1y_newsign': r"$s_{1,y}$ $\cdot$ sign[cos($\iota$)]",
                's2x_newsign': r"$s_{2,x}$ $\cdot$ sign[cos($\iota$)]",
                's2y_newsign': r"$s_{2,y}$ $\cdot$ sign[cos($\iota$)]",
                'cumchidiff': r"$\chi_{\rm cum}^{\rm diff}$"
                }
# units associated to each parameter
units = {mass_key: r'M$_{\odot}$'
         for mass_key in ['mchirp', 'm1', 'm2', 'mtot',
                          'm1_source', 'm2_source', 'mtot_source']}
units.update({k: 'Mpc' for k in ['DL', 'dL', 'deff', 'Dcomov',
                                 'effective_distance', 'd_luminosity']})
units['DL_Gpc'] = 'Gpc'
units['Vcomov'] = r'Mpc$^3$'
units.update({k: 's' for k in ['tc', 'tgeo', 'tgps', 't_geocenter']})
units.update({k: 'rad' for k in ['ra', 'RA', 'dec', 'DEC', 'psi', 'iota', 'vphi', 'thetaligo', 'philigo',
                                 'psiplusvphi', 'psiminusvphi', 's1theta', 's2theta', 's1phi', 's2phi',
                                 'thetaJN', 'phiJL', 'phi12', 's1phi_plus_vphi', 's2phi_plus_vphi',
                                 's1phi_hat', 's2phi_hat', 'psi_minus_psidet', 'psi_hat', 'philigo_hat']})
for k in param_labels.keys():
    if units.get(k, None) is None:
        units[k] = ''

# make version of labels that uses \chi_j for dimensionless spins instead of s_j
param_labels_chi = {k: v.replace(r's_', r'\chi_').replace(r'\hat{s}', r'\vec{\chi}')
                    for k, v in param_labels.items()}


# names for each parameter that are recognized by labelling system
param_names = {'mchirp': 'Detector Frame Chirp Mass',
               'q': 'Mass Ratio',
               'q1': 'Inverse Mass Ratio',
               'chieff': 'Effective Spin',
               'chip': 'In-Plane Spin Measure',
               'eta': 'Symmetric Mass Ratio',
               'm1': 'Detector Frame Larger Mass',
               'm2': 'Detector Frame Smaller Mass',
               'mtot': 'Detector Frame Total Mass',
               'm1_source': 'Source Frame Larger Mass',
               'm2_source': 'Source Frame Smaller Mass',
               'mtot_source': 'Source Frame Total Mass',
               'mchirp_source': 'Source Frame Chirp Mass',
               'd_luminosity': 'Luminosity Distance',
               'DL': 'Luminosity Distance',
               'dL': 'Luminosity Distance',
               'DL_Gpc': 'Luminosity Distance',
               'Dcomov': 'Comoving Distance',
               'Vcomov': 'Comoving Volume',
               'd_hat': 'Effective Distance Over Chirp Mass',
               'z': 'Redshift',
               's1': 'Spin Magnitude of Larger Mass',
               's2': 'Spin Magnitude of Smaller Mass',
               'spin1': 'Spin Magnitude of Larger Mass',
               'spin2': 'Spin Magnitude of Smaller Mass',
               's1theta': 'Spin Tilt of Larger Mass',
               's2theta': 'Spin Tilt of Smaller Mass',
               's1costheta': 'Spin Tilt Cosine of Larger Mass',
               's2costheta': 'Spin Tilt Cosine of Smaller Mass',
               's1r': 'In-Plane Spin Magnitude of Larger Mass',
               's2r': 'In-Plane Spin Magnitude of Smaller Mass',
               's1phi': 'Spin Azimuth of Larger Mass',
               's2phi': 'Spin Azimuth of Smaller Mass',
               's1x': 'Spin X-Component of Larger Mass',
               's1y': 'Spin Y-Component of Larger Mass',
               's1z': 'Spin Z-Component of Larger Mass',
               's2x': 'Spin X-Component of Smaller Mass',
               's2y': 'Spin Y-Component of Smaller Mass',
               's2z': 'Spin Z-Component of Smaller Mass',
               'thetaJN': 'Angle from Total Spin to Line-of-Sight',
               'costhetaJN': 'Cosine from Total Spin to Line-of-Sight',
               'phiJL': 'Angle from Total Spin to Orbital Angular Momentum',
               'phi12': 'Angle from Spin of Larger Mass to Spin of Smaller Mass',
               'RA': 'Right Ascension',
               'DEC': 'Declination',
               'ra': 'Right Ascension',
               'dec': 'Declination',
               'psi': 'Polarization Phase',
               'vphi': 'Orbital Phase',
               'thetaligo': 'Detector Frame Sky Location Polar Angle',
               'philigo': 'Detector Frame Sky Location Azimuth',
               'log10rate': 'Log Base 10 Rate',
               'costhetaligo': 'Detector Frame Sky Location Polar Cosine',
               'cosiota': 'Inclination Cosine',
               'iota': 'Inclination',
               'lnq': 'Log Mass Ratio',
               'deff': 'Effective Distance',
               'chiperp': 'Perpendicular Spin Measure',
               'psiplusvphi': 'Polarization Phase Plus Orbital Phase',
               'psiminusvphi': 'Polarization Phase Minus Orbital Phase',
               'tc': 'Coalescence Time',
               't_geocenter': 'Geocenter Time',
               'tgeo': 'Geocenter Time',
               'tgps': 'GPS Time',
               'lnl': 'Log Likelihood',
               'lnl_H': 'Hanford Log Likelihood',
               'lnl_L': 'Livingston Log Likelihood',
               'lnl_V': 'Virgo Log Likelihood',
               'lnL': 'Log Likelihood',
               'lnL_H1': 'Hanford Log Likelihood',
               'lnL_L1': 'Livingston Log Likelihood',
               'lnL_V1': 'Virgo Log Likelihood',
               'lnPrior': 'Log Prior',
               'lnPosterior': 'Log Posterior',
               'lnLmax': 'Maximized Log Likelihood',
               'snr': 'Signal-to-Noise Ratio',
               'snr2': 'Squared Signal-to-Noise Ratio',
               'fplus_H': 'Hanford Antenna Plus',
               'fplus_L': 'Livingston Antenna Plus',
               'fplus_V': 'Virgo Antenna Plus',
               'fcross_H': 'Hanford Antenna Cross',
               'fcross_L': 'Livingston Antenna Cross',
               'fcross_V': 'Virgo Antenna Cross',
               'antenna_H': 'Hanford Antenna Response',
               'antenna_L': 'Livingston Antenna Response',
               'antenna_V': 'Virgo Antenna Response',
               'Fplus': 'Reference Detector Antenna Plus',
               'Fcross': 'Reference Detector Antenna Cross'}
