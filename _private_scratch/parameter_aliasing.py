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
    return map_to_all_keys.get(key, None)

def get_from_dic(dic, key, alt_val=None, map_to_all_keys=PARKEY_MAP):
    return dic.get(get_key(dic, key, map_to_all_keys=map_to_all_keys), alt_val)