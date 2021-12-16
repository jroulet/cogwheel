from . import standard_intrinsic_transformations as pxform

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
units.update({k: 'rad' for k in ['ra', 'RA', 'dec', 'DEC', 'psi',
    'iota', 'vphi', 'thetaligo', 'philigo', 'psiplusvphi', 'psiminusvphi',
    's1theta', 's2theta', 's1phi', 's2phi', 'thetaJN', 'phiJL', 'phi12',
    's1phi_plus_vphi', 's2phi_plus_vphi', 's1phi_hat', 's2phi_hat',
    'psi_minus_psidet', 'psi_hat', 'philigo_hat']})
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


def fmt_num(num, prec_override=None):
    if prec_override is not None:
        return ('{:.'+str(prec_override)+'f}').format(num)
    if abs(num) > 10000:
        return f'{num:.1e}'
    if abs(num) > 10:
        return f'{num:.0f}'
    if abs(num) > 1:
        return f'{num:.1f}'
    return f'{num:.2f}'

def label_from_pdic(pdic, keys=['mchirp', 'chieff'], pre='', post='',
                    sep=' ', connector=' = ', prec_override=None,
                    add_units=False):
    pstr = ''
    get_sep = lambda k: sep
    if add_units:
        get_sep = lambda k: ' (' + units[k] + ')' + sep
    pdic_use = dict(pdic)
    if any([(pdic_use.get(k) is None) for k in keys]):
        pxform.compute_samples_aux_vars(pdic_use)
    get_num = lambda k: (r'$' + fmt_num(pdic_use.get(k), prec_override)
                         + r'$' + get_sep(k))
    for k in keys:
        pstr += param_labels.get(k, k) + connector + get_num(k)
    return pre + pstr[:-len(sep)] + post


