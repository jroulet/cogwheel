"""
Provide classes ``CornerPlot`` and ``MultiCornerPlot``
similar to those in the ``cogwheel.plotting`` module but
with gravitational-wave-specific defaults for LaTeX labels.
"""
from cogwheel import plotting

_LABELS = {
    # Mass
    'mchirp': r'$\mathcal{M}^{\rm det}$',
    'lnq': r'$\ln q$',
    'q': r'$q$',
    'eta': r'$\eta$',
    'm1': r'$m_1^{\rm det}$',
    'm2': r'$m_2^{\rm det}$',
    'mtot': r'$M_{\rm tot}^{\rm det}$',
    'm1_source': r'$m_1^{\rm src}$',
    'm2_source': r'$m_2^{\rm src}$',
    'mtot_source': r'$M_{\rm tot}^{\rm src}$',
    'mchirp_source': r'$\mathcal{M}^{\rm src}$',
    'l1': r'$\Lambda_1$',
    'l2': r'$\Lambda_2$',
    # Spin
    'chieff': r'$\chi_{\rm eff}$',
    'cumchidiff': r'$C_{\rm diff}$',
    's1phi_hat': r'$\hat{\phi}_{s1}$',
    's2phi_hat': r'$\hat{\phi}_{s2}$',
    'cums1r_s1z': r'$C_1^\perp$',
    'cums2r_s2z': r'$C_2^\perp$',
    'cums1z': r'$C_1^z$',
    'cums2z': r'$C_2^z$',
    's1x': r'$s_{1x}$',
    's1y': r'$s_{1y}$',
    's1z': r'$s_{1z}$',
    's2x': r'$s_{2x}$',
    's2y': r'$s_{2y}$',
    's2z': r'$s_{2z}$',
    's1x_n': r'$s_{1x}^N$',
    's1y_n': r'$s_{1y}^N$',
    's2x_n': r'$s_{2x}^N$',
    's2y_n': r'$s_{2y}^N$',
    's1': r'$|s_1|$',
    's2': r'$|s_2|$',
    's1theta': r'$\theta_{s1}$',
    's2theta': r'$\theta_{s2}$',
    's1r': r'$s_1^\perp$',
    's2r': r'$s_2^\perp$',
    's1phi': r'$\phi_{s1}$',
    's2phi': r'$\phi_{s2}$',
    'chip': r'$\chi_p$',
    # Distance
    'd_hat': r'$\hat{d}$',
    'd_luminosity': r'$d_L$',
    'z': r'$z$',
    # Orientation
    'phi_linfree': r'$\phi_{\rm LF}$',
    'phi_ref': r'$\phi_{\rm ref}$',
    'phi_ref_hat': r'$\hat{\phi}_{\rm ref}$',
    'psi_hat': r'$\hat{\psi}$',
    'psi': r'$\psi$',
    'cosiota': r'$\cos \iota$',
    'iota': r'$\iota$',
    'thetaJN': r'$\theta_{JN}$',
    'costheta_jn': r'$\cos \theta_{JN}$',
    'phi_jl': r'$\phi_{JL}$',
    'phi_jl_hat': r'$\hat\phi_{JL}$',
    'phi12': r'$\phi_{12}$',
    # Location
    'costhetanet': r'$\cos \theta_{\rm net}$',
    'phinet_hat': r'$\hat{\phi}_{\rm net}$',
    'ra': r'$\alpha$',
    'dec': r'$\delta$',
    'thetanet': r'$\theta_{\rm net}$',
    'phinet': r'$\phi_{\rm net}$',
    # Time
    't_refdet': r'$t_{\rm ref\,det}$',
    'tc': r'$t_c$',
    't_geocenter': r'$t_{\rm ⴲ}$',
    't_linfree': r'$t_{\rm LF}$',
    # Likelihood
    'lnl': r'$\ln \mathcal{L}$',
    'lnl_H': r'$\ln \mathcal{L}_H$',
    'lnl_L': r'$\ln \mathcal{L}_L$',
    'lnl_V': r'$\ln \mathcal{L}_V$',
    'lnl_marginalized': r'$\ln \overline{\mathcal{L}}$',
    'h_h': r'$\langle h | h \rangle$',
    # Cumulatives
    'u_t_linfree': r'$u_t$',
    'u_psi': r'$u_\psi$',
    'u_costhetanet': r'$u_{\theta_{\rm net}}$',
    'u_phinet_hat': r'$u_{\hat\phi_{\rm net}}$',
    }

_UNITS = (dict.fromkeys(['mchirp', 'm1', 'm2', 'mtot', 'mtot_source',
                         'm1_source', 'm2_source', 'mchirp_source'],
                        r'M$_\odot$')
          | dict.fromkeys(['t_refdet', 'tc', 't_geocenter', 't_linfree'], 's')
          | {'d_hat': r'$\frac{\rm{Mpc}}{M_{\odot}^{5/6}}$',
             'd_luminosity': 'Mpc',}
         )


class CornerPlot(plotting.CornerPlot):
    """Has default latex labels for gravitational wave parameters."""
    DEFAULT_LATEX_LABELS = plotting.LatexLabels(_LABELS, _UNITS)


class MultiCornerPlot(plotting.MultiCornerPlot):
    """Has default latex labels for gravitational wave parameters."""
    corner_plot_cls = CornerPlot
