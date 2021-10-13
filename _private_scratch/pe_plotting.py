import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy as dcopy
import os
import json
import h5py
from matplotlib.colors import Normalize as colorsNormalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d import Axes3D
import lalsimulation as lalsim

PIPELINE_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'gw_detection_ias'))
sys.path.append(PIPELINE_PATH)
import gw_pe
import utils
import pop_inference.gw_pe_bookkeeping as ias_bookkeeping
bookkeeping = ias_bookkeeping.bookkeeping

from pop_inference import grid as gd
from pop_inference import gw_utils
from ligo_angles import radec_to_thetaphiLV
from pop_inference.gw_utils import dz_dDL

import gw_parameter_dictionary as gwpdic
GWPD = gwpdic.GWParameterDictionary


def printarr(arr, prec=4, pre='', post='', sep='  ', form='f'):
    print(pre + np.array2string(np.asarray(arr), separator=sep,
                                max_line_width=np.inf, threshold=np.inf,
                                formatter={'float_kind':lambda x: f"%.{prec}{form}" % x}) + post)

def fmt(num, prec=4, form='f'):
    formstr = '{:.' + str(prec) + form + '}'
    return formstr.format(num)

def invert_dict(dict_in, iter_val=False):
    """return dictionary with inverted key-value pairs"""
    if iter_val is True:
        dict_out = {}
        for key, val in dict_in.items():
            dict_out.update({v: key for v in val})
        return dict_out
    else:
        return {val: key for key, val in dict_in.items()}

def pull_lvc_samples_from_ias(evname, dname='/data/bzackay/GW/LSC_PE_samples/', user='srolsen'):
    paths = [os.path.join(dname, evname + suf) for suf in ['.h5', '_comoving.h5', '_prior.npy']]
    cmds = [f'scp {user}@ssh1.sns.ias.edu:{p} {p}' for p in paths]
    for cmd in cmds:
        try:
            os.system(cmd)
        except:
            print('cannot execute:', cmd)

def get_best_pdics(samples, key_rngs={}, get_best_inds=np.arange(20, dtype=int), lnLmin=0):
    s = samples[samples['lnL'] > lnLmin]
    for k, rng in key_rngs.items():
        s = s[s[k] > rng[0]]
        s = s[s[k] < rng[1]]
    s = s.sort_values('lnL', ascending=False).reset_index()
    return [dict(s.iloc[j]) for j in get_best_inds]
    
########################################
#### NON-CLASS PLOTTING FUNCTIONS

def label_from_key(key):
    return gw_utils.param_labels.get(gw_utils.PARKEY_MAP.get(key, key), key)

def corner_plot_samples(samps, pvkeys=['mtot', 'q', 'chieff'], title=None,
                        figsize=(9,7), scatter_points=None, weights=None,
                        grid_kws={}, fig=None, ax=None, return_grid=False, **corner_plot_kws):
    """make corner plots"""
    units, plabs = gw_utils.units, gw_utils.param_labels
    for k in pvkeys:
        if k not in units.keys():
            units[k] = ''
        if k not in plabs.keys():
            plabs[k] = k
    sg = gd.Grid.from_samples(pvkeys, samps, weights=weights, pdf_key=None,
                              units=units, labels=plabs, **grid_kws)
    ff, aa = sg.corner_plot(pdf=None, title=title, figsize=figsize, set_legend=True,
                            scatter_points=scatter_points, fig=fig, ax=ax, **corner_plot_kws)
    if return_grid:
        return ff, aa, sg
    return ff, aa

def corner_plot_list(samps_list, samps_names, pvkeys=['mtot', 'q', 'chieff'], weight_key=None,
                     figsize=(9,7), scatter_points=None, fractions=[.5, .9], grid_kws={},
                     multigrid_kws={}, fig=None, ax=None, return_grid=False, **corner_plot_kws):
    grids = []
    units, plabs = gw_utils.units, gw_utils.param_labels
    for k in pvkeys:
        if k not in units.keys():
            units[k] = ''
        if k not in plabs.keys():
            plabs[k] = k
    for p, nm in zip(samps_list, samps_names):
        grids.append(gd.Grid.from_samples(pvkeys, p, units=units, labels=plabs, pdf_key=nm,
                                          weights=(None if weight_key is None
                                                   else p.samples[weight_key]), **grid_kws))
    multigrid = gd.MultiGrid(grids, fractions=fractions, **multigrid_kws)
    ff, aa = multigrid.corner_plot(set_legend=True, figsize=figsize, scatter_points=scatter_points,
                                   fig=fig, ax=ax, **corner_plot_kws)
    if return_grid:
        return ff, aa, multigrid
    return ff, aa

def plot_at_dets(xplot, dets_yplot, ax=None, fig=None, label=None, xlabel='Frequency (Hz)', ylabel='Amplitude',
                 plot_type='loglog', xlim=None, ylim=None, title=None, det_names=['H1', 'L1', 'V1'],
                 figsize=None, **plot_kws):
    if ax is None:
        fig, ax = plt.subplots(len(det_names), sharex=True, figsize=figsize)
        fig.text(.004, .54, ylabel, rotation=90, ha='left', va='center', size=10)
        ax[0].set_xlabel(xlabel)
        for a, det in zip(ax, det_names):
            a.text(.02, .95, det, ha='left', va='top', transform=a.transAxes)
            a.tick_params(which='both', direction='in', right=True, top=True)
    if np.ndim(xplot) == 1:
        xplot = [xplot]*len(det_names)
    mask = slice(None)
    for j, a in enumerate(ax):
        if xlim is not None:
            mask = (xplot[j] >= xlim[0]) & (xplot[j] <= xlim[1])
        plotfunc = (a.loglog if plot_type in ['loglog', 'log'] else 
                    (a.semilogx if plot_type in ['semilogx', 'logx', 'xlog'] else 
                     (a.semilogy if plot_type in ['semilogy', 'logy', 'ylog'] else a.plot)))
        plotfunc(xplot[j][mask], dets_yplot[j][mask], label=label, **plot_kws)
        if label is not None:
            a.legend(title=det_names[j])
        a.set_ylim(ylim)
    if title is not None:
        plt.suptitle(title)
    return fig, ax
    
colorbar_kws_DEFAULTS = {'pad': 0.02, 'fraction': 0.1, 'aspect': 24, 'shrink': 0.5,
                         'ticks': 8, 'format': '%.2f'}
plot3d_kws_DEFAULTS = {'alpha': 0.1, 's': 0.05}

def scatter3d(x, y, z, cs=None, xlab='$\\alpha$', ylab='$\\delta$', zlab='$D_L$',
              clab='ln$\\mathcal{L}$', colorsMap='jet', fig=None, ax=None,
              title=None, titlesize=20, figsize=(14, 14),
              xlim='auto', ylim='auto', zlim='auto', plot_kws=None, colorbar_kws=None):
    if cs is None:
        cs = np.ones(len(x))
    cm = plt.get_cmap(colorsMap)
    cNorm = colorsNormalize(vmin=min(cs), vmax=max(cs))
    scalarMap = ScalarMappable(norm=cNorm, cmap=cm)
    if (fig is None) or (ax is None):
        fig = plt.figure(figsize=figsize)
        ax = Axes3D(fig)
    
    plot_kwargs = dcopy((plot3d_kws_DEFAULTS if plot_kws is None else plot_kws))
    ax.scatter(x, y, z, c=scalarMap.to_rgba(cs), **plot_kwargs)
    scalarMap.set_array(cs)
    cbar_kws = dcopy((colorbar_kws_DEFAULTS if colorbar_kws is None else colorbar_kws))
    if isinstance(cbar_kws.get('ticks'), int):
        cbar_kws['ticks'] = np.linspace(np.min(cs), np.max(cs), cbar_kws['ticks'], endpoint=True)
    cbar = fig.colorbar(scalarMap, **cbar_kws)
    cbar.set_label(clab)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_zlabel(zlab)
    if xlim == 'auto':
        xlim = (np.min(x), np.max(x))
    ax.set_xlim(xlim)
    if ylim == 'auto':
        ylim = (np.min(y), np.max(y))
    ax.set_ylim(ylim)
    if zlim == 'auto':
        zlim = (np.min(z), np.max(z))
    ax.set_zlim(zlim)
    if plot_kwargs.get('label') is not None:
        ax.legend(title=plot_kwargs.get('legend_title'))
    if title is not None:
        ax.set_title(title, size=titlesize)
    return fig, ax

def get_spin_plot_par(samples, key):
    if isinstance(key, list):
        return [get_spin_plot_par(samples, k) for k in key]
    elif len(key) == 3:
        return samples[key]
    else:
        if 'prime' in key:
            j = int(key[1])
            trigfunc = (np.cos if 'x' in key else np.sin)
            r = samples[f'cums{j}r_s{j}z']**.5
            if 'rescale' in key:
                return r * trigfunc(samples[f's{j}phi_hat'])
            else:
                return r * trigfunc(samples[f's{j}phi_hat']) * np.sqrt(1 - samples[f's{j}z']**2)
        else:
            assert 'sign' in key, "key must be 'sjx'+('' or '_newsign' or '_prime_rescale' or 'prime')"
            signcosiota = np.sign((samples['cosiota'] if 'cosiota' in samples else np.cos(samples['iota'])))
            return samples[key[:3]] * signcosiota

def plot_inplane_spin(pe_samples, color_key='q', use_V3=False, secondary_spin=False,
                      fractions=[.5, .95], plotstyle_color='r', scatter_alpha=.5,
                      figsize=None, title=None, tight=False, **colorbar_kws):
    plotstyle_2d = gd.PlotStyle2d(plotstyle_color, fractions=fractions,
                                  show_cl=True, clabel_fs=11)
    j = (2 if secondary_spin else 1)
    plotkeys = [f's{(2 if secondary_spin else 1)}{dct}_' +
                ('prime_rescale' if use_V3 else 'newsign')
                for dct in ['x', 'y']]
    plot_samples = pd.DataFrame({k: get_spin_plot_par(pe_samples, k) for k in plotkeys})
    x = np.linspace(0, 2*np.pi, 300)
    fig, ax = plt.subplots(figsize=figsize)
    if isinstance(color_key, str):
        plt.scatter(plot_samples[plotkeys[0]], plot_samples[plotkeys[1]], s=.8, lw=0,
                    c=pe_samples[color_key], alpha=scatter_alpha)
        colorbar_kws['label'] = colorbar_kws.get('label', label_from_key(color_key))
        plt.colorbar(**colorbar_kws)
    plt.plot(np.cos(x), np.sin(x), lw=1, c='k')
    # Make grid
    g = gd.Grid.from_samples(plotkeys, plot_samples)
    # Make 2d plot
    g.grids_2d[plotkeys[0], plotkeys[1]].plot_pdf('posterior', ax, style=plotstyle_2d)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    plt.grid()
    ax.set_aspect('equal')
    plt.xlabel(label_from_key(plotkeys[0]))
    plt.ylabel(label_from_key(plotkeys[1]))
    if tight:
        plt.tight_layout()
    ax.set_title(title)
    return fig, ax

def plot_spin4d(samples, ckey='q', use_V3=False, secondary_spin=False, sign_or_scale=True,
                 xkey='s1x', ykey='s1y', zkey='s1z', nstep=1, title=None,
                 xlab='auto', ylab='auto', zlab='auto', clab='auto', plotlim=[-1.1, 1.1],
                 mask_keys_min={}, mask_keys_max={}, fig=None, ax=None, plot_kws=None,
                 figsize=(14, 14), titlesize=20, colorbar_kws=None,
                 extra_point_dicts=[(0, 0, 0)],
                 marker_if_not_dict='o', size_if_not_dict=20, color_if_not_dict='k'):
    """scatter3d but using a dataframe and keys instead of using x/y/z/color arrays directly"""
    # get x, y, z to plot
    if use_V3:
        xkey, ykey, zkey = 's1x_prime', 's1y_prime', 's1z'
        if sign_or_scale:
            xkey += '_rescale'
            ykey += '_rescale'
    elif sign_or_scale:
        xkey, ykey = 's1x_newsign', 's1y_newsign'

    if secondary_spin:
        xkey, ykey, zkey = [k.replace('1', '2') for k in [xkey, ykey, zkey]]

    x, y, z = [np.asarray(get_spin_plot_par(samples, k)) for k in [xkey, ykey, zkey]]
    # labels
    if (xlab is None) or (xlab == 'auto'):
        xlab = label_from_key(xkey)
    if (ylab is None) or (ylab == 'auto'):
        ylab = label_from_key(ykey)
    if (zlab is None) or (zlab == 'auto'):
        zlab = label_from_key(zkey)
    if (clab is None) or (clab == 'auto'):
        clab = label_from_key(ckey)
    # get colorbar array and mask based on mask_keys_min/max
    clr = np.asarray(samples[ckey])
    mask = np.ones(len(clr), dtype=bool)
    for k, v in mask_keys_min.items():
        mask *= samples[k].to_numpy() >= v
    for k, v in mask_keys_max.items():
        mask *= samples[k].to_numpy() <= v
    # plot using peplot.scatter3d
    if title == 'flat':
        title = 'Posterior Samples from PE with Flat $\\chi_{eff}$ Prior'
    elif title == 'iso':
        title = 'Posterior Samples from PE with Isotropic $\\vec{\\chi}_1, \\vec{\\chi}_2$ Priors'
    fig, ax = scatter3d(x[mask][::nstep], y[mask][::nstep], z[mask][::nstep], clr[mask][::nstep], title=title,
                      xlab=xlab, ylab=ylab, zlab=zlab, clab=clab, xlim=plotlim, ylim=plotlim, zlim=plotlim,
                      titlesize=titlesize, figsize=figsize, plot_kws=plot_kws, colorbar_kws=colorbar_kws,
                      fig=fig, ax=ax)
    # plot extra points
    for dic in extra_point_dicts:
        if gwpdic.is_dict(dic):
            xx, yy, zz = dic[xkey], dic[ykey], dic[zkey]
            ax.scatter(xx, yy, zz, marker=dic.get('marker', dic.get('m')), s=dic.get('size', dic.get('s')),
                       c=dic.get('color', dic.get('c')))
            if dic.get('text', None) is not None:
                ax.text(xx - 0.15, yy - 0.15, zz - 0.48, dic['text'], color=dic.get('textcolor'),
                        size=dic.get('textsize'))
        else:
            ax.scatter(dic[0], dic[1], dic[2], marker=marker_if_not_dict,
                       s=size_if_not_dict, c=color_if_not_dict)
    circang = np.linspace(0, 2*np.pi, 360)
    ax.plot(np.cos(circang), np.sin(circang), np.zeros(360), lw=2, c='k')
    if 'z' in zkey:
        ax.scatter(0, 0, 1.01, marker='^', c='k', s=25)
        ax.scatter(0, 0, -1, marker='s', c='k', s=20)
        ax.plot(np.zeros(100), np.zeros(100), np.linspace(-1, 1.02, 100), lw=1, c='k')
        if not use_V3:
            ax.plot(np.cos(circang), np.zeros(360), np.sin(circang), lw=1, ls=':', c='k')
            ax.plot(np.zeros(360), np.cos(circang), np.sin(circang), lw=1, ls=':', c='k')
    print(f'Plotted {np.count_nonzero(mask) // nstep} of {len(samples)} samples')
    return fig, ax

########################################
#### GEOMETRIC CONVERSION

def xyzMpc_from_ra_dec_DL(ra, dec, DL):
    """DL in Mpc, ra in [0, 2pi], dec in [-pi/2, pi/2]"""
    theta = 0.5*np.pi - dec
    return np.array([DL*np.sin(theta)*np.cos(ra), DL*np.sin(theta)*np.sin(ra), DL*np.cos(theta)])

def xyzGpc_from_ra_dec_DLMpc(ra, dec, DL):
    return xyzMpc_from_ra_dec_DL(ra, dec, DL) / 1000.

def ra_dec_DL_from_xyzMpc(xmpc, ympc, zmpc):
    dlMpc = np.sqrt(xmpc**2 + ympc**2 + zmpc**2)
    theta = np.arccos(zmpc / dlMpc)
    return np.arctan2(ympc, xmpc) % (2*np.pi), 0.5*np.pi - theta, dlMpc


####    AGN SKY POSITION  (for GW190521)  ####
agn_ra = (12*15/360 + 49*15/(360*60)) * 2*np.pi
agn_dec = (34/360 + 49/(360*60)) * 2*np.pi
agn_dL0, agn_dL1, agn_DL = 2400, 2511, 2464.
agn_mtot, agn_chieff, agn_q, agn_iota = 260, -0.3, 0.9, 1.2
agn_pdic = {'ra': agn_ra, 'dec': agn_dec, 'DL': agn_DL, 'iota': agn_iota,
            'chieff': agn_chieff, 'mtot': agn_mtot, 'q': agn_q}
agn_plot_intrinsic = {'chieff': agn_chieff, 'mtot': agn_mtot, 'q': agn_q, 'm': 'o', 's': 48,
                      'c': 'k', 'text': 'AGN', 'textcolor': 'k', 'textsize': 16}
agn_plot_location = {'ra': agn_ra, 'dec': agn_dec, 'DL': agn_DL, 'm': 'o', 's': 48,
                     'c': 'k', 'text': 'AGN', 'textcolor': 'k', 'textsize': 16}

#########################################
#### 4-DIMENSIONAL SAMPLE PLOTTING

def plot_loc3d(samples, title='flat', xlim='auto', ylim='auto', zlim='auto', nstep=2,
               ckey='lnL', clab=None, mask_keys_min={'lnL': 90}, mask_keys_max={},
               plot_kws=None, figsize=(14, 14), titlesize=20, colorbar_kws=None, units='Mpc',
               extra_point_dicts=[], fig=None, ax=None):
    x, y, z = xyzMpc_from_ra_dec_DL(samples['ra'].to_numpy(), samples['dec'].to_numpy(), samples['DL'].to_numpy())
    if units in ['kpc', 'Gpc']:
        x, y, z = np.array([x, y, z]) * (10**(3 if units == 'kpc' else -3))
    elif units != 'Mpc':
        print(f'WARNING: units={units} not recognized, plotting in MEGAPARSEC')
        units = 'Mpc'
    clr = samples[ckey].to_numpy()
    Ns = len(clr)
    mask = np.ones(Ns, dtype=bool)
    for k, v in mask_keys_min.items():
        mask *= samples[k].to_numpy() >= v
    for k, v in mask_keys_max.items():
        mask *= samples[k].to_numpy() <= v
    
    if title == 'flat':
        title = 'Posterior Samples from PE with Flat $\\chi_{eff}$ Prior'
    elif title == 'iso':
        title = 'Posterior Samples from PE with Isotropic $\\vec{\\chi}_1, \\vec{\\chi}_2$ Priors'
    if (clab is None) or (clab == 'auto'):
        clab = gw_utils.get(ckey, ckey)
    fig, ax = scatter3d(x[mask][::nstep], y[mask][::nstep], z[mask][::nstep], clr[mask][::nstep], title=title,
                      xlab=f'X ({units})', ylab=f'Y ({units})', zlab=f'Z ({units})', clab=clab,
                      xlim=xlim, ylim=ylim, zlim=zlim, titlesize=titlesize, figsize=figsize,
                      plot_kws=plot_kws, colorbar_kws=colorbar_kws, fig=fig, ax=ax)
    # plot earth at origin
    ax.scatter(0, 0, 0, marker='*', s=24, c='k')
    ax.text(-0.15, -0.15, -0.4, 'Earth', color='k', size=14)
    for dic in extra_point_dicts:
        xx, yy, zz = xyzMpc_from_ra_dec_DL(dic['ra'], dic['dec'], dic['DL'])
        if units in ['kpc', 'Gpc']:
            xx, yy, zz = np.array([xx, yy, zz]) * (10**(3 if units == 'kpc' else -3))
        ax.scatter(xx, yy, zz, marker=dic.get('marker', dic.get('m')), s=dic.get('size', dic.get('s')),
                   c=dic.get('color', dic.get('c')))
        if dic.get('text', None) is not None:
            ax.text(xx - 0.15, yy - 0.15, zz - 0.48, dic['text'], color=dic.get('textcolor'),
                    size=dic.get('textsize'))
    print(f'Plotted {np.count_nonzero(mask)} of {Ns} samples')
    return fig, ax


def plot_samples4d(samples, xkey='chieff', ykey='q', zkey='mtot', ckey='lnL',
                   xlim='auto', ylim='auto', zlim='auto', nstep=2, title=None,
                   xlab='auto', ylab='auto', zlab='auto', clab='auto',
                   mask_keys_min={}, mask_keys_max={}, fig=None, ax=None,
                   plot_kws=None, figsize=(14, 14), titlesize=20, colorbar_kws=None,
                   extra_point_dicts=[], size_key=None, size_scale=1):
    """scatter3d but using a dataframe and keys instead of using x/y/z/color arrays directly"""
    if (xlab is None) or (xlab == 'auto'):
        xlab = gw_utils.param_labels.get(xkey, xkey)
    if (ylab is None) or (ylab == 'auto'):
        ylab = gw_utils.param_labels.get(ykey, ykey)
    if (zlab is None) or (zlab == 'auto'):
        zlab = gw_utils.param_labels.get(zkey, zkey)
    if (clab is None) or (clab == 'auto'):
        clab = gw_utils.param_labels.get(ckey, ckey)
    x, y, z, clr = [np.asarray(samples[key]) for key in [xkey, ykey, zkey, ckey]]
    Ns = len(clr)
    mask = np.ones(Ns, dtype=bool)
    for k, v in mask_keys_min.items():
        mask *= samples[k].to_numpy() >= v
    for k, v in mask_keys_max.items():
        mask *= samples[k].to_numpy() <= v
    
    if isinstance(size_key, str):
        size_arr = np.asarray(samples[size_key])[mask]
        plot_kws['s'] = size_arr[::nstep] * size_scale / np.max(size_arr)
    
    if title == 'flat':
        title = 'Posterior Samples from PE with Flat $\\chi_{eff}$ Prior'
    elif title == 'iso':
        title = 'Posterior Samples from PE with Isotropic $\\vec{\\chi}_1, \\vec{\\chi}_2$ Priors'
    fig, ax = scatter3d(x[mask][::nstep], y[mask][::nstep], z[mask][::nstep], clr[mask][::nstep], title=title,
                      xlab=xlab, ylab=ylab, zlab=zlab, clab=clab, xlim=xlim, ylim=ylim, zlim=zlim,
                      titlesize=titlesize, figsize=figsize, plot_kws=plot_kws, colorbar_kws=colorbar_kws,
                      fig=fig, ax=ax)
    # plot extra points
    for dic in extra_point_dicts:
        xx, yy, zz = dic[xkey], dic[ykey], dic[zkey]
        ax.scatter(xx, yy, zz, marker=dic.get('marker', dic.get('m')), s=dic.get('size', dic.get('s')),
                   c=dic.get('color', dic.get('c')))
        if dic.get('text', None) is not None:
            ax.text(xx - 0.15, yy - 0.15, zz - 0.48, dic['text'], color=dic.get('textcolor'),
                    size=dic.get('textsize'))
    print(f'Plotted {np.count_nonzero(mask)} of {Ns} samples')
    return fig, ax

def scatter2d_color(x, y, cs=None, xlab='$\\alpha$', ylab='$\\delta$',
                    clab='ln$\\mathcal{L}$', colorsMap='jet', fig=None, ax=None,
                    title=None, titlesize=20, figsize=(14, 14),
                    xlim='auto', ylim='auto', plot_kws=None, colorbar_kws=None):
    """make 2d scatter plot with colorbar for visualizing third dimension"""
    if cs is None:
        cs = np.ones(len(x))
    cm = plt.get_cmap(colorsMap)
    cNorm = colorsNormalize(vmin=min(cs), vmax=max(cs))
    scalarMap = ScalarMappable(norm=cNorm, cmap=cm)
    if (fig is None) or (ax is None):
        fig, ax = plt.subplots(figsize=figsize)
    
    plot_kwargs = ({} if plot_kws is None else dcopy(plot_kws))
    ax.scatter(x, y, c=scalarMap.to_rgba(cs), **plot_kwargs)
    scalarMap.set_array(cs)
    cbar_kws = dcopy((colorbar_kws_DEFAULTS if colorbar_kws is None else colorbar_kws))
    if isinstance(cbar_kws.get('ticks'), int):
        cbar_kws['ticks'] = np.linspace(np.min(cs), np.max(cs), cbar_kws['ticks'], endpoint=True)
    cbar = fig.colorbar(scalarMap, **cbar_kws)
    cbar.set_label(clab)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    if xlim == 'auto':
        xlim = (np.min(x), np.max(x))
    if ylim == 'auto':
        ylim = (np.min(y), np.max(y))
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    if plot_kwargs.get('label') is not None:
        ax.legend(title=plot_kwargs.get('legend_title'))
    if title is not None:
        ax.set_title(title, size=titlesize)
    return fig, ax

def plot_samples2d_color(samples_list, xkey='chieff', ykey='q', ckey='lnL', samples_per_posterior=None,
                         fig=None, ax=None, colorbar_kws=None, colorsMap='jet', figsize=(14, 14),
                         title=None, titlesize=20, xlim='auto', ylim='auto', clim=None,
                         size_key=None, size_scale=1, alpha_key=None, alpha_scale=1, **plot_kws):
    """2d scatter plotting with color key, calls scatter2d_color()"""
    plot_list = (samples_list if isinstance(samples_list, list) else [samples_list])
    mask = slice(None)
    if samples_per_posterior is None:
        xarr = np.concatenate([np.asarray(s[xkey]) for s in plot_list])
        yarr = np.concatenate([np.asarray(s[ykey]) for s in plot_list])
        carr = np.concatenate([np.asarray(s[ckey]) for s in plot_list])
        if isinstance(size_key, str):
            size_arr = np.concatenate([np.asarray(s[size_key]) for s in plot_list])
            size_arr *= size_scale / np.max(size_arr)
            plot_kws['s'] = size_arr.copy()
        if isinstance(alpha_key, str):
            alpha_arr = np.concatenate([np.asarray(s[alpha_key]) for s in plot_list])
            alpha_arr *= alpha_scale / np.max(alpha_arr)
            plot_kws['alpha'] = alpha_arr.copy()
    else:
        randinds = [np.random.choice(np.arange(len(s)), size=samples_per_posterior)
                    for s in plot_list]
        xarr = np.concatenate([np.asarray(s[xkey])[rinds]
                               for s, rinds in zip(plot_list, randinds)])
        yarr = np.concatenate([np.asarray(s[ykey])[rinds]
                               for s, rinds in zip(plot_list, randinds)])
        carr = np.concatenate([np.asarray(s[ckey])[rinds]
                               for s, rinds in zip(plot_list, randinds)])
        if isinstance(size_key, str):
            size_arr = np.concatenate([np.asarray(s[size_key])[rinds]
                                       for s, rinds in zip(plot_list, randinds)])
            size_arr *= size_scale / np.max(size_arr)
            plot_kws['s'] = size_arr.copy()
        if isinstance(alpha_key, str):
            alpha_arr = np.concatenate([np.asarray(s[alpha_key])[rinds]
                                       for s, rinds in zip(plot_list, randinds)])
            alpha_arr *= alpha_scale / np.max(alpha_arr)
            plot_kws['alpha'] = alpha_arr.copy()
    if clim is not None:
        mask = (carr > clim[0]) & (carr < clim[1])
        xarr = xarr[mask]
        yarr = yarr[mask]
        carr = carr[mask]
        if isinstance(size_key, str):
            size_arr = size_arr[mask]
            plot_kws['s'] = size_arr.copy()
        if isinstance(alpha_key, str):
            alpha_arr = alpha_arr[mask]
            plot_kws['alpha'] = alpha_arr.copy()
    xlab = label_from_key(xkey)
    ylab = label_from_key(ykey)
    clab = label_from_key(ckey)
    return scatter2d_color(xarr, yarr, cs=carr, colorsMap=colorsMap, fig=fig, ax=ax,
              title=title, titlesize=titlesize, figsize=figsize,
              xlim=xlim, ylim=ylim, xlab=xlab, ylab=ylab,
              clab=clab, colorbar_kws=colorbar_kws, plot_kws=plot_kws)


############################################
#### ADDING COMPUTED PARAMETERS TO SAMPLES

def samples_with_ligo_angles(old_samps, f_ref=None, tgps=None, keep_new_spins=False):
    new_samps = old_samps.copy()
    new_samps['ra'] %= (2 * np.pi)
    ns = len(old_samps)
    if (tgps is not None) and ('philigo' not in new_samps):
        new_samps['thetaligo'], new_samps['philigo'] = \
                radec_to_thetaphiLV(new_samps['ra'].to_numpy(), new_samps['dec'].to_numpy(), tgps)
        new_samps['costhetaligo'] = np.cos(new_samps['thetaligo'])
    if f_ref is not None:
        getparkeys = ['iota', 's1x', 's1y', 's1z', 's2x', 's2y', 's2z', 'm1', 'm2', 'vphi']
        missingkeys = [k for k in getparkeys if k not in new_samps]
        if len(missingkeys) > 0:
            print(f'missing {missingkeys}  ==> cannot get theta_JN')
        p_in = np.concatenate([new_samps[getparkeys[:-1]].to_numpy(), \
                               f_ref * np.ones(ns)[:, np.newaxis], \
                               new_samps['vphi'].to_numpy()[:, np.newaxis]], axis=1)
        new_cols = np.zeros((ns, 7), dtype=np.float64)
        for j in range(ns):
            new_cols[j] = lalsim.SimInspiralTransformPrecessingWvf2PE(*p_in[j])
        new_samps['thetaJN'] = new_cols[:, 0]
        new_samps['phiJL'] = new_cols[:, 1]
        new_samps['phi12'] = new_cols[:, 4]
        if ('s1theta' not in new_samps) or keep_new_spins:
            new_samps['s1theta'] = new_cols[:, 2]
        if ('s2theta' not in new_samps) or keep_new_spins:
            new_samps['s2theta'] = new_cols[:, 3]
        if ('s1' not in new_samps) or keep_new_spins:
            new_samps['s1'] = new_cols[:, 5]
        if ('s2' not in new_samps) or keep_new_spins:
            new_samps['s2'] = new_cols[:, 6]
    return new_samps

def samples_add_antenna_response(old_samps, likelihood_obj):
    for i, d in enumerate(likelihood_obj.det_names):
        old_samps[f'fplus_{d}'], old_samps[f'fcross_{d}'] = \
            np.transpose([likelihood_obj.get_Fplus_Fcross(ra, dec, psi, i_det=i) \
                          for ra, dec, psi in old_samps[['ra', 'dec', 'psi']].to_numpy()])
        old_samps[f'antenna_{d}'] = old_samps[f'fplus_{d}'].to_numpy()**2 + old_samps[f'fcross_{d}'].to_numpy()**2
    return

def samples_add_cosmo_prior(old_samps):
    z, DL = old_samps['z'], old_samps['DL']
    old_samps['cosmo_prior'] = (1+z)**-4 * (1 - DL/(1+z)*dz_dDL(DL))
    return


#### DEFAULTS ****
# base directory
DEFAULT_BASE_DIRS = [f'/data/{usr}/GW/gw_pe/' for usr in ['srolsen', 'jroulet', 'hschia']]
# DEFAULT IAS APPROXIMANTS
DEFAULT_PE_APPROXIMANTS = {'IMR': {'A22': 'IMRPhenomD',
                                   'AHM': 'IMRPhenomHM',
                                   'P22': 'IMRPhenomPv2',
                                   'PHM': 'IMRPhenomXPHM'},
                           'EOB': None,
                           'SUR': {'AHM': 'NRHybSur3dq8',
                                   'PHM': 'NRSur7dq4'}}
DEFAULT_PE_MDL = 'IMR'
DEFAULT_PE_SLM = 'PHM'
DEFAULT_PE_PRIOR = 'flat'

DEFAULT_LIGO_RUN = 'O3'
def run_from_evname(evname):
    if evname[:3] in ['GW1', 'GW2']:
        event_yr_mo = int(evname[2:6])
    elif evname[:4] in ['GWC1', 'GWC2']:
        event_yr_mo = int(evname[3:7])
    elif evname.isnumeric() and 1e9 < int(evname) < 2e9:
        # It's a gps time
        return utils.get_run(int(evname))
    else:
        raise RuntimeError(f'evname = {evname} is not a recognized format')
    return ('O3' if event_yr_mo > 1802 else
            ('O2' if event_yr_mo > 1605 else 'O1'))

def evname_dirpost_from_dir(pe_dir):
    dirlist = [dnm for dnm in pe_dir.split('/') if dnm != '']
    parts = [s for s in dirlist[-1].split('_') if s != '']
    evname, dir_post, post_start = parts[0], '', 1
    if len(parts) > 1:
        if (len(parts[1]) == 6) and parts[1].isnumeric():
            post_start += 1
            evname += '_' + parts[1]
        dir_post = ''
        if len(parts) > post_start:
            for xtra in parts[post_start:]:
                dir_post += '_' + xtra
    return evname, dir_post

####################################### **** UPDATE coming
# will update these once ParameterEstimationPrecessingSpinsHM
#  & ParameterEstimationPrecessingSpinsHMLVCPrior are used
PE_CLASSES = {
    'flat': {
        'A22': gw_pe.ParameterEstimationAlignedSpins,
        'AHM': gw_pe.ParameterEstimationAlignedSpinsHM,
        'P22': gw_pe.ParameterEstimationPrecessingSpins,
        'PHM': gw_pe.ParameterEstimationPrecessingSpins
    },
    'iso': {
        'A22': gw_pe.ParameterEstimationAlignedSpinsLVCPrior,
        'AHM': gw_pe.ParameterEstimationAlignedSpinsHMLVCPrior,
        'P22': gw_pe.ParameterEstimationPrecessingSpinsLVCPrior,
        'PHM': gw_pe.ParameterEstimationPrecessingSpinsLVCPrior
    },
    'tidal': {'A22': gw_pe.ParameterEstimationTidallyAlignedSpin},
    'fixskyflat': {'AHM': gw_pe.ParameterEstimationAlignedSpinsHMFixedSkyLoc,
                   'PHM': gw_pe.ParameterEstimationPrecessingSpinsFixedSkyLoc},
    'fixskyiso': {'PHM': gw_pe.ParameterEstimationPrecessingSpinsFixedSkyLocLVCPrior},
    'nitzflat': {'PHM': gw_pe.ParameterEstimationPrecessingSpinsNitz},
    'nitziso': {'PHM': gw_pe.ParameterEstimationPrecessingSpinsNitzLVCPrior},
    'flatV2': {'PHM': gw_pe.ParameterEstimationPrecessingSpinsV2},
    'isoV2': {'PHM': gw_pe.ParameterEstimationPrecessingSpinsLVCPriorV2},
    'nitzflatV2': {'PHM': gw_pe.ParameterEstimationPrecessingSpinsNitzV2},
    'nitzisoV2': {'PHM': gw_pe.ParameterEstimationPrecessingSpinsNitzLVCPriorV2},
    'flatV3': {'PHM': gw_pe.ParameterEstimationPrecessingSpinsV3},
    'isoV3': {'PHM': gw_pe.ParameterEstimationPrecessingSpinsLVCPriorV3},
    'nitzflatV3': {'PHM': gw_pe.ParameterEstimationPrecessingSpinsNitzV3},
    'nitzisoV3': {'PHM': gw_pe.ParameterEstimationPrecessingSpinsNitzLVCPriorV3}
}

def get_samples_path(evname=None, cls=None, base_dir=None, samples_dir=None,
                     prior=DEFAULT_PE_PRIOR, slm=DEFAULT_PE_SLM, full_dir=None,
                     dir_post='', fname=None, directory_only=False):
    """
    get path to IAS samples = full_dir = os.path.join(base_dir, samples_dir, cls.__name__ + f'_{evname}_samples.pkl')
    where base_dir can be passed or all from DEFAULT_BASE_DIRS will be searched
    and samples_dir = <'aligned' OR 'precessing'> + '_' + <'flatchieff' OR 'isotropic'> < + '_HM'>
                        + '/{evname}' < + dir_post>
        can be passed or inferred from prior, slm, evname (must pass dir_post if necessary)
    and cls is the class from gw_pe, which can be passed or inferred from prior, slm
    :param evname: str 'GWYYYYMMDD' (may include additional '_HHMMSS') specifying event
    :param prior: str specifying prior, e.g. 'flat', 'iso', 'fixskyflat', etc. (see ALL_PRIORS)
    :param slm: str (3 chars) specifying spin physics ('A' for aligned, 'P' for precessing) and
        mode physics ('22' for 22-only, 'HM' for higher-order multipole modes)
        --> choose from 'A22', 'AHM', 'P22', 'PHM'
    :param dir_post: str suffix of PE directory name after evname (i.e., full_dir = ... + '/' + evname + dir_post)
    :param cls: (optional) PE class (default: infer from prior, slm)
    :param full_dir: (optional) directory containing samples (default: infer from evname, prior, slm & try default base dirs)
    :param base_dir: (optional) absolute path of base directory containing samples_dir (default: try all from DEFAULT_BASE_DIRS)
    :param samples_dir: (optional) relative path to samples directory from base_dir (default: infer from evname, prior, slm)
    :param fname: (optional) name of file to search for for validating directory (default: infer from evname, prior, slm)
    :param directory_only: bool flag to return only the directory path (default False: return the file path)
    """
    prior, slm = get_prior_slm(prslm=None, split_slm=False, prior=prior, slm=slm, all_priors=ALL_PRIORS)
    if fname is None:
        if cls is None:
            cls = PE_CLASSES[prior][slm]
        fname = cls.__name__ + f'_{evname}_samples.pkl'
    if full_dir is None:
        if samples_dir is None:
            samples_dir = ('precessing_' if 'p' in slm.lower() else 'aligned_')
            samples_dir += ('flatchieff' if 'flat' in prior.lower() else 'isotropic')
            samples_dir += ('_HM/' if 'hm' in slm.lower() else '/')
            samples_dir += evname + dir_post
            if samples_dir[-1] != '/':
                samples_dir += '/'
        if base_dir is None:
            valid_full_dirs = []
            for fulldir in [os.path.join(dbd, samples_dir) for dbd in DEFAULT_BASE_DIRS]:
                if os.path.exists(os.path.join(fulldir, fname)):
                    valid_full_dirs.append(fulldir)
            nresults = len(valid_full_dirs)
            if nresults < 1:
                print(f'No samples paths exist for default base directories with relative path =')
                print(os.path.join(samples_dir, fname))
                return None
            elif nresults > 1:
                print('Multiple valid samples paths for default base directories')
                print(f'--> by priority, selecting {os.path.join(valid_full_dirs[0], fname)}')
                print(f'-- other valid paths: {[os.path.join(d, fname) for d in valid_full_dirs[1:]]}')
            full_dir = valid_full_dirs[0]
        else:
            full_dir = os.path.join(base_dir, samples_dir)
    return (full_dir if directory_only else os.path.join(full_dir, fname))


OLD_LVC_SAMPLES_DIR = '/data/jroulet/GW/gw_pe/lvc_samples/'
NEW_LVC_SAMPLES_DIR = '/data/bzackay/GW/LSC_PE_samples/'
DEFAULT_LVC_SAMPLES_BASE_DIRS = [OLD_LVC_SAMPLES_DIR, NEW_LVC_SAMPLES_DIR]

utils_LSC_PE_fname = lambda evname: f'{evname}.h5'
utils_LSC_PE_comoving_fname = lambda evname: f'{evname}_comoving.h5' 
############ just fname below as above? or since dir is part of convention after base_dir?
o3special_LSC_PE_fname = lambda evname: f'{evname}/posterior_samples.h5'
o2_LSC_PE_fname = lambda evname: f'{evname}/{evname}_GWTC-1.hdf5'

def utils_LSC_PE_path(evname, base_dir=None):
    return os.path.join((NEW_LVC_SAMPLES_DIR if base_dir is None else base_dir), utils_LSC_PE_fname(evname))
def utils_LSC_PE_comoving_path(evname, base_dir=None):
    return os.path.join((NEW_LVC_SAMPLES_DIR if base_dir is None else base_dir), utils_LSC_PE_comoving_fname(evname)) 
def o3special_LSC_PE_path(evname, base_dir=None):
    return os.path.join((OLD_LVC_SAMPLES_DIR if base_dir is None else base_dir), o3special_LSC_PE_fname(evname))
def o2_LSC_PE_path(evname, base_dir=None):
    return os.path.join((OLD_LVC_SAMPLES_DIR if base_dir is None else base_dir), o2_LSC_PE_fname(evname))

utils_LSC_PE_prior_fname = lambda evname: f'{evname}_prior.npy'
def utils_LSC_PE_prior_path(evname, base_dir=None):
    return os.path.join((NEW_LVC_SAMPLES_DIR if base_dir is None else base_dir), utils_LSC_PE_prior_fname(evname))

LVC_PE_path_of_evname_funcs = [o2_LSC_PE_path, o3special_LSC_PE_path, utils_LSC_PE_path,
                               utils_LSC_PE_comoving_path]

LVC_PE_fname_of_evname_funcs = [o2_LSC_PE_fname, o3special_LSC_PE_fname, utils_LSC_PE_fname,
    utils_LSC_PE_comoving_fname, lambda evname: 'posterior_samples.h5', lambda evname: f'{evname}_GWTC-1.hdf5']

########################################################################
########################################################################
### USE os.path.exists() or something like that to use ^^ those in vv there
def get_lvc_sample_path(evname, samples_base_dir=None, comoving=False):
    """
    get path to LVC samples on IAS compute servers from event name
    :param evname: str 'GWYYYYMMDD' (may include additional '_HHMMSS') specifying event
    :param samples_base_dir: (optional) absolute path of base directory containing samples directory
            (default: try all from DEFAULT_LVC_SAMPLES_BASE_DIRS)
    :param comoving: bool flag to get comoving samples (default: False)
    """
    valid_paths = []
    for func in LVC_PE_path_of_evname_funcs:
        fnm = func(evname, base_dir=samples_base_dir)
        if os.path.exists(fnm):
            valid_paths.append(fnm)
    comoving_paths = [fnm for fnm in valid_paths if 'comov' in fnm.lower()]
    valid_paths = (comoving_paths if comoving else [fnm for fnm in valid_paths if fnm not in comoving_paths])
    nresults = len(valid_paths)
    if nresults < 1:
        printstr = 'No LVC ' + ('comoving ' if comoving else '') + 'samples paths exist for inputs'
        print(f'{printstr} evname={evname}, samples_base_dir={samples_base_dir}')
        return None
    elif nresults > 1:
        print(f'Multiple valid LVC sample paths for inputs evname={evname}, samples_base_dir={samples_base_dir}')
        print(f'--> by priority, selecting {valid_paths[0]}')
        print(f'-- other valid paths: {valid_paths[1:]}')
    return valid_paths[0]

# DEFAULT LVC APPROXIMANTS
DEFAULT_LVC_APPROXIMANTS = {'IMR': {'A22': 'IMRPhenomD',
                                    'AHM': 'IMRPhenomHM',
                                    'P22': 'IMRPhenomPv2',
                                    'PHM': 'IMRPhenomPv3HM'},
                            'EOB': {'A22': 'SEOBNRv4',
                                    'AHM': 'SEOBNRv4HM',
                                    'P22': 'SEOBNRv4P',
                                    'PHM': 'SEOBNRv4PHM'},
                            'SUR': {'A22': None,
                                    'AHM': 'NRHybSur3dq8',
                                    'P22': None,
                                    'PHM': 'NRSur7dq4'}}

DEFAULT_LVC_KEYS_DICT = {'m1': 'mass_1',  'm2': 'mass_2', 'q': 'mass_ratio',  'mtot': 'total_mass',
    's1x': 'spin_1x',  's1y': 'spin_1y',  's1z': 'spin_1z', 's2x': 'spin_2x',  's2y': 'spin_2y',  's2z': 'spin_2z',
    'ra': 'ra',  'dec': 'dec',  'psi':  'psi', 'vphi': 'phase',  'iota': 'iota',  'DL': 'luminosity_distance',
    'z': 'redshift', 'mchirp':  'chirp_mass',  'chieff':  'chi_eff', 'mchirp_source': 'chirp_mass_source',
    'm1_source': 'mass_1_source',  'm2_source': 'mass_2_source', 'mtot_source': 'total_mass_source',
    's1phi': 'phi_1',  's2phi': 'phi_2', 's1costheta': 'cos_tilt_1',  's2costheta': 'cos_tilt_2',
    'thetaJN': 'theta_jn', 'cosiota': 'cos_iota', 's1theta': 'tilt_1',  's2theta': 'tilt_2',
    'phiJL': 'phi_jl', 'phi12': 'phi_12', 'lnL': 'log_likelihood', 's1': 'a_1', 's2': 'a_2',
    'chiperp': None, 'cums1r_s1z': None, 'cums2r_s2z': None, 'costhetaligo': None, 'philigo': None, 'chia': None}

DEFAULT_LVC_KEYMAP_O3 = {'m1': 'mass_1',  'm2': 'mass_2', 'q': 'mass_ratio',  'mtot': 'total_mass',
    's1x': 'spin_1x',  's1y': 'spin_1y',  's1z': 'spin_1z', 's2x': 'spin_2x',  's2y': 'spin_2y',  's2z': 'spin_2z',
    'ra': 'ra',  'dec': 'dec',  'psi':  'psi', 'vphi': 'phase',  'iota': 'iota',  'DL': 'luminosity_distance',
    'z': 'redshift', 'mchirp':  'chirp_mass',  'chieff':  'chi_eff', 'mchirp_source': 'chirp_mass_source',
    'm1_source': 'mass_1_source',  'm2_source': 'mass_2_source', 'mtot_source': 'total_mass_source',
    's1phi': 'phi_1',  's2phi': 'phi_2', 's1costheta': 'cos_tilt_1',  's2costheta': 'cos_tilt_2',
    'thetaJN': 'theta_jn', 'cosiota': 'cos_iota', 's1theta': 'tilt_1',  's2theta': 'tilt_2',
    'phiJL': 'phi_jl', 'phi12': 'phi_12', 'lnL': 'deltalogL', 'lnL_H1': 'deltaloglH1', 'lnL_L1': 'deltaloglL1',
    'lnL_V1': 'deltaloglV1'}

DEFAULT_LVC_KEYMAP_O2 = {
        'm1': 'm1_detector_frame_Msun', 'm2': 'm2_detector_frame_Msun', 'DL': 'luminosity_distance_Mpc',
        'ra': 'right_ascension', 'dec': 'declination', 's1': 'spin1', 's2': 'spin2',
        'costilt1': 'costilt1', 'costilt2': 'costilt2', 'costhetaJN': 'costheta_jn'}

#######################################################################

##############################################
#### GROUPING WAVEFORM MODEL APPROXIMANTS ####
##############################################
# list all approximants by SLM code
SLM_APPROX_LISTS = {
    'A22': ['IMRPhenomD', 'SEOBNRv4_ROM', 'SEOBNRv4'],
    'AHM': ['IMRPhenomHM', 'SEOBNRv4HM_ROM', 'NRHybSur3dq8', 'SEOBNRv4HM'],
    'P22': ['IMRPhenomPv2', 'IMRPhenomPv3', 'SEOBNRv3', 'SEOBNRv4P', 'PrecessingSpinIMR'],
    'PHM': ['IMRPhenomXPHM', 'IMRPhenomPv3HM', 'SEOBNRv4PHM', 'NRSur7dq4',
            'PrecessingSpinIMRHM', 'combinedPHM']}

# list all approximants by model series
MDL_APPROX_LISTS = {
    'IMR': ['IMRPhenomXPHM', 'IMRPhenomPv3HM', 'IMRPhenomHM', 'PrecessingSpinIMRHM',
            'IMRPhenomPv3', 'IMRPhenomPv2', 'PrecessingSpinIMR', 'IMRPhenomD'],
    'EOB': ['SEOBNRv4PHM', 'SEOBNRv4HM_ROM', 'SEOBNRv4_ROM',
            'SEOBNRv3', 'SEOBNRv4P', 'SEOBNRv4HM', 'SEOBNRv4'],
    'SUR': ['NRSur7dq4', 'NRHybSur3dq8']}

# narrow down to approximants of a given MDL and SLM 
def mdl_slm_approxlist(mdl='IMR', slm='PHM', extra=None):
    ret_list = [apr for apr in MDL_APPROX_LISTS[mdl] if apr in SLM_APPROX_LISTS[slm]]
    if type(extra) == str:
        return [apr for apr in ret_list if extra in apr]
    else:
        return ret_list

def approx_to_mdl(approximant):
    if 'IMR' in approximant:
        return 'IMR'
    elif 'SEOBNR' in approximant:
        return 'EOB'
    elif ('NRSur' in approximant) or ('NRHybSur' in approximant):
        return 'SUR'
    elif approximant.lower() in ['combined', 'overall', 'publicationsamples', 'prior', 'priors']:
        return approximant
    else:
        raise ValueError(f'{approximant} not associated to recognized model')

def approx_to_slm(approximant):
    if 'PHM' in approximant:
        return 'PHM'
    elif 'Sur' in approximant:
        return ('AHM' if 'Hyb' in approximant else 'PHM')
    else:
        lm = ('HM' if 'HM' in approximant else '22')
        s = 'A'
        if ('recess' in approximant) or ('Pv' in approximant) or (approximant[-1] == 'P'):
            s = 'P'
        return s + lm

###########################################################
### IMPORTANT VARIABLE INDICATORS
## S : spin physics = 'A' (aligned) or 'P' (precessing)
## LM : mode physics = '22' (22-only) or 'HM' (higher modes)
## SLM : spin+mode code (S+LM) = 'A22' or 'AHM' or 'P22' or 'PHM'
## PRIOR: spin prior = 'flat' (flat chie_eff) or 
##          'iso' (isotropic spins) or 'tidal' (tidally aligned spin-up)
##    --> these all assume geometric distance/angle and LVC mass priors,
##   to use fixed sky prepend 'fixsky' to whichever spin prior, and similarly
##   to use the uniform mtot, q1 as in Nitz GW190521, prepend 'nitz'
## PRSLM: (aka prior_code) = PRIOR + '_' + SLM
## MDL: waveform model series = 'IMR' or 'EOB' or 'SUR'

PRIOR_LIST = ['flat', 'iso', 'tidal', 'fixskyflat', 'fixskyiso',
              'nitzflat', 'nitziso', 'flatV2', 'isoV2', 'nitzflatV2',
              'nitzisoV2', 'flatV3', 'isoV3', 'nitzflatV3', 'nitzisoV3']
S_LIST = ['A', 'P']
LM_LIST = ['22', 'HM']
SLM_LIST = ['A22', 'AHM', 'P22', 'PHM']
MDL_LIST = ['IMR', 'EOB', 'SUR']

# aliases for priors -- in matching these, can use
# any combo of upper/lowercase and can include any number of '_'
FLAT_PRIORS = ['flat', 'flatchieff', 'ias']
ISO_PRIORS = ['iso', 'isotropic', 'lvc', 'ligo', 'isospin', 'isotropicspin']
TIDAL_PRIORS = ['tidal', 'tdl', 'tidalspin', 'tidalalign', 'tidallyaligned']
FIXSKY_FLAT_PRIORS = ['fixskyflat', 'fixskyflatchieff', 'fixskyias', 'flatfixsky',
                      'flatchiefffixsky', 'iasfixsky']
FIXSKY_ISO_PRIORS = ['fixskyiso', 'fixskyisotropic', 'fixskylvc', 'isofixsky',
                      'isotropicfixsky', 'lvcfixsky']
NITZ_FLAT_PRIORS = ['nitzflat', 'nitzflatchieff', 'nitzias', 'flatnitz',
                      'flatchieffnitz', 'iasnitz']
NITZ_ISO_PRIORS = ['nitziso', 'nitzisotropic', 'nitzlvc', 'isonitz',
                      'isotropicnitz', 'lvcnitz']
FLAT_V2_PRIORS = [k + 'v2' for k in FLAT_PRIORS]
ISO_V2_PRIORS = [k + 'v2' for k in ISO_PRIORS]
NITZ_FLAT_V2_PRIORS = [k + 'v2' for k in NITZ_FLAT_PRIORS]
NITZ_ISO_V2_PRIORS = [k + 'v2' for k in NITZ_ISO_PRIORS]
FLAT_V3_PRIORS = [k + 'v3' for k in FLAT_PRIORS]
ISO_V3_PRIORS = [k + 'v3' for k in ISO_PRIORS]
NITZ_FLAT_V3_PRIORS = [k + 'v3' for k in NITZ_FLAT_PRIORS]
NITZ_ISO_V3_PRIORS = [k + 'v3' for k in NITZ_ISO_PRIORS]

ALL_PRIORS = {'flat': FLAT_PRIORS, 'iso': ISO_PRIORS, 'tidal': TIDAL_PRIORS,
              'fixskyflat': FIXSKY_FLAT_PRIORS, 'fixskyiso': FIXSKY_ISO_PRIORS,
              'nitzflat': NITZ_FLAT_PRIORS, 'nitziso': NITZ_ISO_PRIORS,
              'flatV2': FLAT_V2_PRIORS, 'isoV2': ISO_V2_PRIORS,
              'nitzflatV2': NITZ_FLAT_V2_PRIORS, 'nitzisoV2': NITZ_ISO_V2_PRIORS,
              'flatV3': FLAT_V3_PRIORS, 'isoV3': ISO_V3_PRIORS,
              'nitzflatV3': NITZ_FLAT_V3_PRIORS, 'nitzisoV3': NITZ_ISO_V3_PRIORS}

###################### NOTE: prslm is the shorthand for prior_code = 'prior_SLM'
######################     AND may contain alias, so could be  'priorAlias_SLM'
def get_prior_slm(prslm=None, split_slm=False, prior=None, slm=None,
                  all_priors=ALL_PRIORS):
    """
    get (prior, slm) in (all_priors.keys(), {'A22', 'AHM', 'P22', 'PHM'}) 
    from prslm = f'{priorAlias}_{slm}' (see all_priors for aliases, default=ALL_PRIORS)
     or from prior='{priorAlias}' 
    """
    if (prior is None) or (slm is None):
        prior, slm = prslm[:-4], prslm[-3:]
    for pkey, aliases in all_priors.items():
        if (prior.lower()).replace('_', '') in aliases:
            prior = pkey
    if isinstance(slm, str):
        slm = slm.upper()
    else:
        print(f"WARNING: invalid slm = {slm}, choose from 'A22', 'AHM', 'P22', 'PHM'")
    if split_slm is True:
        return prior, slm[0], slm[1:]
    else:
        return prior, slm

#################################################### CLASS from PRSLM
def prslm_cls(prslm=None, prior=None, slm=None,
              pe_cls_dict=PE_CLASSES, all_priors=ALL_PRIORS):
    """get pe class from prslm"""
    prior, slm = get_prior_slm(prslm=prslm, prior=prior, slm=slm,
                               all_priors=all_priors)
    return pe_cls_dict[prior][slm]

# pe filename conventions within pe directory
def pe_fname(pe_cls, evname, ftype='j'):
    """get pe file name by type code"""
    if ftype in ['j', 'json', '.json', 'config', 'metadata', 'init']:
        return f'{pe_cls.__name__}_{evname}.json'
    elif ftype in ['s', 'samples', 'pkl', '.pkl', 'p']:
        return f'{pe_cls.__name__}_{evname}_samples.pkl'
    elif ftype in ['e', 'ev', 'evidence', 'logev']:
        return f'{pe_cls.__name__}_{evname}_evidence.json'
    elif ftype in ['rb', 'RB', 'rbtest', 'RBtest', 'rb_test', 'RB_test']:
        return f'{pe_cls.__name__}_{evname}_lnL_fft_rb_allsamples.npy'
    elif ftype in ['mcmc', 'MCMC', 'mcmctest', 'MCMCtest', 'mcmc_test', 'MCMC_test']:
        raise RuntimeError(f'MCMC test recording is not implemented yet')
    else:
        raise ValueError(f'ftype = {ftype} is not (yet) a valid code')


class ParameterEstimationHandle(object):
    """class for handling PE objects for plotting and comparison"""
    # printing names of priors
    PRIOR_NAMES = {'flat':  'uniform chi_eff',
                   'iso':  'isotropic spins',
                   'tidal':  'tidal spin-up',
                   'fixskyflat': 'fixed sky (ra, dec), uniform chi_eff',
                   'fixskyiso': 'fixed sky (ra, dec), isotropic spins',
                   'nitzflat': 'uniform q1 & mtot, uniform chi_eff',
                   'nitziso': 'uniform q1 & mtot, isotropic spins'}
    # add the V2 angle classes to names
    for prV1 in ['flat', 'iso', 'nitzflat', 'nitziso']:
        PRIOR_NAMES[prV1 + 'V2'] = PRIOR_NAMES[prV1] + ', V2 angles'
        PRIOR_NAMES[prV1 + 'V3'] = PRIOR_NAMES[prV1] + ', V3 angles'
    assert all([k in ALL_PRIORS.keys() for k in PRIOR_NAMES.keys()]), \
        'need to expand global ALL_PRIORS to include all priors in self.PRIOR_NAMES'

    # printing names of SLM = spin-alignment and multipole modes
    S_NAMES = {'A': 'aligned', 'P': 'precessing'}
    LM_NAMES = {'22': '22-only', 'HM': 'higher modes'}
    SLM_NAMES = {'A22': S_NAMES['A'] + ' ' + LM_NAMES['22'],
                 'AHM': S_NAMES['A'] + ' ' + LM_NAMES['HM'],
                 'P22': S_NAMES['P'] + ' ' + LM_NAMES['22'],
                 'PHM': S_NAMES['P'] + ' ' + LM_NAMES['HM']}
    
    # labeling plots with priors
    PRIOR_LABELS = {'flat': '$\\chi_{eff} \\sim U(-1, 1)$',
                    'iso': 'isotropic $\\vec{\\chi}_{1, 2}$',
                    'tidal': '$\\chi_1^z = 0$, $\\chi_2^z \\sim U(0, 1)$',
                    'fixskyflat': 'fixed sky ($\\alpha$, $\\delta$), $\\chi_{eff} \\sim U(-1, 1)$',
                    'fixskyiso': 'fixed sky ($\\alpha$, $\\delta$), isotropic $\\vec{\\chi}_{1, 2}$',
                    'nitzflat': 'uniform $\\frac{m_1}{m_2}$ & $M_{tot}$, $\\chi_{eff} \\sim U(-1, 1)$',
                    'nitziso': 'uniform $\\frac{m_1}{m_2}$ & $M_{tot}$, isotropic $\\vec{\\chi}_{1, 2}$'}
    # add the V2 angle classes to labels
    for prV1 in ['flat', 'iso', 'nitzflat', 'nitziso']:
        PRIOR_LABELS[prV1 + 'V2'] = PRIOR_LABELS[prV1] + ', V2 angles'
        PRIOR_LABELS[prV1 + 'V3'] = PRIOR_LABELS[prV1] + ', V3 angles'
    assert all([k in ALL_PRIORS.keys() for k in PRIOR_LABELS.keys()]), \
        'need to expand global ALL_PRIORS to include all priors in self.PRIOR_LABELS'

    # labeling plots with S = spin-alignment and LM = multipole modes
    S_LABELS = {'A': '$\\chi_{1, 2}^{x, y} = 0$', 'P': '$\\chi_{1, 2}^{x, y} \\in [-1, 1]$'}
    LM_LABELS = {'22': '$(l, |m|) = (2, 2)$', 'HM': '$l \\geq 2, |m| \\leq l$'}
    SLM_LABELS = {'A22': S_LABELS['A'] + '; ' + LM_LABELS['22'],
                  'AHM': S_LABELS['A'] + '; ' + LM_LABELS['HM'],
                  'P22': S_LABELS['P'] + '; ' + LM_LABELS['22'],
                  'PHM': S_LABELS['P'] + '; ' + LM_LABELS['HM']}
    
    # DIRECTORY NAMING
    S_DNAMES = {'A': 'aligned', 'P': 'precessing'}
    PRIOR_DNAMES = {'flat': '_flatchieff', 'iso': '_isotropic', 'tidal': '_tidal',
                    'nitzflat': '_flatchieff', 'nitziso': '_isotropic',
                    'fixskyflat': '_flatchieff', 'fixskyiso': '_isotropic'}
    # add the V2 angle classes to directory suffix 
    for prV1 in ['flat', 'iso', 'nitzflat', 'nitziso']:
        PRIOR_DNAMES[prV1 + 'V2'] = PRIOR_DNAMES[prV1]
        PRIOR_DNAMES[prV1 + 'V3'] = PRIOR_DNAMES[prV1]
    assert all([k in ALL_PRIORS.keys() for k in PRIOR_DNAMES.keys()]), \
        'need to expand global ALL_PRIORS to include all priors in self.PRIOR_DNAMES'

    LM_DNAMES = {'22': '', 'HM': '_HM'}
    
    def __init__(self, prslm=None, dir_post='', base_dir=None,
                 evname=None, mdl='IMR', pe_cls_dict=PE_CLASSES,
                 load_samples=True, load_evidence=True, load_pe=False,
                 pe_approx_dict=DEFAULT_PE_APPROXIMANTS, approximant=None,
                 name_pre='', name_post=None, label_pre='', label_post=None,
                 prior=None, slm=None, s=None, lm=None, priorAlias=None,
                 report_rb_test=False, report_mcmc_test=False, convert_angles=False,
                 compute_antenna=False, compute_cosmo_prior=False,
                 compute_lnL=False, compute_lnPrior=False,
                 gwpdic_keymap=gw_utils.PARKEY_MAP, name=None, keep_new_spins=False):
        """
        IF FOLLOWING IAS PE DIRECTORY CONVENTIONS, easiest to initialize with:
                ParameterEstimationHandle.from_dir(absosulte_path_to_pe_directory)
        ...OTHERWISE unexpected behavior may occur
        for constructor initialization:
        --> in order of reverse priority (i.e., will first look at prslm to derive others, then upward)
        s : spin physics 'A' (aligned) or 'P' (precessing)
        lm : mode physics '22' (22-only) or 'HM' (higher modes)
        slm : spin+mode code (S+LM) 'A22' or 'AHM' or 'P22' or 'PHM'
        priorAlias: one of the listed aliases in <PRIOR>_PRIORS for correcponding PRIOR
        prior: spin prior = 'flat' (flat chie_eff) or   'iso' (isotropic spins)
                        or  'tidal' (tidally aligned spin-up)
        mdl: waveform model series 'IMR' or 'EOB' or 'SUR'
        prslm: (aka prior_code with possible alias) PRIOR + '_' + SLM
        base_dir should be /data/srolsen/GW/gw_pe/ or /data/jroulet/GW/gw_pe/ or /data/bzackay/GW/gw_pe/
        --> PE directory will be inside one of the subdirectories of base_dir indicating the spin prior and
         modes included (this is the information in prslm / prior+slm), and the directory names by prslm are
         flat_A22: aligned_flatchieff
         iso_A22: aligned_isotropic
         flat_AHM: aligned_flatchieff_HM
         iso_AHM: aligned_isotropic_HM
         flat_P22: precessing_flatchieff
         iso_P22: precessing_isotropic
         flat_PHM: precessing_flatchieff_HM
         iso_PHM: precessing_isotropic_HM
        --> the PE directory will have the absolute path obtained from:
                os.path.join(base_dir, sub_dir[prslm], evname + dir_post)
        --> the PE class will be cls = pe_cls_dict[prior][slm]
        --> inside the PE directory the following files will be expected:
            always:
                cls.__name__ + '_' + evname + '.json'
                (pe only loaded is load_pe=True, but always need to collect info from json file)
            if load_samples=True:
                cls.__name__ + '_' + evname + '_samples.pkl'
            if load_evidence=True:
                cls.__name__ + '_' + evname + '_evidence.json'
        :param evname: str 'GWYYYYMMDD' (may include additional '_HHMMSS') specifying event
        **for INITIALIZATION** sufficient subsets of (prslm, prior, priorAlias, slm, s, lm) are:
            1. prslm
            2. priorAlias, slm
            3. priorAlias, s, lm
            4. prior, slm
            5. prior, s, lm
        """
        # get other directly set attributes
        self.base_dir = base_dir
        self.evname = evname
        self.obs_run, self.tgps = None, None
        if evname is None:
            print('NO EVENT NAME GIVEN, set evname=GWyymmdd or evname=GWyymmdd_hhmmss')
        else:
            self.obs_run = run_from_evname(self.evname)
            if evname in bookkeeping:
                self.tgps = bookkeeping[evname][0]
                
        self.dir_post = dir_post
        self.mdl = (mdl.upper() if (isinstance(mdl, str) and (len(mdl) == 3)) else mdl)
        self.approximant = approximant
        self.samples, self.pe, self.like = None, None, None
        
        # if not passing prior_code, build from priorAlias (else prior) and slm (else s, lm)
        self.priorAlias = (prior if priorAlias is None else priorAlias)
        if prslm is None:
            if (not isinstance(slm, str)) and isinstance(s, str) and isinstance(lm, str):
                slm = s + lm
            if isinstance(self.priorAlias, str) and isinstance(slm, str):
                prslm = self.priorAlias + '_' + slm.upper()
            else:
                raise RuntimeError('NEED sufficient subset of prslm, priorAlias, prior, slm, s, lm'
                                   + '\n--> see class docstring for list of sufficient subsets')
        # now set prslm and use to set other prslm indicator attributes
        self.prslm = prslm
        if not isinstance(self.priorAlias, str):
            self.priorAlias = self.prslm[:-4]
        self.prior, self.s, self.lm = get_prior_slm(prslm=self.prslm, split_slm=True,
                                                    all_priors=ALL_PRIORS)
        if (self.priorAlias != self.prslm[:-4]) and (self.prior != self.prslm[:-4]):
            print('WARNING: potential mismatch with (prslm, priorAlias, prior) =')
            print(f'\t({self.prslm}, {self.priorAlias}, {self.priorAlias})')
        self.slm = self.s + self.lm
        self.prior_code = self.prior + '_' + self.slm
        # and set class from prslm
        self.cls = prslm_cls(prslm=self.prslm, pe_cls_dict=pe_cls_dict, all_priors=ALL_PRIORS)
        
        # now check if model and approximant were set (consistently)
        if self.approximant is None:
            try:
                self.approximant = pe_approx_dict[self.mdl][self.slm]
            except:
                print(f'APPROXIMANT ERROR: could not be found from mdl={self.mdl}, slm={self.slm}')
        elif self.mdl is None:
            try:
                self.mdl = approx_to_mdl(self.approximant)
            except:
                print(f'MODEL ERROR: could not be found from approximant={self.approximant}')
        else:
            try:
                if self.mdl != approx_to_mdl(self.approximant):
                    print(f'WARNING: model={self.mdl} and approximant={self.approximant} do not match!')
            except:
                print(f'WARNING: cannot associate model to approximant={self.approximant}')
            try:
                if self.approximant != pe_approx_dict[self.mdl][self.slm]:
                    print(f'WARNING: SLM code={self.slm} and approximant={self.approximant} do not match!')
            except:
                print(f'WARNING: pe_approx_dict does not contain keys self.mdl={self.mdl}, self.slm={self.slm}')

        # get directory name
        self.dname = self.S_DNAMES[self.s] + self.PRIOR_DNAMES[self.prior] + self.LM_DNAMES[self.lm]
        self.dir = os.path.join(self.base_dir, self.dname, self.evname + self.dir_post)
        # get file names
        self.json_fname = os.path.join(self.dir, pe_fname(self.cls, self.evname, ftype='j'))
        self.samples_fname = os.path.join(self.dir, pe_fname(self.cls, self.evname, ftype='s'))
        self.evidence_fname = os.path.join(self.dir, pe_fname(self.cls, self.evname, ftype='ev'))

        self.pe, self.like, self.samples, self.Nsamples, self.samples_run_inds, self.samples_slices_per_run, \
          self.log_ev_per_run, self.log_ev_std_per_run, self.log_ev, self.log_ev_std = [None] * 10
        # LOAD PE RESULTS
        if load_pe is True:
            self.load_pe_json()
        else:
            try:
                with open(self.json_fname, 'r') as json_file:
                    json_dic = json.load(json_file)
                self.pe_params = dcopy(json_dic)
                self.tgps = self.pe_params.get('tgps', getattr(self, 'tgps', None))
                self.f_ref = self.pe_params.get('f_ref', None)
                if 'likelihood_kwargs' in self.pe_params:
                    f_ref_likelihood = self.pe_params['likelihood_kwargs'].get('f_ref', None)
                    if self.f_ref is None:
                        self.f_ref = f_ref_likelihood
                    elif f_ref_likelihood is not None:
                        if not np.isclose(self.f_ref, f_ref_likelihood):
                            print(f'WARNING: f_ref in PE object is {self.f_ref}, but',
                                  f'in likelihood it is {f_ref_likelihood}')
                            print(f' --> taking likelihood value')
                            self.f_ref = f_ref_likelihood
            except:
                print(f'LOAD ERROR: {self.json_fname} could not be loaded, no PE information gathered')
                self.pe_params = {}
                self.tgps = getattr(self, 'tgps', None)
                self.f_ref = None
        
        # loading samples and parameter conversion
        if load_samples is True:
            self.load_samples_pkl(convert_angles=convert_angles, compute_antenna=compute_antenna,
                                  compute_cosmo_prior=compute_cosmo_prior, compute_lnL=compute_lnL,
                                  compute_lnPrior=compute_lnPrior, gwpdic_keymap=gwpdic_keymap,
                                  keep_new_spins=keep_new_spins)
        else:
            self.samples = None
            self.Nsamples = 0
        # loading pymultinest evidence
        if load_evidence is True:
            self.load_evidence_json()
        
        # TESTING REPORT
        if report_rb_test is True:
            self.report_rb_test_results()
        if report_mcmc_test is True:
            # IMPLEMENT THIS with test_pymultinest_runs_consistency() results
            print('MCMC TEST REPORT NOT YET IMPLEMENTED')
        
        # set print name and plot label
        self.name = name
        try:
            if self.name is None:
                slmname =('Precession Disabled, ' if 'noprec' in self.dir_post.lower()
                          else self.S_NAMES[self.s]) + ' '
                slmname += ('HM Disabled' if 'nohm' in self.dir_post.lower()
                            else self.LM_NAMES[self.lm])
                if name_post is None:
                    name_post = ('' if self.dir_post == '' else ' ' + self.dir_post)
                self.name = name_pre + self.PRIOR_NAMES[self.prior] + ', ' + slmname + name_post
            slmlab = ('Precession Disabled' if 'noprec' in self.dir_post.lower() else self.S_LABELS[self.s]) + '; '
            slmlab += ('HM Disabled' if 'nohm' in self.dir_post.lower() else self.LM_LABELS[self.lm])
            if label_post is None:
                label_post = ('' if self.dir_post == '' else ' ' + self.dir_post)
            self.label = label_pre + self.PRIOR_LABELS[self.prior] + '; ' + slmlab + label_post
        except:
            print(f'ERROR in label/name construction for prior={self.prior}, s={self.s}, lm={self.lm}')
            if self.name is None:
                self.name = name_pre + self.prslm + self.dir_post + (name_post or '')
            self.label = label_pre + self.prslm + ' ' + self.dir_post + (label_post or '')
        return

    def angle_conversion(self, f_ref=None, keep_new_spins=False):
        if f_ref is None:
            f_ref = self.f_ref
        self.samples = samples_with_ligo_angles(self.samples, f_ref=f_ref,
                                                tgps=self.tgps, keep_new_spins=keep_new_spins)
        return
    
    @classmethod
    def from_dir(cls, pe_dir, mdl='IMR', pe_cls_dict=PE_CLASSES, load_samples=True,
                 load_evidence=True, load_pe=False, approximant=None,
                 pe_approx_dict=DEFAULT_PE_APPROXIMANTS,
                 name_pre='', name_post=None, label_pre='', label_post=None,
                 prior=None, slm=None, s=None, lm=None, priorAlias=None,
                 report_rb_test=False, report_mcmc_test=False, convert_angles=False,
                 compute_antenna=False, compute_cosmo_prior=False,
                 compute_lnL=False, compute_lnPrior=False,
                 gwpdic_keymap=gw_utils.PARKEY_MAP, name=None, keep_new_spins=False):
        # get event name and directory suffix
        evname, dir_post = evname_dirpost_from_dir(pe_dir)
        dirlist = [dnm for dnm in pe_dir.split('/') if dnm != '']
        # get prior, spin, and mode physics from directory name
        dir_s = ('P' if 'precess' in dirlist[-2].lower() else 'A')
        dir_lm = ('HM' if 'hm' in dirlist[-2].lower() else '22')
        dir_prior = ('flat' if 'flat' in dirlist[-2].lower()
                     else ('tidal' if 'tidal' in dirlist[-2].lower()
                           else 'iso'))
        # add version suffix if indicated by directory name
        if '_v2' in dirlist[-1].lower():
            dir_prior += 'V2'
        elif '_v3' in dirlist[-1].lower():
            dir_prior += 'V3'
        # add nitz mass prior prefix if indicated by directory name
        if 'nitz' in dirlist[-1].lower():
            dir_prior = 'nitz' + dir_prior
        # add fixed sky location prefix if indicated by directory name
        if 'sky' in dirlist[-1].lower():
            dir_prior = 'fixsky' + dir_prior
        # combine into prslm code
        prslm = dir_prior + '_' + dir_s + dir_lm
        # make base directory
        base_dir = ('/' + dirlist[0] if pe_dir[0] == '/' else dirlist[0])
        for add_dir in dirlist[1:-2]:
            base_dir = os.path.join(base_dir, add_dir)
        return cls(prslm=prslm, dir_post=dir_post, base_dir=base_dir,
                   evname=evname, mdl=mdl, pe_cls_dict=pe_cls_dict,
                   load_samples=load_samples, load_evidence=load_evidence,
                   load_pe=load_pe, pe_approx_dict=pe_approx_dict,
                   approximant=approximant, name_pre=name_pre, name_post=name_post,
                   label_pre=label_pre, label_post=label_post,
                   prior=prior, slm=slm, s=s, lm=lm, priorAlias=priorAlias,
                   report_rb_test=report_rb_test, report_mcmc_test=report_mcmc_test,
                   convert_angles=convert_angles, compute_antenna=compute_antenna,
                   compute_cosmo_prior=compute_cosmo_prior, compute_lnL=compute_lnL,
                   compute_lnPrior=compute_lnPrior, gwpdic_keymap=gwpdic_keymap,
                   name=name, keep_new_spins=keep_new_spins)


    @classmethod
    def from_evname(cls, evname, approximant=None, dir_post='',
                    prior='flat', slm='PHM', mdl='IMR', prslm=None,
                    load_samples=True, load_evidence=True, load_pe=False,
                    base_dir=None, pe_approx_dict=DEFAULT_PE_APPROXIMANTS,
                    pe_cls_dict=PE_CLASSES, convert_angles=False,
                    name_pre='', name_post=None, label_pre='', label_post=None,
                    compute_antenna=False, compute_cosmo_prior=False,
                    compute_lnL=False, compute_lnPrior=False, report_rb_test=False,
                    report_mcmc_test=False, gwpdic_keymap=gw_utils.PARKEY_MAP,
                    name=None, keep_new_spins=False):
            """initialize PE handle from evname"""
            if approximant is not None:
                mdl = approx_to_mdl(approximant)
                slm = approx_to_slm(approximant)
            pe_dir = get_samples_path(evname=evname, cls=None, base_dir=base_dir, samples_dir=None,
                                      prior=prior, slm=slm, full_dir=None, dir_post=dir_post,
                                      directory_only=True, fname=None)
            return cls.from_dir(pe_dir, mdl=mdl, pe_cls_dict=pe_cls_dict, load_samples=load_samples,
                 load_evidence=load_evidence, load_pe=load_pe, approximant=approximant,
                 pe_approx_dict=pe_approx_dict, convert_angles=convert_angles,
                 name_pre=name_pre, name_post=name_post, label_pre=label_pre, label_post=label_post,
                 prior=None, slm=None, s=None, lm=None, priorAlias=prior,
                 report_rb_test=report_rb_test, report_mcmc_test=report_mcmc_test,
                 compute_antenna=compute_antenna, compute_cosmo_prior=compute_cosmo_prior,
                 compute_lnL=compute_lnL, compute_lnPrior=compute_lnPrior,
                 gwpdic_keymap=gwpdic_keymap, name=name, keep_new_spins=keep_new_spins)
            

    def load_pe_json(self, periodicity_test=False, save_finite_psd=False):
        self.pe = self.cls.from_json(self.json_fname, save_finite_psd=save_finite_psd,
                                     periodicity_test=periodicity_test)
        self.like = self.pe.likelihood
        self.f_ref = self.like.f_ref
        self.tgps = self.like.tgps
        with open(self.json_fname, 'r') as json_file:
            json_dic = json.load(json_file)
        self.pe_params = dcopy(json_dic)
        return
        
    def load_samples_pkl(self, convert_angles=False, compute_antenna=False,
                         compute_cosmo_prior=False, compute_lnL=False, compute_lnPrior=False,
                         gwpdic_keymap=gw_utils.PARKEY_MAP, new_samples_fname=None,
                         recompute_aux_vars=False, gwpdic_completion=False, keep_new_spins=False):
        """load PE samples, with options for conversion/computation and naming conventions"""
        if isinstance(new_samples_fname, str):
            self.samples_fname = new_samples_fname
        self.samples = pd.read_pickle(self.samples_fname)
        # if gw_utils.PARKEY_MAP does not contain gwpdic_keymap, will need to add those keynames later
        orig_samples = None
        if isinstance(gwpdic_keymap, dict) or (gwpdic_keymap is None):
            if not gwpdic.keymap1_contains_keymap2(gw_utils.PARKEY_MAP, gwpdic_keymap):
                orig_samples = self.samples.copy()
        # now start with names from default keymap for doing computations
        self.samples = pd.DataFrame({gw_utils.PARKEY_MAP.get(k, k): np.asarray(self.samples[k])
                                     for k in self.samples.columns})
        # watch out for precession-disabled runs that still have sampling parameters with inplane spin!
        if self.pe_params.get('disable_precession') == True:
            self.samples = self.samples[[k for k in self.samples.columns
                                         if k not in ['cums1r_s1z', 's1phi', 'cums2r_s2z', 's2phi']]]
            if orig_samples is not None:
                orig_samples = orig_samples[[k for k in orig_samples.columns
                                             if k not in ['cums1r_s1z', 's1phi', 'cums2r_s2z', 's2phi']]]
        self.Nsamples = len(self.samples)
        if recompute_aux_vars:
            try:
                gw_utils.compute_samples_aux_vars(self.samples)
            except:
                print('unable to recompute samples aux vars')
        if gwpdic_completion:
            try:
                self.samples = GWPD(self.samples, complete=True, tgps=self.pe_params.get('tgps', None),
                                    f_ref=self.pe_params.get('f_ref', None)).dataframe()
            except:
                print('unable to convert with GWParameterDictionary.completion()')
        if not 'l1' in self.samples.columns:
            self.samples['l1'] = np.zeros(self.Nsamples)
        if not 'l2' in self.samples.columns:
            self.samples['l2'] = np.zeros(self.Nsamples)
        if not 'tc' in self.samples.columns:
            self.samples['tc'] = np.zeros(self.Nsamples)
        if 'fixsky' in self.prior:
            if ('ra' in self.samples.columns) and (np.allclose(self.samples['ra'], self.pe_params['ra']) == False):
                raise RuntimeError('RA of samples is NOT fixed to pe.ra ')
            else:
                self.samples['ra'] = np.full(self.Nsamples, self.pe_params['ra'])
            if ('dec' in self.samples.columns) and (np.allclose(self.samples['dec'], self.pe_params['dec']) == False):
                raise RuntimeError('RA of samples is NOT fixed to pe.ra ')
            else:
                self.samples['dec'] = np.full(self.Nsamples, self.pe_params['dec'])
        if convert_angles:
            try:
                self.angle_conversion(keep_new_spins=keep_new_spins)
            except:
                print('unable to compute f_ref-independent angles')
        if compute_antenna:
            if self.like is not None:
                try:
                    samples_add_antenna_response(self.samples, self.like)
                except:
                    print('unable to compute antenna response')
            else:
                print('cannot add antenna response without loading pe (to get likelihood object)')
            ############################################################
        if compute_lnL:
            if self.like is not None:
                lnL_dets = np.transpose(
                    [self.like.lnlike(par_vals, return_all_detectors=True)
                     for par_vals in self.samples[self.like.params].to_numpy()])
                for det_name, lnL_det in zip(self.like.det_names, lnL_dets):
                    self.samples[f'lnL_{det_name}'] = lnL_det
                self.samples['lnL'] = lnL_dets.sum(axis=0)
            else:
                print('cannot compute likelihoods without loading pe (to get likelihood object)')
        if compute_lnPrior:
            if self.pe is not None:
                try:
                    self.samples['lnPrior'] = np.array([self.pe.lnprior(par_vals) for
                                                        par_vals in self.samples[self.pe.params].to_numpy()])
                except:
                    print('unable to compute prior probablility')
            else:
                print('cannot compute prior without loading pe')
            ############################################################
            
        if compute_cosmo_prior:
            try:
                samples_add_cosmo_prior(self.samples)
            except:
                print('unable to compute cosmological prior weights')
        if 'lnL' in self.samples:
            self.best_pars_ind = np.argmax(self.samples['lnL'])
            self.best_pars = dict(self.samples.iloc[self.best_pars_ind])
            self.best_pars.update({k: self.best_pars.get(k, 0) for k in ['l1', 'l2', 'tc']})
            self.lnL_max = self.best_pars['lnL']
            if 'lnPrior' in self.samples:
                self.samples['lnPosterior'] = self.samples['lnPrior'] + self.samples['lnL']
        else:
            print('WARNING: no lnL in samples')
            self.lnL_max, self.best_pars_ind, self.best_pars = [None]*3

        # if using gwpdic_keymap different from default conventions, add those
        # (or if gwpdic_keymap=None, make sure we keep all original key names)
        if orig_samples is not None:
            getkey = lambda k: k
            if isinstance(gwpdic_keymap, dict):
                getkey = lambda k: gwpdic_keymap.get(k, k)
            for k in orig_samples.columns:
                if not (getkey(k) in self.samples):
                    self.samples[getkey(k)] = np.asarray(orig_samples[k])
        
        # make sure we didn't take any cosines without the angles themselves
        for k in self.samples.columns:
            if (('cos' in k) and (k.replace('cos', '') not in self.samples.columns)
                and (k != 'cosmo_prior')):
                self.samples[k.replace('cos', '')] = np.arccos(self.samples[k])

        ###########################

        self.means, self.medians, self.stds = {}, {}, {}
        for k in self.samples.columns:
            try:
                self.means[k] = np.mean(self.samples[k])
            except:
                self.means[k] = None
            try:
                self.medians[k] = np.median(self.samples[k])
            except:
                self.medians[k] = None
            try:
                self.stds[k] = np.std(self.samples[k])
            except:
                self.stds[k] = None
        return

    def sample_completion(self, new_samples=None, new_samples_fname=None, **gwpdic_kwargs):
        """use GWPD init completion to compute all parameter conversions"""
        # see if passing anything new, otherwise start with existing samples
        if new_samples is None:
            if isinstance(new_samples_fname, str):
                self.samples_fname = new_samples_fname
                new_samples = pd.read_pickle(self.samples_fname)
            else:
                new_samples = getattr(self, 'samples', pd.read_pickle(self.samples_fname))
        # get keyword arguments for GWPD conversion
        # --> default to complete all with f_ref, tgps from PE json
        kws = dcopy(gwpdic_kwargs)
        kws['allow_bad_keys'] = True
        if kws.get('complete', None) is None:
            kws['complete'] = True
        if kws.get('f_ref', None) is None:
            kws['f_ref'] = self.f_ref
        if kws.get('tgps', None) is None:
            kws['tgps'] = self.tgps
        self.samples = GWPD(par_dic=new_samples, **kws).dataframe()
        return
            

    def get_best_samples(self, key_rngs={}, get_best_inds=np.arange(20, dtype=int), lnLmin=0, dataframe=False):
        """
        consider samples in bounds given by key_rngs, and sort them by lnL (descending),
        so the highest SNR sample is at index 0, etc. --> from this ordering, take samples at the indices
        given in get_best_inds and return them as a pd.DataFrame if dataframe=True, otherwise return
        them as a list of dicts
        """
        pdics = get_best_pdics(self.samples, key_rngs=key_rngs, get_best_inds=get_best_inds, lnLmin=lnLmin)
        return (pd.DataFrame(pdics) if dataframe else pdics)

    def load_evidence_json(self):
        with open(self.evidence_fname, 'r') as json_file:
            ev_info_dict = json.load(json_file)
        self.samples_run_inds = ev_info_dict['samples_run_inds']
        self.samples_slices_per_run = [
            slice(*ind_lo_hi) for ind_lo_hi in self.samples_run_inds]
        self.log_ev_per_run = ev_info_dict['log_ev_per_run']
        self.log_ev_std_per_run = ev_info_dict['log_ev_std_per_run']
        self.log_ev = ev_info_dict['log_ev']
        self.log_ev_std = ev_info_dict['log_ev_std']
        self.log_ev_str = f'{self.log_ev:.2f} $\\pm$ {self.log_ev_std:.2f}'
        self.Nsamples_per_run = [inds[1] - inds[0] for inds in self.samples_run_inds]
        return

    def load_rb_test_results(self, run_inds=None):
        run_inds = run_inds or list(range(16))
        # get test run filenames
        res_fnames = [self.cls.__name__ + f'_{self.evname}_lnL_fft_rb_run{irun}.npy'
                      for irun in run_inds]
        self.rb_test_runs = run_inds.copy()
        # load test results if filename exists, else remove it from list
        for irun, fnm in zip(run_inds, res_fnames):
            if fnm in os.listdir(self.dir):
                setattr(self, f'lnL_fft_rb_run{irun}',
                        np.load(os.path.join(self.dir, fnm)))
            else:
                self.rb_test_runs.remove(irun)
        if len(self.rb_test_runs) > 0:
            # concatenate available test results
            self.lnL_fft = np.concatenate([getattr(self, f'lnL_fft_rb_run{irun}')[0]
                                           for irun in self.rb_test_runs])
            self.lnL_rb = np.concatenate([getattr(self, f'lnL_fft_rb_run{irun}')[1]
                                          for irun in self.rb_test_runs])
        else:
            print('No RB test results to load!')
        return
    
    def report_rb_test_results(self, plot=True):
        # load test results if not yet loaded
        if False in [hasattr(self, f'lnL_fft'), hasattr(self, f'lnL_rb')]:
            self.load_rb_test_results()
        if len(self.rb_test_runs) > 0:
            # print fractional error info
            lnLdif = (self.lnL_rb - self.lnL_fft) / self.lnL_fft
            print(f'{self.evname} {self.name}\nrelative binning fractional error has')
            print(f'mean = {np.mean(lnLdif):.3f}, variance = {np.std(lnLdif)**2:.3f}')
            # plot absolute error if indicated
            if plot:
                lnLerr = self.lnL_rb - self.lnL_fft
                plt.figure(figsize=(8, 6))
                plt.title(f'{self.evname} relative binning test:\n{self.name}')
                plt.text(.75, .85, f'{len(lnLerr)} samples:\nmean = ' + \
                         f'{np.mean(lnLerr):.3f}\nvar = {np.std(lnLerr)**2:.3f}',
                         transform=plt.gca().transAxes, size='large')
                plt.scatter(self.lnL_fft, lnLerr, s=1)
                plt.xlabel('log likelihood (computed with FFT)')
                plt.ylabel('relative binning error $ln L_{RB}  -  ln L_{FFT}$')
        else:
            print('No RB test results to report!')
        return
    
    def load_mcmc_test_results(self):
        return
    
    def report_mcmc_test_results(self):
        return

    def get_wf(self, par_dic, whiten=True, time_domain=True, nfft=None, dt=None, f=None):
        if self.pe is None:
            self.pe = self.cls.from_json(self.json_fname)
            self.like = self.pe.likelihood
        h_td = self.like.get_h_td(par_dic, nfft=nfft, dt=dt, f=f, whiten=whiten)
        return (h_td if time_domain else utils.RFFT(h_td, axis=-1))


    def get_f_psd(self, det=None, try_finite_psd=False):
        if self.pe is None:
            print('PE initializing from json')
            self.load_pe_json(periodicity_test=False, save_finite_psd=try_finite_psd)
        if isinstance(det, int) or isinstance(det, np.ndarray):
            det_inds = det
        elif det in [None, 'all']:
            det_inds = np.arange(self.like.ndet, dtype=int)
        elif det in self.like.det_names:
            det_inds = self.like.det_names.index(det)
        else:
            raise ValueError(f'cannot associate detectors with input det = {det}')
        if try_finite_psd:
            try:
                return self.like.f, self.like.psd_fft[det_inds]
            except:
                print('unable to get finite PSD, returning PSD used in sampling')
        return self.like.f, self.like.psd_f[det_inds]

    def get_f_data(self, det=None):
        if self.pe is None:
            print('PE initializing from json')
            self.load_pe_json(periodicity_test=False, save_finite_psd=try_finite_psd)
        if isinstance(det, int) or isinstance(det, np.ndarray):
            det_inds = det
        elif det in [None, 'all']:
            det_inds = np.arange(self.like.ndet, dtype=int)
        elif det in self.like.det_names:
            det_inds = self.like.det_names.index(det)
        else:
            raise ValueError(f'cannot associate detectors with input det = {det}')
        return self.like.f, self.like.data_f[det_inds] * self.like.T

    def corner_plot(self, pvkeys=['m2_source', 'mchirp', 'q', 'chieff'], compare_runs=False,
                    title=None, figsize=(9,7), weights=None, scatter_points=None, grid_kws={},
                    **kwargs):
        """make corner plots"""
        if self.samples is None:
            self.load_samples_pkl()
        
        units, plabs = gw_utils.units, gw_utils.param_labels
        for k in pvkeys:
            if k not in units.keys():
                units[k] = ''
            if k not in plabs.keys():
                plabs[k] = k

        if isinstance(weights, str):
            weights = self.samples[weights]
        ################################################################
        # if comparing runs
        if (compare_runs == True) or hasattr(compare_runs, '__len__'):
            if self.samples_slices_per_run is None:
                self.load_evidence_json()
            if isinstance(compare_runs, bool):
                compare_runs = range(len(self.samples_slices_per_run))
            grids = []
            for j in compare_runs:
                use_weights = None if weights is None else weights[self.samples_slices_per_run[j]]
                nm = f'Run {j} (Ns={self.samples_run_inds[j][1] - self.samples_run_inds[j][0]}), ' \
                    + f'ln(Ev) = {self.log_ev_per_run[j]:.2f} $\\pm$ {self.log_ev_std_per_run[j]:.2f}'
                grids.append(gd.Grid.from_samples(pvkeys, self.samples.iloc[self.samples_slices_per_run[j]],
                                                  pdf_key=nm, units=units, labels=plabs, weights=use_weights,
                                                  **grid_kws))
            if title is None:
                ev, std = self.log_ev, self.log_ev_std
                title = f'{self.evname}: {self.name} PE Runs\nLog Evidence = {ev:.2f} $\\pm$ {std:.2f}' \
                    + f'  from {self.Nsamples} samples'
            return gd.MultiGrid(grids).corner_plot(set_legend=True, title=title, figsize=figsize,
                                                   scatter_points=scatter_points, **kwargs)
        # otherwise use all runs together
        else:
            nm = f'{self.evname}: {self.name}\n{self.Nsamples} samples'
            if None not in [self.log_ev, self.log_ev_std]:
                nm += f', Log Evidence = {self.log_ev:.2f} $\\pm$ {self.log_ev_std:.2f}'
            if title is None:
                title, pdfnm = nm, None
            else:
                pdfnm = nm
            # gd.corner_plot(pdf=None, title=None, subplot_size=2., fig=None, ax=None,
            #    figsize=None, nbins=6, set_legend=True, save_as=None, y_title=.98,
            #    plotstyle=None, show_titles_1d=True, scatter_points=None, **kwargs)
            sg =  gd.Grid.from_samples(pvkeys, self.samples, pdf_key=pdfnm, units=units,
                                       labels=plabs, weights=weights, **grid_kws)
            return sg.corner_plot(pdf=pdfnm, title=title, figsize=figsize,set_legend=True,
                                  scatter_points=scatter_points, **kwargs)


######## CLASS ALIAS ########
PEHAND = ParameterEstimationHandle
########################################################################################

def make_key_combos(k1, k2, post=None):
    """make key combinations"""
    if isinstance(post, str):
        sep = ('' if post[0] == '_' else '_')
        return [k1 + k2 + post, k1 + '_' + k2 + sep + post, (k1, k2, post),
                k2 + k1 + post, k2 + '_' + k1 + sep + post, (k2, k1, post)]
    else:
        return [k1 + k2, k1 + '_' + k2, (k1, k2),
                k2 + k1, k2 + '_' + k1, (k2, k1)]

def expand_nested_keys(dics, post=None, keep_orig=True, new_dics=False, add_lowercase=False):
    """
    for nested dic[k1][k2] or dic[k1][k2][k3] allow access to val using
    all keys in make_key_combos(k1, k2, post) or make_key_combos(k1, k2, k3)
    """
    diclist = ([dics] if isinstance(dics, dict) else dics)
    newdics = []
    for dic in diclist:
        orig_keys = list(dic.keys())
        newdic = (dcopy(dic) if keep_orig is True else {})
        for k1, val in dic.items():
            if isinstance(val, dict):
                for k2, v in val.items():
                    if isinstance(v, dict):
                        for k3, v3 in v.items():
                            for key in make_key_combos(k1, k2, post=k3):
                                newdic[key] = v3
                    elif v is not None:
                        for key in make_key_combos(k1, k2, post=post):
                            newdic[key] = v
        if add_lowercase:
            newdic.update({k.lower(): v for k, v in newdic.items() if isinstance(k, str)})
        if new_dics is True:
            newdics.append(newdic)
        else:
            if keep_orig is not True:
                for k in orig_keys:
                    _ = dic.pop(k)
            dic.update(newdic)
    if new_dics is True:
        return new_dics
    else:
        return

########################################################################
########################################################################

class LVCsampleHandle(object):
    """
    similar to ParameterEstimationHandle but for LVC sample releases
    for O2 posterior sample releases, maybe easier to add these manually to pe_list and plot via index
    when including in ParameterEstimationComparison (can try automated method, updates in progress)
    """
    default_approximant_priority = ['PrecessingSpinIMRHM', 'IMRPhenomPv3HM', 'NRSur7dq4',
                                    'SEOBNRv4PHM', 'IMRPhenomPv2_posterior']
    def __init__(self, lvc_h5, dataset_name=None, evname=None, keymap=None, tgps=None,
                 compute_aux_O3=False, convert_angles=False, gwpdic_keymap=gw_utils.PARKEY_MAP,
                 no_print=False, keep_unmatched_keys=False, try_dataset_variants=True):
        """
        class for handling LVC posterior samples, maybe easier to initialize with cls.from_evname()
        :param lvc_h5: h5py.File(samples_h5_path) or samples_h5_path
        :param dataset_name: name of approximant or dataset to get from LVC file, or if None (default) will use
          default_approximant_priority and if None are present there's an algorithm below to get the closest one
        :param evname: 'GWYYYYMMDD' < + '_HHMMSS'>
        :param tgps: float gps time, only needed if converting angles
        :param keymap: dict of key mappings with IAS keys as keys and LVC keys as values (take default from obs_run)
        :param gwpdic_keymap: dict of key mappings with LVC keys as keys and IAS keys as values (default is good)
        :param compute_O3_aux: bool flag to run compute_samples_aux_vars() on O3 samples (always run on O1&O2)
        :param convert_angles: bool flag to compute f_ref-independent angular variables (need tgps for this)
        """
        if not isinstance(lvc_h5, h5py._hl.files.File):
            try:
                lvc_h5 = h5py.File(lvc_h5, 'r')
            except:
                print(f'ERROR loading LVC samples, lvc_h5={lvc_h5} is not h5py._hl.files.File or valid path')
                return -1
        
        # event and run
        self.evname, self.tgps, self.obs_run = evname, tgps, None
        if evname is None:
            if tgps is not None:
                evn_tdif = [[k, abs(v[0] - tgps)] for k, v in bookkeeping.items()]
                self.evname, tdif = evn_tdif[np.argmin([pair[1] for pair in evn_tdif])]
                if tdif > 2:
                    print(f'NO VALID EVENT NAME near tgps = {tgps}\n(closes is {self.evname})')
                    self.evname = None
            else:
                print('NO EVENT NAME (or tgps) GIVEN, set evname=GWyymmdd or evname=GWyymmdd_hhmmss (or give tgps)')
        else:
            try:
                self.obs_run = run_from_evname(self.evname)
            except:
                print(f'using default run = {DEFAULT_LIGO_RUN} since none found for {self.evname}')
                self.obs_run = DEFAULT_LIGO_RUN
            if (self.tgps is None) and (evname in bookkeeping):
                self.tgps = bookkeeping[evname][0]

        # dataset name is key in h5 object
        if dataset_name is None:
            opts = list(lvc_h5.keys())
            j = 0
            while (dataset_name is None) and (j < len(self.default_approximant_priority)):
                if self.default_approximant_priority[j] in opts:
                    dataset_name = self.default_approximant_priority[j]
                elif f'C01:{self.default_approximant_priority[j]}' in opts:
                    dataset_name = f'C01:{self.default_approximant_priority[j]}'
                elif f'{self.default_approximant_priority[j]}_posterior' in opts:
                    dataset_name = f'{self.default_approximant_priority[j]}_posterior'
                elif f'C01:{self.default_approximant_priority[j]}_posterior' in opts:
                    dataset_name = f'C01:{self.default_approximant_priority[j]}_posterior'
                j += 1
            if dataset_name is None:
                j = 4
                opts_vals = zip(opts[::-1], [np.count_nonzero([(s in k) for s in ['IMR', 'HM', 'recess', 'Pv3']])
                                             for k in opts[::-1]])
                while (dataset_name is None) and (j > 0):
                    for k, v in opts_vals:
                        if v == j:
                            dataset_name = k
                    j -= 1
                if dataset_name is None:
                    dataset_name = opts[0]
        # in case we specified one that is off by an arbitrary prefix/suffix
        if isinstance(dataset_name, str) and (not (dataset_name in lvc_h5.keys())) and try_dataset_variants:
            if f'C01:{dataset_name}' in lvc_h5.keys():
                dataset_name = f'C01:{dataset_name}'
            elif f'{dataset_name}_posterior' in lvc_h5.keys():
                dataset_name = f'{dataset_name}_posterior'
        self.dataset_name = dataset_name

        # get datasets ready for conversion to sample dataframes
        if self.obs_run == 'O3':
            self.keymap = keymap or DEFAULT_LVC_KEYMAP_O3
            dataset = lvc_h5[dataset_name]['posterior_samples']
            try:
                priors_dataset = lvc_h5[dataset_name]['priors']['samples']
            except:
                priors_dataset = None
            try:
                self.approximant = str(lvc_h5[dataset_name]['approximant'][0])[2:-1]
            except:
                try:
                    self.approximant = str(lvc_h5[dataset_name]['meta_data']['meta_data']['approximant'][0])[2:-1]
                except:
                    self.approximant = None
            try:
                self.psds = {k: lvc_h5[dataset_name]['psds'][k][()] for k in lvc_h5[dataset_name]['psds'].keys()}
                self.det_names = list(lvc_h5[dataset_name]['psds'].keys())
                self.psd_frequencies = self.psds[self.det_names[0]][:, 0]
                self.psd_dets = np.array([self.psds[k][:, 1] for k in self.det_names])
            except:
                self.psds, self.det_names, self.psd_frequencies, self.psd_dets = [None]*4
            try:
                self.f_ref = lvc_h5[dataset_name]['meta_data']['meta_data']['f_ref'][()][0]
            except:
                self.f_ref = None
            try:
                self.f_low = lvc_h5[dataset_name]['meta_data']['meta_data']['f_low'][()][0]
            except:
                self.f_low = None
                
        else:
            self.approximant, self.f_ref, self.f_low, self.psds, self.det_names, \
                self.psd_frequencies, self.psd_dets = [None]*7
            self.keymap = keymap or DEFAULT_LVC_KEYMAP_O2
            dataset = lvc_h5[dataset_name]
            try:
                priors_dataset = lvc_h5['prior']
            except:
                try:
                    priors_dataset = lvc_h5['priors']
                except:
                    priors_dataset = None
                    if not no_print:
                        print('this hdf5 includes no prior')

        # here we are allowing for corrections and additions to what is included in the LVC key mapping
        # --> I think this actually renders the other keymap totally redundant => TODO: reduce to this keymap only
        if isinstance(gwpdic_keymap, dict):
            if keep_unmatched_keys:
                self.keymap.update({gwpdic_keymap.get(v, v): v for v in dataset.dtype.names \
                                    if not (('calib' in v.lower()) or ('spcal' in v.lower()))})
            else:
                self.keymap.update({gwpdic_keymap[v]: v for v in dataset.dtype.names \
                                    if (v in gwpdic_keymap.keys())})
        elif keep_unmatched_keys:
            self.keymap.update({v: v for v in dataset.dtype.names if not (v in self.keymap.values())})
        # make posterior and prior sample dataframes
        self.samples = pd.DataFrame({k: dataset[v] for k, v in self.keymap.items() \
                                     if v in dataset.dtype.names})
        self.Nsamples = len(self.samples)
        # make sure we didn't take any cosines without the angles themselves
        for k in self.samples.columns:
            if ('cos' in k) and (k.replace('cos', '') not in self.samples.columns):
                self.samples[k.replace('cos', '')] = np.arccos(self.samples[k])

        # convert_angles if we have tgps
        if (self.tgps is not None) and (convert_angles):
            try:
                self.samples = samples_with_ligo_angles(self.samples, f_ref=self.f_ref, tgps=self.tgps)
            except:
                if not no_print:
                    print('--> unable to compute f_ref-independent angles')

        # get log likelihoods
        if 'lnL' not in self.samples:
            if 'deltalogl' in dataset.dtype.names:
                self.samples['lnL'] = dataset['deltalogl']
            elif 'log_likelihood' in dataset.dtype.names:
                self.samples['lnL'] = dataset['log_likelihood']
            elif 'likelihood' in dataset.dtype.names:
                try:
                    self.samples['lnL'] = np.log(dataset['likelihood'])
                except:
                    self.samples['lnL'] = dataset['likelihood']
            else:
                self.samples['lnL'] = np.zeros(len(self.samples))
                print('no likelihood information for these LVC samples')
        if ('lnL_H1' not in self.samples) and ('loglH1' in dataset.dtype.names):
            self.samples['lnL_H1'] = dataset['loglH1']
        if ('lnL_L1' not in self.samples) and ('loglL1' in dataset.dtype.names):
            self.samples['lnL_L1'] = dataset['loglL1']
        if ('lnL_V1' not in self.samples) and ('loglV1' in dataset.dtype.names):
            self.samples['lnL_V1'] = dataset['loglV1']

        # get best parameter dictionary
        self.best_pars = None
        if 'lnL' in self.samples:
            self.best_pars = dict(self.samples.iloc[np.argmax(self.samples['lnL'])])
            self.best_pars.update({'tc': 0, 'l1': 0, 'l2': 0})
        # get mean/median statistics for each parameter
        self.means, self.medians, self.stds = {}, {}, {}
        for k in self.samples.columns:
            try:
                self.means[k] = np.mean(self.samples[k])
            except:
                self.means[k] = None
            try:
                self.medians[k] = np.median(self.samples[k])
            except:
                self.medians[k] = None
            try:
                self.stds[k] = np.std(self.samples[k])
            except:
                self.stds[k] = None

        self.prior_samples = None 
        try:
            try:
                parnames = priors_dataset.dtype.names
            except:
                parnames = priors_dataset.keys()
            self.prior_samples = pd.DataFrame({k: priors_dataset[v] for k, v in self.keymap.items() \
                                               if v in parnames})
            for k in self.prior_samples.columns:
                if ('cos' in k) and (k.replace('cos', '') not in self.prior_samples.columns):
                    self.prior_samples[k.replace('cos', '')] = np.arccos(self.prior_samples[k])
        except:
            if not no_print:
                print('no prior samples loaded for this LVC handle')
        if (self.obs_run != 'O3') or compute_aux_O3:
            try:
                gw_utils.compute_samples_aux_vars(self.samples)
                if self.prior_samples is not None:
                    try:
                        gw_utils.compute_samples_aux_vars(self.prior_samples)
                    except:
                        print('posterior parameter conversion successful but failure in priors')
            except:
                print('ERROR: could not convert parameters with gw_utils.compute_samples_aux_vars')

        ## CODES/LABELS/NAMES for plotting and comparison
        self.prior, self.mdl, self.slm = 'iso', 'LVC', '???'
        # get approximant from dataset_name
        if self.approximant is None:
            if (dataset_name[0] == 'C') and (dataset_name[3] == ':'):
                self.approximant = dataset_name[4:]
            elif dataset_name[-10:] == '_posterior':
                self.approximant = dataset_name[:-10]
            else:
                self.approximant = dataset_name
        # get mdl and slm from approximant
        for k, v in MDL_APPROX_LISTS.items():
            if self.approximant in v:
                self.mdl = k
        if self.mdl == 'LVC':
            try:
                self.mdl = approx_to_mdl(self.approximant)
            except:
                print(f'{self.approximant} not in MDL_APPROX_LISTS, no waveform model code found')
        for k, v in SLM_APPROX_LISTS.items():
            if self.approximant in v:
                self.slm = k
        self.s, self.lm = self.slm[0], self.slm[1:]
        # set name and label for plotting
        self.name, self.label = [f'LVC: {self.approximant} ('] * 2
        try:
            self.name += f'{ParameterEstimationHandle.SLM_NAMES[self.slm]})'
            self.label += f'{ParameterEstimationHandle.SLM_LABELS[self.slm]})'
        except:
            print(f'slm={self.slm} is not in [A22, AHM, P22, PHM], so waveform model physics unspecified')
            self.name += 'model physics unspecified)'
            self.label += 'model physics unspecified)'
        if not no_print:
            print(self.Nsamples, 'samples from', dataset_name)
        return

    def sample_completion(self, new_samples=None, **gwpdic_kwargs):
        """use GWPD init completion to compute all parameter conversions"""
        # see if passing anything new, otherwise start with existing samples
        if new_samples is None:
            new_samples = self.samples
        # get keyword arguments for GWPD conversion
        # --> default to complete all with f_ref, tgps from PE json
        kws = dcopy(gwpdic_kwargs)
        kws['allow_bad_keys'] = True
        if kws.get('complete', None) is None:
            kws['complete'] = True
        if kws.get('f_ref', None) is None:
            kws['f_ref'] = self.f_ref
        if kws.get('tgps', None) is None:
            kws['tgps'] = self.tgps
        self.samples = GWPD(par_dic=new_samples, **kws).dataframe()
        return


    def get_best_samples(self, key_rngs={}, get_best_inds=np.arange(20, dtype=int), lnLmin=0, dataframe=False):
        """
        consider samples in bounds given by key_rngs, and sort them by lnL (descending),
        so the highest SNR sample is at index 0, etc. --> from this ordering, take samples at the indices
        given in get_best_inds and return them as a pd.DataFrame if dataframe=True, otherwise return
        them as a list of dicts
        """
        pdics = get_best_pdics(self.samples, key_rngs=key_rngs, get_best_inds=get_best_inds, lnLmin=lnLmin)
        return (pd.DataFrame(pdics) if dataframe else pdics)

    @classmethod
    def from_evname(cls, evname, approximants=SLM_APPROX_LISTS['PHM'], samples_base_dir=None,
                    keymap=None, tgps=None, compute_aux_O3=False, convert_angles=False,
                    get_h5=False, gwpdic_keymap=gw_utils.PARKEY_MAP, no_print=False,
                    keep_unmatched_keys=False):
        """
        load LVC samples from event name and approximant(s)
        :param evname: str 'GWYYYYMMDD' (may include additional '_HHMMSS') specifying event
        :param approximants: if single approximant string, return single LVCsampleHandle, OR
          if list of approximant strings, return list with an LVCsampleHandle for entry of approximants
          --> default to None: use default_approximant_priority and if None are present there's an
          algorithm in cls.__init__ to get the closest one (prioritize precession, higher modes, IMR)
        """
        lvc_h5 = h5py.File(get_lvc_sample_path(evname, samples_base_dir=samples_base_dir), mode='r')
        if isinstance(approximants, str) or (approximants is None):
            ret = cls(lvc_h5, approximants, evname=evname, keymap=keymap, tgps=tgps,
                    compute_aux_O3=compute_aux_O3, convert_angles=convert_angles, gwpdic_keymap=gwpdic_keymap,
                    no_print=no_print, keep_unmatched_keys=keep_unmatched_keys, try_dataset_variants=True)
        else:
            dsns = approximants + ['C01:' + apr for apr in approximants if isinstance(apr, str)] \
                    + [apr + '_posterior' for apr in approximants if isinstance(apr, str)]
            ret = [cls(lvc_h5, dsn, evname=evname, keymap=keymap, tgps=tgps,
                    compute_aux_O3=compute_aux_O3, convert_angles=convert_angles,
                    gwpdic_keymap=gwpdic_keymap, no_print=no_print, keep_unmatched_keys=keep_unmatched_keys)
                    for dsn in dsns if dsn in lvc_h5.keys()]
        return ((lvc_h5, ret) if get_h5 else ret)

    @classmethod
    def prior_only(cls, lvc_h5, dataset_name=None, evname=None, keymap=None,
                   compute_aux_O3=False, convert_angles=False, tgps=None,
                   gwpdic_keymap=gw_utils.PARKEY_MAP, no_print=False, keep_unmatched_keys=False):
        """load only the prior samples as self.samples"""
        if dataset_name is None:
            dataset_name = ('combined' if run_from_evname(evname) == 'O3'
                            else 'Overall_posterior')
        if dataset_name not in lvc_h5.keys():
            dataset_name = list(lvc_h5.keys())[0]
        instance = cls(lvc_h5, dataset_name, evname=evname, keymap=keymap,
                       compute_aux_O3=compute_aux_O3, gwpdic_keymap=gwpdic_keymap,
                       no_print=no_print, keep_unmatched_keys=keep_unmatched_keys)
        instance.samples = instance.prior_samples
        instance.approximant = 'prior'
        instance.name, instance.label = ['LVC prior distribution'] * 2
        instance.approximant, instance.mdl, instance.slm, instance.s, instance.lm = ['prior'] * 5
        return instance

    def get_f_psd(self, det=None):
        """
        return frequencies, psd(frequencies) at detector specified by det (index of string)
        --> if det = None or 'all', psd will be list of length ndet with element j being
            psd(frequencies) at detector lvc_handle.det_names[j]
            --> if getting all detectors and theie frequency arrays are different, then
                frequencies will be a list of length ndet with f at each detector
        """
        if self.psds is None:
            return None
        else:
            freqs = np.array([self.psds[d][:, 0] for d in self.det_names])
            psds = np.array([self.psds[d][:, 1] for d in self.det_names])
            if isinstance(det, int) or isinstance(det, np.ndarray):
                det_inds = det
            elif det in [None, 'all']:
                det_inds = np.arange(len(self.det_names), dtype=int)
            elif det in self.det_names:
                det_inds = self.det_names.index(det)
            else:
                raise ValueError(f'cannot associate detectors with input det = {det}')
            if hasattr(det_inds, '__len__') and (not np.all([np.allclose(freqs[0], f) for f in freqs])):
                print('WARNING: detectors have different frequency grids, returning f at each detector')
                return freqs[det_inds], psds[det_inds]
            else:
                return freqs[0], psds[det_inds]

    def get_f_data(self, det=None):
        raise RuntimeError('LVC DATA PULL NOT YET IMPLEMENTED')

    def corner_plot(self, pvkeys=['m2_source', 'mchirp', 'q', 'chieff'],
                    title=None, figsize=(9,7), scatter_points=None, weights=None, **kwargs):
        """make corner plots"""
        units, plabs = gw_utils.units, gw_utils.param_labels
        for k in pvkeys:
            if k not in units.keys():
                units[k] = ''
            if k not in plabs.keys():
                plabs[k] = k
        nm = f'{self.evname}: {self.name}\n{self.Nsamples} samples'
        if title is None:
            title, pdfnm = nm, None
        else:
            pdfnm = nm
        sg =  gd.Grid.from_samples(pvkeys, self.samples, weights=weights, pdf_key=pdfnm, units=units, labels=plabs)
        return sg.corner_plot(pdf=pdfnm, title=title, figsize=figsize, set_legend=True,
                              scatter_points=scatter_points, **kwargs)

######## CLASS ALIAS ########
LVCHAND = LVCsampleHandle
########################################################################################


class ParameterEstimationComparison(object):
    """class for comparing different PE results"""
    def __init__(self, base_dir=None, evname=None,
                 lvc_samples_path=None, lvc_base_dir=None,
                 lvc_approximants_dict=DEFAULT_LVC_APPROXIMANTS,
                 lvc_keys_dict=DEFAULT_LVC_KEYS_DICT, load_lvc=True,
                 keep_unmatched_lvc_keys=False, convert_lvc_angles=True,
                 det_names=['H1', 'L1', 'V1'], pe_cls_dict=PE_CLASSES,
                 pe_approx_dict=DEFAULT_PE_APPROXIMANTS, pe_handle_list=[],
                 pe_prslm_list=[], pe_dir_post_list=[], pe_kwargs_list=[],
                 default_mdl='IMR', lvc_priors_dataset=None, tgps=None,
                 gwpdic_keymap=gw_utils.PARKEY_MAP):
        """
        initialize PE comparison object containing one or more posteriors from a single event
        :param evname: str 'GWYYYYMMDD' (may include additional '_HHMMSS') specifying event
        """
        super().__init__()
        # get basics
        self.evname = evname
        self.obs_run = run_from_evname(self.evname)
        self.base_dir = base_dir
        self.det_names = det_names
        self.ndet = len(self.det_names)
        self.default_mdl = default_mdl
        self.pe_list = pe_handle_list
        self.pe_registry = {}
        self.pe_registry_update(last=False)
        # this will expand/correct parameter key names
        self.gwpdic_keymap = (dcopy(gwpdic_keymap) if isinstance(gwpdic_keymap, dict)
                                else gwpdic_keymap)
        
        # will allow for access using 'priorSLM' or 'prior_SLM' or ('prior','SLM')
        self.cls_dict = dcopy(pe_cls_dict)
        # will allow for access using 'MDLSLM' or 'MDL_SLM' or ('MDL','SLM')
        self.approx_dict = dcopy(pe_approx_dict)
        self.lvc_approx_dict = dcopy(lvc_approximants_dict)
        # expand key access as promised
        expand_nested_keys([self.cls_dict, self.approx_dict, self.lvc_approx_dict],
                           post=None, keep_orig=True, add_lowercase=True)
        
        # ADD NEW PE INSTANCES
        while len(pe_prslm_list) > len(pe_kwargs_list):
            pe_kwargs_list.append({'mdl': default_mdl})
        while len(pe_kwargs_list) > len(pe_prslm_list):
            pe_prslm_list.append(None)
        while len(pe_prslm_list) > len(pe_dir_post_list):
            pe_dir_post_list.append('')
        # attach pe data
        for prslm, dpost, kwargs in zip(pe_prslm_list, pe_dir_post_list, pe_kwargs_list):
            self.add_pe(prslm=prslm, dir_post=dpost, **kwargs)

        for j in range(len(self.pe_list)):
            self.pe_list[j].list_ind = j
        print('loaded IAS samples:')
        for p in self.pe_list:
            p.key = f'{p.prslm}{p.dir_post}'
            p.infostr = f'{p.Nsamples} IAS samples, {p.key}'
            print(f'pe_list[{p.list_ind}]: {p.prior} {p.approximant} ({p.infostr})')
        print('\n')
        # set gps time based on tgps kwarg in init or in leading pe
        if (len(self.pe_list) > 0) and (tgps is None):
            self.tgps = self.pe_list[0].pe_params.get('tgps')
        else:
            self.tgps = tgps
        
        # get ligo stuff
        self.lvc_samples, self.lvc_inds = {}, {}
        self.lvc_keys = dcopy(lvc_keys_dict)
        self.lvc_path = lvc_samples_path
        self.lvc_data = None
        self.lvc_list, self.lvc_samples, self.lvc_approximants = [], {}, []
        if load_lvc is True:
            if self.lvc_path is None:
                self.lvc_path = get_lvc_sample_path(self.evname, samples_base_dir=lvc_base_dir)
            print(f'\nloading LVC samples from {self.lvc_path}:')
            try:
                self.lvc_data = h5py.File(self.lvc_path, 'r')
                try:
                    self.lvc_list = [LVCsampleHandle(self.lvc_data, k, evname=self.evname, tgps=self.tgps,
                        compute_aux_O3=convert_lvc_angles, convert_angles=convert_lvc_angles,
                        gwpdic_keymap=gwpdic_keymap, no_print=True, keep_unmatched_keys=keep_unmatched_lvc_keys)
                                     for k in self.lvc_data.keys() if k not in ['version', 'prior', 'history']]
                except:
                    self.lvc_list = [LVCsampleHandle(self.lvc_data, k, evname=self.evname, tgps=self.tgps,
                                    compute_aux_O3=False, convert_angles=False, gwpdic_keymap=gwpdic_keymap,
                                    no_print=True, keep_unmatched_keys=keep_unmatched_lvc_keys)
                                     for k in self.lvc_data.keys() if k not in ['version', 'prior', 'history']]
            except:
                print(f'\ncould not get LVC samples from {self.lvc_path}...trying other paths')
                try:
                    self.lvc_data, self.lvc_list = LVCsampleHandle.from_evname(self.evname, tgps=self.tgps,
                                compute_aux_O3=convert_lvc_angles, convert_angles=convert_lvc_angles,
                                get_h5=True, gwpdic_keymap=gwpdic_keymap, no_print=True,
                                keep_unmatched_keys=keep_unmatched_lvc_keys)
                except:
                    self.lvc_data, self.lvc_list = LVCsampleHandle.from_evname(self.evname, tgps=self.tgps,
                                compute_aux_O3=False, convert_angles=False, get_h5=True, no_print=True,
                                gwpdic_keymap=gwpdic_keymap, keep_unmatched_keys=keep_unmatched_lvc_keys)
            
            try:
                prior_handle = LVCsampleHandle.prior_only(self.lvc_data, dataset_name=lvc_priors_dataset,
                            evname=self.evname, tgps=self.tgps, convert_angles=convert_lvc_angles,
                            gwpdic_keymap=gwpdic_keymap, no_print=True, keep_unmatched_keys=keep_unmatched_lvc_keys)
                if prior_handle.samples is not None:
                    if len(prior_handle.samples) > 1:
                        self.lvc_list.append(prior_handle)
                    else:
                        print(f'no LVC priors found in {lvc_priors_dataset}')
                elif lvc_priors_dataset is not None:
                    print(f'no LVC priors found in {lvc_priors_dataset}')
            except:
                if lvc_priors_dataset is not None:
                    print(f'no LVC priors found in {lvc_priors_dataset}')
        
        for j, p in enumerate(self.lvc_list):
            self.lvc_samples[p.approximant] = p.samples
            self.lvc_approximants.append(p.approximant)
            p.list_ind = j
            p.key = f'{p.mdl}_{p.slm}'
            self.lvc_inds[p.approximant] = j
            p.infostr = f'{p.Nsamples} LVC samples, {p.key}'
            print(f'lvc_list[{j}]: {p.approximant} ({p.infostr})')
        
        return


    def sample_completion(self, convert_ias=True, convert_lvc=False, **gwpdic_kwargs):
        """use GWPD init completion to compute all parameter conversions"""
        # see if passing anything new, otherwise start with existing samples
        if convert_ias:
            for p in self.pe_list:
                try:
                    p.sample_completion(**gwpdic_kwargs)
                except:
                    print(p.name, ': unable to complete parameter conversion')
        if convert_lvc:
            for p in self.lvc_list:
                try:
                    p.sample_completion(**gwpdic_kwargs)
                except:
                    print(p.name, ': unable to complete parameter conversion')
        return


    def add_pe_handle(self, new_pe_handle):
        """add fully instantiated pe handle to comparison list"""
        assert new_pe_handle.evname == self.evname, 'all pe in comparrison must have same evname'
        self.pe_list.append(new_pe_handle)
        self.pe_list[-1].list_ind = len(self.pe_list) - 1
        self.pe_registry_update(last=True)
        return
    
    def add_pe(self, prslm=None, dir_post='', mdl=None, base_dir=None, pe_cls_dict=None,
               load_ev=True, load_pe=False, pe_approx_dict=None, approximant=None,
               name_pre='', name_post=None, label_pre='', label_post='',
               prior=None, slm=None, s=None, lm=None, priorAlias=None,
               report_rb_test=False, report_mcmc_test=False, convert_angles=False,
               compute_antenna=False, compute_cosmo_prior=False,
               gwpdic_keymap=gw_utils.PARKEY_MAP, name=None, load_samples=True):
        """
        add pe by specifying prslm or ((prior or priorAlias), (slm or (s, lm)))
          AND specifying dir_post = str following evname in directory (IF any suffix)
          all other arguments can be deduced from instance given defaults
        """
        mdl = mdl or self.default_mdl
        base_dir = base_dir or self.base_dir
        pe_cls_dict = pe_cls_dict or self.cls_dict
        pe_approx_dict = pe_approx_dict or self.approx_dict
        # now append to pe list
        self.add_pe_handle(
            ParameterEstimationHandle(prslm=prslm, dir_post=dir_post, base_dir=base_dir,
                evname=self.evname, mdl=mdl, pe_cls_dict=pe_cls_dict,
                load_samples=load_samples, load_evidence=load_ev, load_pe=load_pe,
                pe_approx_dict=pe_approx_dict, approximant=approximant,
                name_pre=name_pre, name_post=name_post, label_pre=label_pre, label_post=label_post,
                prior=prior, slm=slm, s=s, lm=lm, priorAlias=priorAlias,
                report_rb_test=report_rb_test, report_mcmc_test=report_mcmc_test, convert_angles=convert_angles,
                compute_antenna=compute_antenna, compute_cosmo_prior=compute_cosmo_prior,
                gwpdic_keymap=gwpdic_keymap, name=name))
        return
    
    def add_pe_dir(self, new_pe_dir, mdl='IMR', pe_cls_dict=None,
                   load_ev=True, load_pe=False, pe_approx_dict=None, approximant=None,
                   name_pre='', name_post=None, label_pre='', label_post=None,
                   prior=None, slm=None, s=None, lm=None, priorAlias=None,
                   report_rb_test=False, report_mcmc_test=False, convert_angles=False,
                   compute_antenna=False, compute_cosmo_prior=False,
                   gwpdic_keymap=gw_utils.PARKEY_MAP, name=None, load_samples=True):
        """add pe by specifying absolute path to directory"""
        pe_cls_dict = pe_cls_dict or self.cls_dict
        pe_approx_dict = pe_approx_dict or self.approx_dict
        # now append to pe list
        self.add_pe_handle(
            ParameterEstimationHandle.from_dir(new_pe_dir, mdl=mdl, pe_cls_dict=pe_cls_dict,
                load_samples=load_samples, load_evidence=load_ev, load_pe=load_pe,
                pe_approx_dict=pe_approx_dict, approximant=approximant,
                name_pre=name_pre, name_post=name_post, label_pre=label_pre, label_post=label_post,
                prior=prior, slm=slm, s=s, lm=lm, priorAlias=priorAlias,
                report_rb_test=report_rb_test, report_mcmc_test=report_mcmc_test, convert_angles=convert_angles,
                compute_antenna=compute_antenna, compute_cosmo_prior=compute_cosmo_prior,
                gwpdic_keymap=gwpdic_keymap, name=name))
        return
    
    @classmethod
    def from_dirs(cls, pe_dir_list, pe_kwargs_list=[], default_mdl='IMR',
                  load_lvc=True, lvc_samples_path=None, load_pe_objects=False,
                  load_evidence=False, lvc_approximants_dict=DEFAULT_LVC_APPROXIMANTS,
                  lvc_keys_dict=DEFAULT_LVC_KEYS_DICT, keep_unmatched_lvc_keys=False,
                  det_names=['H1', 'L1', 'V1'], pe_cls_dict=PE_CLASSES,
                  pe_approx_dict=DEFAULT_PE_APPROXIMANTS, lvc_priors_dataset=None,
                  gwpdic_keymap=gw_utils.PARKEY_MAP, convert_angles=True,
                  compute_lnPrior=False, convert_lvc_angles=True, pe_names=None,
                  load_samples=True):
        while len(pe_dir_list) > len(pe_kwargs_list):
            pe_kwargs_list.append({'mdl': default_mdl, 'pe_cls_dict': pe_cls_dict,
                                   'pe_approx_dict': pe_approx_dict})
        ################## DO I NEED TO MAKE SURE THAT ^^ THESE ^^ ARE IN ALL KWARGS DICS???
        if isinstance(pe_names, list):
            for pekws, nm in zip(pe_kwargs_list, pe_names):
                pekws['name'] = nm
        for pekws in pe_kwargs_list:
            pekws['mdl'] = pekws.get('mdl', default_mdl)
            pekws['pe_cls_dict'] = pekws.get('pe_cls_dict', pe_cls_dict)
            pekws['pe_approx_dict'] = pekws.get('pe_approx_dict', pe_approx_dict)
            pekws['gwpdic_keymap'] = pekws.get('gwpdic_keymap', gwpdic_keymap)
            pekws['pe_cls_dict'] = pekws.get('pe_cls_dict', pe_cls_dict)
            pekws['convert_angles'] = pekws.get('convert_angles', convert_angles)
            pekws['load_pe'] = pekws.get('load_pe_objects', load_pe_objects)
            pekws['load_evidence'] = pekws.get('load_evidence', load_evidence)
            pekws['compute_lnPrior'] = pekws.get('compute_lnPrior', compute_lnPrior)
            pekws['load_samples'] = pekws.get('load_samples', load_samples)
        pe_handle_list = [ParameterEstimationHandle.from_dir(pedir, **pekws)
                          for pedir, pekws in zip(pe_dir_list, pe_kwargs_list)]
        base_dir, evname = pe_handle_list[0].base_dir, pe_handle_list[0].evname
        print(evname, 'parameter estimation comparison\n')
        assert all([peh.evname == evname for peh in pe_handle_list]), \
            'all PE in comparison must have same evname, for event comparison use POPPE class'
        return cls(base_dir=base_dir, evname=evname, lvc_samples_path=lvc_samples_path,
                   lvc_approximants_dict=lvc_approximants_dict, lvc_keys_dict=lvc_keys_dict,
                   load_lvc=load_lvc, det_names=det_names, pe_cls_dict=pe_cls_dict,
                   pe_approx_dict=DEFAULT_PE_APPROXIMANTS, pe_handle_list=pe_handle_list,
                   pe_prslm_list=[], pe_dir_post_list=[], pe_kwargs_list=[], default_mdl=default_mdl,
                   lvc_priors_dataset=lvc_priors_dataset, gwpdic_keymap=gwpdic_keymap,
                   keep_unmatched_lvc_keys=keep_unmatched_lvc_keys, convert_lvc_angles=convert_lvc_angles)

    @classmethod
    def from_evname(cls, evname, pe_kwargs_list=[], prior_list=[], slm_list=[], dir_post_list=[],
                    base_dir_list=[], default_prior='flat', default_slm='PHM', default_mdl='IMR',
                    load_lvc=True, lvc_samples_path=None, load_pe_objects=False,
                    load_evidence=False, lvc_approximants_dict=DEFAULT_LVC_APPROXIMANTS,
                    lvc_keys_dict=DEFAULT_LVC_KEYS_DICT, keep_unmatched_lvc_keys=False,
                    det_names=['H1', 'L1', 'V1'], pe_cls_dict=PE_CLASSES,
                    pe_approx_dict=DEFAULT_PE_APPROXIMANTS, lvc_priors_dataset=None,
                    gwpdic_keymap=gw_utils.PARKEY_MAP, convert_angles=True,
                    compute_lnPrior=False, convert_lvc_angles=True, pe_names=None, load_samples=True):
        """initialize from evname"""
        ndirs = max([1] + [len(l) for l in [prior_list, slm_list, dir_post_list, base_dir_list]])
        while len(slm_list) < ndirs:
            slm_list.append(default_slm)
        while len(prior_list) < ndirs:
            prior_list.append(default_prior)
        while len(base_dir_list) < ndirs:
            base_dir_list.append(None)
        while len(dir_post_list) < ndirs:
            dir_post_list.append('')
        pe_dir_list = [get_samples_path(evname=evname, base_dir=dbase, cls=None, samples_dir=None,
                                        prior=prior, slm=slm, full_dir=None, dir_post=dpost, fname=None,
                                        directory_only=True) for dbase, prior, slm, dpost in
                       zip(base_dir_list, prior_list, slm_list, dir_post_list)]
        dirs, kws = [], []
        for j, d in enumerate(pe_dir_list):
            if isinstance(d, str):
                dirs.append(d)
                if len(pe_kwargs_list) > j:
                    kws.append(pe_kwargs_list[j])
        return cls.from_dirs(dirs, pe_kwargs_list=kws, default_mdl=default_mdl,
                  load_lvc=load_lvc, lvc_samples_path=lvc_samples_path, load_pe_objects=load_pe_objects,
                  load_evidence=load_evidence, lvc_approximants_dict=lvc_approximants_dict,
                  lvc_keys_dict=lvc_keys_dict, keep_unmatched_lvc_keys=keep_unmatched_lvc_keys,
                  det_names=det_names, pe_cls_dict=pe_cls_dict, pe_approx_dict=pe_approx_dict,
                  lvc_priors_dataset=lvc_priors_dataset, gwpdic_keymap=gwpdic_keymap,
                  convert_angles=convert_angles, compute_lnPrior=compute_lnPrior,
                  convert_lvc_angles=convert_lvc_angles, pe_names=pe_names, load_samples=load_samples)
    
    def pe_reg_entry(self, ind, keep_nest=False, add_lowercase=True):
        """associate pe_list index with lots of keys"""
        p = self.pe_list[ind]
        if p.dir_post in [None, '']:
            reg = {p.prior: {p.slm: ind}}
        else:
            reg = {p.prior: {p.slm: {p.dir_post: ind}}}
        reg[p.priorAlias] = reg[p.prior]
        expand_nested_keys(reg, post=None, keep_orig=keep_nest, add_lowercase=add_lowercase)
        return reg
    
    def pe_registry_update(self, last=False):
        """update registry that associates keys to pe_list indices"""
        self.Npe = len(self.pe_list)
        if (last is True) and (self.Npe > 0):
            self.pe_registry.update(self.pe_reg_entry(self.Npe - 1, keep_nest=False, add_lowercase=True))
        elif self.Npe > 0:
            for j in range(self.Npe):
                self.pe_registry.update(self.pe_reg_entry(j, keep_nest=False, add_lowercase=True))
        return

    def get_handle(self, key):
        return (self.pe_list[key] if isinstance(key, int) else self.pe_list[self.pe_registry[key]])
    
    def get_pe(self, key):
        return self.get_handle(key).pe

    def get_like(self, key):
        return self.get_handle(key).like

    def get_likelihood(self, key):
        return self.get_handle(key).like

    def get_samples(self, key):
        return self.get_handle(key).samples
    
    def get_lvc(self, key=0, samples_only=False):
        if isinstance(key, str):
            key = self.lvc_inds.get(key) or self.lvc_inds[self.lvc_approx_dict[key]]
        return (self.lvc_list[key].samples if samples_only
                else self.lvc_list[key])
    
    def report_rb_test_results(self, pe_codes=[], plot=True):
        pe_list = (self.pe_list if pe_codes in ['all', None, 0]
                   else [self.get_handle(key) for key in pe_codes])
        for p in pe_list:
            p.report_rb_test_results(plot)
        return
    
    def lvc_f_psd(self, key=0, det=None):
        """
        return frequencies, psd(frequencies) at detector specified by det (index of string)
        --> if det = None or 'all', psd will be list of length ndet with element j being
            psd(frequencies) at detector lvc_handle.det_names[j]
            --> if getting all detectors and theie frequency arrays are different, then
                frequencies will be a list of length ndet with f at each detector
        """
        p = self.get_lvc(key)
        output = p.get_f_psd(det=det)
        if output is None:
            print('No PSDs in LVC handle with key =', key)
        return output

    def lvc_f_data(self, key=0, det=None):
        raise RuntimeError('LVC DATA PULL NOT YET IMPLEMENTED')
    
    def pe_f_psd(self, key=0, det=None, try_finite_psd=False):
        return self.get_handle(key).get_f_psd(det=det, try_finite_psd=try_finite_psd)
    
    def pe_f_data(self, key=0, det=None):
        return self.get_handle(key).get_f_data(det=det)

    def inds_by_attr(self, attr_dict, lvc=False):
        """
        give dict with attr names as keys and a list of desired attr vals as values
        get list of unique inds satisfying criteria
        """
        inds = []
        loop_j_p = enumerate((self.lvc_list if lvc else self.pe_list))
        for k, v in attr_dict.items():
            vlist = ([v] if isinstance(v, str) else v)
            assert isinstance(vlist, list), 'attr_dict vals must be strings or str lists'
            for j, p in loop_j_p:
                if (p.list_ind not in inds) and (p.__dict__[k] in vlist):
                    inds.append(p.list_ind)
        return inds
    
    def corner_plot(self, pe_codes=[], lvc_codes=[], pvkeys=['m2_source', 'mchirp', 'q', 'chieff'],
                    keys_min_means={}, keys_max_means={}, keys_min_medians={}, keys_max_medians={},
                    weight_key=None, pe_names=None, lvc_names=None, fractions=[.5, .9],
                    labs=False, add_approx=True, no_add=False, figsize=(9, 7),
                    extra_samples=None, extra_samples_names=None, scatter_points=None,
                    grid_kws={}, multigrid_kws={}, corner_plot_kws={}, pe_weight_keys=None,
                    lvc_weight_keys=None, extra_samples_weight_keys=None, title=None,
                    alphas=None, return_multigrid=False):
        """
        make corner plots
        :param pe_codes: list of indices/codes to plot from events (and variations) in self.pe_list
          --> use None or 'all' to get all IAS posteriors (empty list for no IAS posteriors)
        :param lvc_codes: list of indices/codes to plot from events (and approximants) in self.lvc_list
          --> use None or 'all' to get all LVC posteriors (empty list for no LVC posteriors)
        :param pvkeys: list of keys for parameters to include in corner plots
        :param keys_min_means: dict w/ keys & values = parameter keys & minimum mean values of those
          parameters, determining whether or not a given posterior is included (default: empty dict)
        :param keys_max_means: dict w/ keys & values = parameter keys & maximum mean values of those
          parameters, determining whether or not a given posterior is included (default: empty dict)
        :param keys_min_medians: dict w/ keys & values = parameter keys & minimum median values of those
          parameters, determining whether or not a given posterior is included (default: empty dict)
        :param keys_max_medians: dict w/ keys & values = parameter keys & maximum median values of those
          parameters, determining whether or not a given posterior is included (default: empty dict)
        :param weight_key: key to use as sample weights (default: None --> no additional weighting)
        :param fractions: list of fractions to show as contours in plots (MultiGrid initializer kwarg)
        :param scatter_points: DataFrame of samples to mark with lines in plots (MultiGrid.corner_plot() kwarg)
        :param grid_kws: additional kwargs passed on to grid.Grid
        :param multigrid_kws: additional kwargs passed on to grid.MultiGrid
        :param corner_plot_kws: additional kwargs passed on to grid.MultiGrid.corner_plot()
        :param extra_samples: list of additional sample DataFrame objects
        :param extra_samples_names: names of additional samples in extra_samples
        :param pe_names: list of same lenth as pe_codes with names of IAS posteriors in plot
          --> if no names given or list does not have one per plotted IAS posterior, use handle.name
        :param lvc_names: list of same lenth as lvc_codes with names of LVC posteriors in plot
          --> if no names given or list does not have one per plotted LVC posterior, use handle.name
        **remaining params are for figure and label formatting** 
        """
        if pe_codes is None:
            pe_list = self.pe_list
        elif isinstance(pe_codes, int) or isinstance(pe_codes, str):
            if pe_codes == 'all':
                pe_list = self.pe_list
            else:
                pe_list = [self.get_handle(pe_codes)]
        elif isinstance(pe_codes, list) or isinstance(pe_codes, tuple):
            pe_list = [self.get_handle(key) for key in pe_codes]
        elif isinstance(pe_codes, dict):
            pe_list = [self.pe_list[k] for k in self.inds_by_attr(pe_codes, lvc=False)]
        else:
            raise ValueError(f'{pe_codes} is not a valid value for pe_codes')

        if lvc_codes is None:
            lvc_list = self.lvc_list
        elif isinstance(lvc_codes, int) or isinstance(lvc_codes, str):
            if lvc_codes == 'all':
                lvc_list = self.lvc_list
            else:
                lvc_list = [self.get_lvc(key=lvc_codes, samples_only=False)]
        elif isinstance(lvc_codes, list) or isinstance(lvc_codes, tuple):
            lvc_list = [self.get_lvc(key=key, samples_only=False) for key in lvc_codes]
        elif isinstance(lvc_codes, dict):
            lvc_list = [self.lvc_list[k] for k in self.inds_by_attr(lvc_codes, lvc=True)]
        else:
            raise ValueError(f'{lvc_codes} is not a valid value for lvc_codes')

        # now remove posteriors with means and medians outside the range
        for k, v in keys_min_means.items():
            pe_list = [p for p in pe_list if p.means[k] > v]
            lvc_list = [p for p in lvc_list if p.means[k] > v]
        for k, v in keys_max_means.items():
            pe_list = [p for p in pe_list if p.means[k] < v]
            lvc_list = [p for p in lvc_list if p.means[k] < v]
        for k, v in keys_min_medians.items():
            pe_list = [p for p in pe_list if p.medians[k] > v]
            lvc_list = [p for p in lvc_list if p.medians[k] > v]
        for k, v in keys_max_medians.items():
            pe_list = [p for p in pe_list if p.medians[k] < v]
            lvc_list = [p for p in lvc_list if p.medians[k] < v]
            
        grids = []
        units, plabs = gw_utils.units, gw_utils.param_labels
        for k in pvkeys:
            if k not in units.keys():
                units[k] = ''
            if k not in plabs.keys():
                plabs[k] = k
        # IAS SAMPLES
        if pe_weight_keys is None:
            pe_weight_keys = [weight_key] * len(pe_list)
        elif isinstance(pe_weight_keys, str):
            pe_weight_keys = [pe_weight_keys] * len(pe_list)
        else:
            assert len(pe_weight_keys) == len(pe_list), \
                    'if giving pe_weight_keys list, must be same length as pe_codes'
        for i, p in enumerate(pe_list):
            labadd = ('' if no_add else f' ({p.approximant if add_approx else p.mdl})')
            pdfkey = (((p.label if labs else p.name) + labadd) if pe_names is None else pe_names[i])
            grids.append(gd.Grid.from_samples(pvkeys, p.samples, units=units, labels=plabs, pdf_key=pdfkey,
                                              weights=p.samples.get(pe_weight_keys[i], None), **grid_kws))
        # LVC SAMPLES
        if lvc_weight_keys is None:
            lvc_weight_keys = [weight_key] * len(lvc_list)
        elif isinstance(lvc_weight_keys, str):
            lvc_weight_keys = [lvc_weight_keys] * len(lvc_list)
        else:
            assert len(lvc_weight_keys) == len(lvc_list), \
                    'if giving lvc_weight_keys list, must be same length as lvc_codes'
        for i, p in enumerate(lvc_list):
            pdfkey = ((p.label if labs else p.name) if lvc_names is None else lvc_names[i])
            grids.append(gd.Grid.from_samples(pvkeys, p.samples, units=units, labels=plabs, pdf_key=pdfkey,
                                              weights=p.samples.get(lvc_weight_keys[i], None), **grid_kws))
        # EXTRA SAMPLES
        if isinstance(extra_samples, pd.DataFrame):
            if extra_samples_weight_keys is None:
                extra_samples_weight_keys = weight_key
            grids.append(gd.Grid.from_samples(pvkeys, extra_samples, units=units, labels=plabs,
                                              pdf_key=(extra_samples_names or 'Extra samples'),
                                              weights=extra_samples.get(extra_samples_weight_keys, None),
                                              **grid_kws))
        elif isinstance(extra_samples, list):
            if extra_samples_weight_keys is None:
                extra_samples_weight_keys = [weight_key] * len(extra_samples)
            elif isinstance(extra_samples_weight_keys, str):
                extra_samples_weight_keys = [extra_samples_weight_keys] * len(extra_samples)
            else:
                assert len(extra_samples_weight_keys) == len(extra_samples), \
                    'if giving extra_samples_weight_keys list, must be same length as extra_samples'

            if extra_samples_names is None:
                extra_samples_names = 'Extra samples'
            if isinstance(extra_samples_names, str):
                extra_samples_names = [extra_samples_names + f' {i}' for i in range(len(extra_samples))]
            else:
                assert len(extra_samples_names) == len(extra_samples), \
                    'if giving extra_samples_names list, must be same length as extra_samples'

            grids += [gd.Grid.from_samples(pvkeys, s, pdf_key=nm, units=units, labels=plabs,
                                           weights=s.get(wk, None), **grid_kws)
                      for s, nm, wk in zip(extra_samples, extra_samples_names, extra_samples_weight_keys)]
        if title is None:
            title = self.evname
        mg = gd.MultiGrid(grids, **multigrid_kws)
        mg.set_fractions(fractions, alphas)
        fig, ax = mg.corner_plot(set_legend=True, title=title, figsize=figsize,
                                 scatter_points=scatter_points, **corner_plot_kws)
        return ((fig, ax, mg) if return_multigrid else (fig, ax))

    def combined_hist(self, pe_codes=[], lvc_codes=[], xkey='mtot', ykey=None, samples_per_posterior=None,
                      keys_min_means={}, keys_max_means={}, keys_min_medians={}, keys_max_medians={},
                      bins=100, cmap='rainbow', extra_samples=None, extra_samples_names=None,
                      title=None, figsize=(10, 10), xlim=None, ylim=None, **hist_kwargs):
        """
        make 1D or 2D histogram of combined posteriors
        :param pe_codes: list of indices/codes to plot from events (and variations) in self.pe_list
          --> use None or 'all' to get all IAS posteriors (empty list for no IAS posteriors)
        :param lvc_codes: list of indices/codes to plot from events (and approximants) in self.lvc_list
          --> use None or 'all' to get all LVC posteriors (empty list for no LVC posteriors)
        :param xkey: key for parameter to go on x-axis
        :param ykey: (optional) key for parameter to go on y-axis
        :param lvc_codes_comoving: list of indices/codes to plot from samples in self.lvc_list_comoving
        :param keys_min_means: dict w/ keys & values = parameter keys & minimum mean values of those
          parameters, determining whether or not a given posterior is included (default: empty dict)
        :param keys_max_means: dict w/ keys & values = parameter keys & maximum mean values of those
          parameters, determining whether or not a given posterior is included (default: empty dict)
        :param keys_min_medians: dict w/ keys & values = parameter keys & minimum median values of those
          parameters, determining whether or not a given posterior is included (default: empty dict)
        :param keys_max_medians: dict w/ keys & values = parameter keys & maximum median values of those
          parameters, determining whether or not a given posterior is included (default: empty dict)
        :param bins: bins argument passed to hist or hist2d
        :param cmap: cmap argument passed to hist2d
        :param samples_per_posterior:
        :param extra_samples: list of additional sample DataFrame objects
        :param extra_samples_names: names of additional samples in extra_samples
        **remaining kwargs for hist or hist2d are for figure and label formatting** 
        """
        if pe_codes is None:
            pe_list = self.pe_list
        elif isinstance(pe_codes, int) or isinstance(pe_codes, str):
            if pe_codes == 'all':
                pe_list = self.pe_list
            else:
                pe_list = [self.get_handle(pe_codes)]
        elif isinstance(pe_codes, list) or isinstance(pe_codes, tuple):
            pe_list = [self.get_handle(key) for key in pe_codes]
        elif isinstance(pe_codes, dict):
            pe_list = [self.pe_list[k] for k in self.inds_by_attr(pe_codes, lvc=False)]
        else:
            raise ValueError(f'{pe_codes} is not a valid value for pe_codes')

        if lvc_codes is None:
            lvc_list = self.lvc_list
        elif isinstance(lvc_codes, int) or isinstance(lvc_codes, str):
            if lvc_codes == 'all':
                lvc_list = self.lvc_list
            else:
                lvc_list = [self.get_lvc(key=lvc_codes, samples_only=False)]
        elif isinstance(lvc_codes, list) or isinstance(lvc_codes, tuple):
            lvc_list = [self.get_lvc(key=key, samples_only=False) for key in lvc_codes]
        elif isinstance(lvc_codes, dict):
            lvc_list = [self.lvc_list[k] for k in self.inds_by_attr(lvc_codes, lvc=True)]
        else:
            raise ValueError(f'{lvc_codes} is not a valid value for lvc_codes')

        # now remove posteriors with means and medians outside the range
        plot_list = pe_list + lvc_list
        for k, v in keys_min_means.items():
            plot_list = [p for p in plot_list if p.means[k] > v]
        for k, v in keys_max_means.items():
            plot_list = [p for p in plot_list if p.means[k] < v]
        for k, v in keys_min_medians.items():
            plot_list = [p for p in plot_list if p.medians[k] > v]
        for k, v in keys_max_medians.items():
            plot_list = [p for p in plot_list if p.medians[k] < v]
        
        print(' samples included in histogram:', '--------------------------------')
        for p in plot_list:
            if hasattr(p, 'name'):
                print(p.name)
        if extra_samples_names is not None:
            for nm in extra_samples_names:
                print(nm)

        plot_list = [p.samples for p in plot_list]
        if isinstance(extra_samples, pd.DataFrame):
            plot_list.append(extra_samples)
        elif isinstance(extra_samples, list):
            plot_list += extra_samples
        
        if samples_per_posterior is None:
            xarr = np.concatenate([np.asarray(s[xkey]) for s in plot_list])
        else:
            xarr = np.concatenate([np.asarray(s[xkey])[np.random.choice(np.arange(len(s)),
                                                                        size=samples_per_posterior)]
                                   for s in plot_list])
        try:
            xlab = gw_utils.param_labels[xkey]
        except:
            xlab = xkey

        # now make histogram (2d if ykey given, else 1d)
        plt.figure(figsize=figsize)
        plt.title(title)
        if isinstance(ykey, str):
            if samples_per_posterior is None:
                yarr = np.concatenate([np.asarray(s[ykey]) for s in plot_list])
            else:
                yarr = np.concatenate([np.asarray(s[ykey])[np.random.choice(np.arange(len(s)),
                                                                            size=samples_per_posterior)]
                                       for s in plot_list])
            try:
                ylab = gw_utils.param_labels[ykey]
            except:
                ylab = ykey
            hist_out = plt.hist2d(xarr, yarr, bins=bins, cmap=cmap, **hist_kwargs)
            plt.ylabel(ylab)
            plt.ylim(ylim)
        else:
            hist_out = plt.hist(xarr, bins=bins, **hist_kwargs)
        plt.xlim(xlim)
        plt.xlabel(xlab)
        return hist_out

    
######## CLASS ALIAS ########
PECOMP = ParameterEstimationComparison
########################################################################################

class PopulationParameterEstimation(object):
    """class for comparing different PE results"""

    def __init__(self, pe_handle_list=[], load_lvc=False, load_lvc_comoving=False,
                 lvc_keys_dict=DEFAULT_LVC_KEYS_DICT, lvc_approximants=None,
                 pe_cls_dict=PE_CLASSES, pe_approx_dict=DEFAULT_PE_APPROXIMANTS, default_mdl='IMR',
                 lvc_evnames=None, lvc_evnames_comoving=None, convert_lvc_angles=False,
                 lvc_approximants_comoving='PrecessingSpinsIMRPHM', lvc_base_dir=None,
                 keep_unmatched_lvc_keys=False, gwpdic_keymap=gw_utils.PARKEY_MAP):
        """
        initialize object to compare PE across events and between IAS and LVC
        lvc_approximants is one or more approximants for which we get LVC samples for EACH evname
        """
        super().__init__()
        # get basics
        self.default_mdl = default_mdl
        self.pe_list = pe_handle_list
        # this will expand/correct parameter key names
        self.gwpdic_keymap = (dcopy(gwpdic_keymap) if isinstance(gwpdic_keymap, dict)
                              else gwpdic_keymap)

        # will allow for access using 'priorSLM' or 'prior_SLM' or ('prior','SLM')
        self.cls_dict = dcopy(pe_cls_dict)
        # will allow for access using 'MDLSLM' or 'MDL_SLM' or ('MDL','SLM')
        self.approx_dict = dcopy(pe_approx_dict)
        # expand key access as promised
        expand_nested_keys([self.cls_dict, self.approx_dict], post=None, keep_orig=True, add_lowercase=True)
        self.pe_registry = {}
        self.pe_registry_update(last=False)

        self.tgps_list = []
        self.evnames, self.evnames_unique = [], []
        print(('\n...loaded IAS samples:' if len(self.pe_list) > 0 else 'no IAS samples loaded'))
        for j, p in enumerate(self.pe_list):
            self.evnames.append(p.evname)
            if p.evname not in self.evnames_unique:
                self.evnames_unique.append(p.evname)
            p.list_ind = j
            p.label = p.name
            p.name = f'{p.evname}{p.dir_post} {p.prslm}'
            p.key = f'{p.evname}{p.dir_post}_{p.prslm}'
            try:
                self.tgps_list.append(p.pe_params['tgps'])
            except:
                try:
                    self.tgps_list.append(bookkeeping[p.evname][0])
                except:
                    self.tgps_list.append(None)
            p.infostr = f'{p.Nsamples} IAS samples, {p.key}'
            print(f'pe_list[{j}]: {p.evname} {p.prior} {p.approximant} ({p.infostr})')
        if len(self.evnames) == len(self.evnames_unique):
            # here we have only 1 IAS posterior per event, so make the event name one of the key aliases
            old_registry = dcopy(self.pe_registry)
            # put the event name keys first so they are first when printed
            self.pe_registry = {p.evname: j for j, p in enumerate(self.pe_list)}
            self.pe_registry.update(old_registry)

        #### get LVC stuff
        self.lvc_keys = dcopy(lvc_keys_dict)

        # get non-comoving LVC samples
        if isinstance(lvc_approximants, str) or (lvc_approximants is None):
            lvc_approximants = [lvc_approximants]
        if len(lvc_approximants) < 2:
            lvc_handle_key = lambda lvchand: lvchand.evname
        else:
            lvc_handle_key = lambda lvchand: lvchand.evname + '_' + lvchand.approximant
        if lvc_evnames is None:
            lvc_evnames = self.evnames_unique
        self.lvc_paths, self.lvc_data, self.lvc_list = [], [], []
        self.lvc_evnames, self.lvc_approximants = [], []
        self.lvc_samples, self.lvc_inds = {}, {}
        if load_lvc:
            print(f'\nloading LVC samples...')
            self.lvc_paths = [get_lvc_sample_path(evnm, samples_base_dir=lvc_base_dir, comoving=False)
                              for evnm in lvc_evnames]
        for j, lvcpath in enumerate(self.lvc_paths):
            evnm = lvc_evnames[j]
            lvch_kws = {'evname': evnm, 'tgps': ias_bookkeeping.bookkeeping[evnm][0], \
                        'compute_aux_O3': convert_lvc_angles, 'convert_angles': convert_lvc_angles, \
                        'gwpdic_keymap': gwpdic_keymap, 'no_print': True, \
                        'keep_unmatched_keys': keep_unmatched_lvc_keys, 'try_dataset_variants': True}
            backup_kws = {'evname': evnm, 'tgps': None, 'gwpdic_keymap': gwpdic_keymap, \
                          'compute_aux_O3': False, 'convert_angles': False, 'no_print': True, \
                          'keep_unmatched_keys': keep_unmatched_lvc_keys, 'try_dataset_variants': True}
            try:
                tmpdata = h5py.File(lvcpath, 'r')
            except:
                tmpdata = None
            if tmpdata is not None:
                n_success = 0
                for k in lvc_approximants:
                    try:
                        self.lvc_list.append(LVCsampleHandle(tmpdata, k, **lvch_kws))
                        n_success += 1
                    except:
                        try:
                            self.lvc_list.append(LVCsampleHandle(tmpdata, k, **backup_kws))
                            n_success += 1
                        except:
                            print(f'\napproximant = {k} is not available in LVC sample path = {lvcpath}')
                if n_success > 0:
                    self.lvc_data.append(tmpdata)
            else:
                print(f'\ncould not get LVC samples from {evnm} with hdf path = {lvcpath}')
        if load_lvc:
            print(f'\n...loaded LVC samples:')
        for j, lh in enumerate(self.lvc_list):
            lh.list_ind = j
            lh.key = lvc_handle_key(lh)
            self.lvc_evnames.append(lh.evname)
            self.lvc_approximants.append(lh.approximant)
            self.lvc_samples[lh.key] = lh.samples
            self.lvc_inds[lh.key] = j
            lh.infostr = f'{lh.Nsamples} LVC samples, {lh.key}'
            print(f'lvc_list[{j}]: {lh.evname} {lh.approximant} ({lh.infostr})')

        # get comoving LVC samples
        if isinstance(lvc_approximants_comoving, str) or (lvc_approximants_comoving is None):
            lvc_approximants_comoving = [lvc_approximants_comoving]
        if len(lvc_approximants_comoving) < 2:
            lvc_handle_key = lambda lvchand: lvchand.evname
        else:
            lvc_handle_key = lambda lvchand: lvchand.evname + '_' + lvchand.approximant
        if lvc_evnames_comoving is None:
            lvc_evnames_comoving = self.evnames_unique
        self.lvc_paths_comoving, self.lvc_data_comoving, self.lvc_list_comoving = [], [], []
        self.lvc_evnames_comoving, self.lvc_approximants_comoving = [], []
        self.lvc_samples_comoving, self.lvc_inds_comoving = {}, {}
        if load_lvc_comoving:
            print(f'\nloading LVC comoving samples...')
            self.lvc_paths_comoving = [get_lvc_sample_path(evnm, samples_base_dir=lvc_base_dir, comoving=True) \
                                       for evnm in lvc_evnames_comoving]
        for j, lvcpath in enumerate(self.lvc_paths_comoving):
            evnm = lvc_evnames_comoving[j]
            lvch_kws = {'evname': evnm, 'tgps': ias_bookkeeping.bookkeeping[evnm][0], \
                        'compute_aux_O3': convert_lvc_angles, 'convert_angles': convert_lvc_angles, \
                        'gwpdic_keymap': gwpdic_keymap, 'no_print': True, \
                        'keep_unmatched_keys': keep_unmatched_lvc_keys, 'try_dataset_variants': True}
            backup_kws = {'evname': evnm, 'tgps': None, 'gwpdic_keymap': gwpdic_keymap, \
                          'compute_aux_O3': False, 'convert_angles': False, 'no_print': True, \
                          'keep_unmatched_keys': keep_unmatched_lvc_keys, 'try_dataset_variants': True}
            try:
                tmpdata = h5py.File(lvcpath, 'r')
            except:
                tmpdata = None
            if tmpdata is not None:
                n_success = 0
                for k in lvc_approximants_comoving:
                    try:
                        self.lvc_list_comoving.append(LVCsampleHandle(tmpdata, k, **lvch_kws))
                        n_success += 1
                    except:
                        try:
                            self.lvc_list_comoving.append(LVCsampleHandle(tmpdata, k, **backup_kws))
                            n_success += 1
                        except:
                            print(f'\napproximant = {k} is not available in LVC sample path = {lvcpath}')
                if n_success > 0:
                    self.lvc_data_comoving.append(tmpdata)
            else:
                print(f'\ncould not get LVC comoving samples from {evnm} with hdf path = {lvcpath}')
        if load_lvc_comoving:
            print(f'\n...loaded LVC comoving samples:')
        for j, lh in enumerate(self.lvc_list_comoving):
            lh.list_ind = j
            lh.key = lvc_handle_key(lh)
            self.lvc_evnames_comoving.append(lh.evname)
            self.lvc_approximants_comoving.append(lh.approximant)
            self.lvc_samples_comoving[lh.key] = lh.samples
            self.lvc_inds_comoving[lh.key] = j
            lh.infostr = f'{lh.Nsamples} LVC comoving samples, {lh.key}'
            print(f'lvc_list_comoving[{j}]: {lh.evname} {lh.approximant} ({lh.infostr})')

        return

    def sample_completion(self, convert_ias=True, convert_lvc=False, convert_lvc_comoving=False, **gwpdic_kwargs):
        """use GWPD init completion to compute all parameter conversions"""
        # see if passing anything new, otherwise start with existing samples
        if convert_ias:
            for p in self.pe_list:
                try:
                    p.sample_completion(**gwpdic_kwargs)
                except:
                    print(p.name, ': unable to complete parameter conversion')
        if convert_lvc:
            for p in self.lvc_list:
                try:
                    p.sample_completion(**gwpdic_kwargs)
                except:
                    print(p.name, ': unable to complete parameter conversion')
        if convert_lvc_comoving:
            for p in self.lvc_list_comoving:
                try:
                    p.sample_completion(**gwpdic_kwargs)
                except:
                    print(p.name, ': unable to complete parameter conversion')
        return

    def add_pe_handle(self, new_pe_handle):
        """add fully instantiated pe handle to comparison list"""
        self.pe_list.append(new_pe_handle)
        self.pe_list[-1].list_ind = len(self.pe_list) - 1
        self.pe_registry_update(last=True)
        return

    def add_pe(self, evname, prslm=None, dir_post='', mdl=None, base_dir=None, pe_cls_dict=None,
               load_ev=True, load_pe=False, pe_approx_dict=None, approximant=None,
               name_pre='', name_post=None, label_pre='', label_post='',
               prior=None, slm=None, s=None, lm=None, priorAlias=None,
               report_rb_test=False, report_mcmc_test=False, convert_angles=False,
               compute_antenna=False, compute_cosmo_prior=False,
               gwpdic_keymap=gw_utils.PARKEY_MAP, name=None, load_samples=True):
        """
        add pe by specifying prslm or ((prior or priorAlias), (slm or (s, lm)))
          AND specifying dir_post = str following evname in directory (IF any suffix)
          all other arguments can be deduced from instance given defaults
        """
        mdl = mdl or self.default_mdl
        base_dir = base_dir or self.base_dir
        pe_cls_dict = pe_cls_dict or self.cls_dict
        pe_approx_dict = pe_approx_dict or self.approx_dict
        # now append to pe list
        self.add_pe_handle(
            ParameterEstimationHandle(prslm=prslm, dir_post=dir_post, base_dir=base_dir,
                evname=evname, mdl=mdl, pe_cls_dict=pe_cls_dict, load_samples=load_samples,
                load_evidence=load_ev, load_pe=load_pe, pe_approx_dict=pe_approx_dict,
                approximant=approximant, name_pre=name_pre, name_post=name_post,
                label_pre=label_pre, label_post=label_post, prior=prior, slm=slm,
                s=s, lm=lm, priorAlias=priorAlias, report_rb_test=report_rb_test,
                report_mcmc_test=report_mcmc_test, convert_angles=convert_angles,
                compute_antenna=compute_antenna, compute_cosmo_prior=compute_cosmo_prior,
                gwpdic_keymap=gwpdic_keymap, name=name))
        return


    def add_pe_dir(self, new_pe_dir, mdl='IMR', pe_cls_dict=None,
                   load_ev=True, load_pe=False, pe_approx_dict=None, approximant=None,
                   name_pre='', name_post=None, label_pre='', label_post=None,
                   prior=None, slm=None, s=None, lm=None, priorAlias=None,
                   report_rb_test=False, report_mcmc_test=False, convert_angles=False,
                   compute_antenna=False, compute_cosmo_prior=False,
                   gwpdic_keymap=gw_utils.PARKEY_MAP, name=None, load_samples=True):
        """add pe by specifying absolute path to directory"""
        pe_cls_dict = pe_cls_dict or self.cls_dict
        pe_approx_dict = pe_approx_dict or self.approx_dict
        # now append to pe list
        self.add_pe_handle(
            ParameterEstimationHandle.from_dir(new_pe_dir, mdl=mdl, pe_cls_dict=pe_cls_dict,
                load_samples=load_samples, load_evidence=load_ev, load_pe=load_pe,
                pe_approx_dict=pe_approx_dict, approximant=approximant,
                name_pre=name_pre, name_post=name_post, label_pre=label_pre, label_post=label_post,
                prior=prior, slm=slm, s=s, lm=lm, priorAlias=priorAlias,
                report_rb_test=report_rb_test, report_mcmc_test=report_mcmc_test,
                convert_angles=convert_angles, compute_antenna=compute_antenna,
                compute_cosmo_prior=compute_cosmo_prior, gwpdic_keymap=gwpdic_keymap, name=name))
        return

    @classmethod
    def from_dirs(cls, pe_dir_list, pe_kwargs_list=[], default_mdl='IMR',
                  load_lvc=False, load_lvc_comoving=False, lvc_approximants=None,
                  load_pe_objects=False, load_evidence=False, convert_angles=False,
                  compute_lnPrior=False, lvc_keys_dict=DEFAULT_LVC_KEYS_DICT,
                  pe_cls_dict=PE_CLASSES, pe_approx_dict=DEFAULT_PE_APPROXIMANTS, lvc_evnames=None,
                  lvc_evnames_comoving=None, lvc_approximants_comoving='PrecessingSpinsIMRPHM',
                  gwpdic_keymap=gw_utils.PARKEY_MAP, convert_lvc_angles=False,
                  lvc_base_dir=None, keep_unmatched_lvc_keys=False, pe_names=None, load_samples=True):
        """
        initialize from a list of directory paths
        :param pe_dir_list: list of paths to directories containing samples
        :param pe_kwargs_list: list of dictionaries to unpack as kwargs when initializing
          the handle from the corresponding sample directory (do not need to pass any)
        ALL other params passed directly to cls.__init__() or used as defaults for handle kwargs
        """
        pe_dir_list = [fnm for fnm in pe_dir_list if isinstance(fnm, str)]
        while len(pe_dir_list) > len(pe_kwargs_list):
            pe_kwargs_list.append({})
        if isinstance(pe_names, list):
            for pekws, nm in zip(pe_kwargs_list, pe_names):
                pekws['name'] = nm
        for pekws in pe_kwargs_list:
            pekws['mdl'] = pekws.get('mdl', default_mdl)
            pekws['pe_cls_dict'] = pekws.get('pe_cls_dict', pe_cls_dict)
            pekws['pe_approx_dict'] = pekws.get('pe_approx_dict', pe_approx_dict)
            pekws['gwpdic_keymap'] = pekws.get('gwpdic_keymap', gwpdic_keymap)
            pekws['convert_angles'] = pekws.get('convert_angles', convert_angles)
            pekws['load_pe'] = pekws.get('load_pe', load_pe_objects)
            pekws['load_evidence'] = pekws.get('load_evidence', load_evidence)
            pekws['compute_lnPrior'] = pekws.get('compute_lnPrior', compute_lnPrior)
            pekws['load_samples'] = pekws.get('load_samples', load_samples)

        pe_handle_list = [ParameterEstimationHandle.from_dir(pedir, **pekws)
                          for pedir, pekws in zip(pe_dir_list, pe_kwargs_list)]
        return cls(pe_handle_list=pe_handle_list, load_lvc=load_lvc, load_lvc_comoving=load_lvc_comoving,
                   lvc_keys_dict=lvc_keys_dict, lvc_approximants=lvc_approximants,
                   pe_cls_dict=pe_cls_dict, pe_approx_dict=pe_approx_dict, default_mdl=default_mdl,
                   lvc_evnames=lvc_evnames, lvc_evnames_comoving=lvc_evnames_comoving,
                   lvc_approximants_comoving=lvc_approximants_comoving, lvc_base_dir=lvc_base_dir,
                   convert_lvc_angles=convert_lvc_angles, gwpdic_keymap=gwpdic_keymap,
                   keep_unmatched_lvc_keys=keep_unmatched_lvc_keys)

    @classmethod
    def from_evnames(cls, evname_list, pe_kwargs_list=[], prior_list=[], slm_list=[],
                     base_dir_list=[], dir_post_list=[], default_prior='flat', default_slm='PHM',
                     default_mdl='IMR', load_lvc=False, load_lvc_comoving=False,
                     lvc_approximants=None, lvc_evnames=None,
                     lvc_approximants_comoving='PrecessingSpinsIMRPHM', lvc_evnames_comoving=None,
                     load_pe_objects=False, load_evidence=False, convert_angles=False,
                     compute_lnPrior=False, lvc_keys_dict=DEFAULT_LVC_KEYS_DICT,
                     pe_cls_dict=PE_CLASSES, pe_approx_dict=DEFAULT_PE_APPROXIMANTS,
                     gwpdic_keymap=gw_utils.PARKEY_MAP, convert_lvc_angles=False, lvc_base_dir=None,
                     keep_unmatched_lvc_keys=False, pe_names=None, load_samples=True):
        """
        initialize from a list of event names
        :param evname_list: list of strings 'GWYYYYMMDD' (might also include '_HHMMSS') for each event
          --> can have repeats in evname_list when including more than one IAS posterior per event
        :param default_prior: prior to use if not passing prior_list
        :param default_slm: slm to use if not passing slm_list
        :param prior_list: list of prior to use for corresponding evname (default: use default_prior for all)
        :param default_slm: list of slm to use for corresponding evname (default: use default_slm for all)
        ALL other params passed directly to cls.from_dirs()
        """
        ndirs = len(evname_list)
        while len(slm_list) < ndirs:
            slm_list.append(default_slm)
        while len(prior_list) < ndirs:
            prior_list.append(default_prior)
        while len(base_dir_list) < ndirs:
            base_dir_list.append(None)
        while len(dir_post_list) < ndirs:
            dir_post_list.append('')
        pe_dir_list = [get_samples_path(evname=evnm, base_dir=basedir, cls=None, samples_dir=None,
                                        prior=prior, slm=slm, full_dir=None, dir_post=dpost,
                                        fname=None, directory_only=True)
                       for evnm, basedir, prior, slm, dpost in
                       zip(evname_list, base_dir_list, prior_list, slm_list, dir_post_list)]
        dirs, kws = [], []
        for j, d in enumerate(pe_dir_list):
            if isinstance(d, str):
                dirs.append(d)
                if len(pe_kwargs_list) > j:
                    kws.append(pe_kwargs_list[j])
        return cls.from_dirs(dirs, pe_kwargs_list=kws, default_mdl=default_mdl,
                             load_lvc=load_lvc, load_lvc_comoving=load_lvc_comoving,
                             lvc_keys_dict=lvc_keys_dict, lvc_approximants=lvc_approximants,
                             load_pe_objects=load_pe_objects, load_evidence=load_evidence,
                             convert_angles=convert_angles, compute_lnPrior=compute_lnPrior,
                             pe_cls_dict=pe_cls_dict, pe_approx_dict=pe_approx_dict,
                             lvc_base_dir=lvc_base_dir, lvc_evnames=lvc_evnames,
                             lvc_evnames_comoving=lvc_evnames_comoving,
                             lvc_approximants_comoving=lvc_approximants_comoving,
                             gwpdic_keymap=gw_utils.PARKEY_MAP, convert_lvc_angles=convert_lvc_angles,
                             keep_unmatched_lvc_keys=keep_unmatched_lvc_keys,
                             pe_names=pe_names, load_samples=load_samples)

    def pe_reg_entry(self, ind, keep_nest=False, add_lowercase=True):
        """associate pe_list index with lots of keys"""
        p = self.pe_list[ind]
        if p.dir_post in [None, '']:
            reg = {p.evname: {p.prior: {p.slm: ind}}}
            reg[p.evname].update({p.priorAlias: {p.slm: ind}})
        else:
            reg = {p.evname + p.dir_post: {p.prior: {p.slm: ind}}}
            reg[p.evname + p.dir_post].update({p.priorAlias: {p.slm: ind}})
        expand_nested_keys(reg, post=None, keep_orig=keep_nest, add_lowercase=add_lowercase)
        return reg

    def pe_registry_update(self, last=False):
        """update registry that associates keys to pe_list indices"""
        self.Npe = len(self.pe_list)
        if (last is True) and (self.Npe > 0):
            self.pe_registry.update(self.pe_reg_entry(self.Npe - 1, keep_nest=False, add_lowercase=True))
        elif self.Npe > 0:
            for j in range(self.Npe):
                self.pe_registry.update(self.pe_reg_entry(j, keep_nest=False, add_lowercase=True))
        return

    def get_handle(self, key):
        return (self.pe_list[key] if isinstance(key, int) else self.pe_list[self.pe_registry[key]])

    def get_pe(self, key):
        return self.get_handle(key).pe

    def get_like(self, key):
        return self.get_handle(key).like

    def get_likelihood(self, key):
        return self.get_handle(key).like

    def get_samples(self, key):
        return self.get_handle(key).samples

    def get_lvc(self, key=None, evname=None, approximant=None, samples_only=False):
        if key is None:
            key = (0 if (evname is None) \
                    else evname + ('' if approximant is None else '_' + approximant))
        ind = self.lvc_inds.get(key, key)
        try:
            return (self.lvc_list[ind].samples if samples_only \
                    else self.lvc_list[ind])
        except:
            print(f'no LVC handle for key = {key}')
            return None

    def lvc_f_psd(self, key=0, det=None):
        """
        return frequencies, psd(frequencies) at detector specified by det (index of string)
        --> if det = None or 'all', psd will be list of length ndet with element j being
            psd(frequencies) at detector lvc_handle.det_names[j]
            --> if getting all detectors and theie frequency arrays are different, then
                frequencies will be a list of length ndet with f at each detector
        """
        p = self.get_lvc(key)
        output = p.get_f_psd(det=det)
        if output is None:
            print('No PSDs in LVC handle with key =', key)
        return output

    def lvc_f_data(self, key=0, det=None):
        raise RuntimeError('LVC DATA PULL NOT YET IMPLEMENTED')
    
    def pe_f_psd(self, key=0, det=None, try_finite_psd=False):
        return self.get_handle(key).get_f_psd(det=det, try_finite_psd=try_finite_psd)
    
    def pe_f_data(self, key=0, det=None):
        return self.get_handle(key).get_f_data(det=det)

    def inds_by_attr(self, attr_dict, lvc=False, comoving=False):
        """
        give dict with attr names as keys and a list of desired attr vals as values
        get list of unique inds satisfying criteria
        """
        inds = []
        loop_j_p = enumerate(((self.lvc_list_comoving if comoving else self.lvc_list) \
                              if lvc else self.pe_list))
        for k, v in attr_dict.items():
            vlist = ([v] if isinstance(v, str) else v)
            assert isinstance(v, list), 'attr_dict vals must be strings or str lists'
            for j, p in loop_j_p:
                if (p.list_ind not in inds) and (p.__dict__.get(k) in vlist):
                    inds.append(p.list_ind)
        return inds

    def corner_plot(self, pe_codes=[], lvc_codes=[], pvkeys=['mtot', 'q', 'chieff'], lvc_codes_comoving=[],
                    keys_min_means={}, keys_max_means={}, keys_min_medians={}, keys_max_medians={},
                    weight_key=None, fractions=[.5, .9], pe_names=None, lvc_names=None, lvc_names_comoving=None,
                    labs=False, add_approx=True, no_add=False, figsize=(9, 7), scatter_points=None,
                    extra_samples=None, extra_samples_names=None, grid_kws={}, multigrid_kws={}, corner_plot_kws={},
                    pe_weight_keys=None, lvc_weight_keys=None, lvc_weight_keys_comoving=None,
                    extra_samples_weight_keys=None, title=None, alphas=None, return_multigrid=False):
        """
        make corner plots
        :param pe_codes: list of indices/codes to plot from events (and variations) in self.pe_list
          --> use None or 'all' to get all IAS posteriors (empty list for no IAS posteriors)
        :param lvc_codes: list of indices/codes to plot from events (and approximants) in self.lvc_list
          --> use None or 'all' to get all LVC posteriors (empty list for no LVC posteriors)
        :param pvkeys: list of keys for parameters to include in corner plots
        :param lvc_codes_comoving: list of indices/codes to plot from samples in self.lvc_list_comoving
          --> use None or 'all' to get all LVC comoving posteriors (empty list for none of them)
        :param keys_min_means: dict w/ keys & values = parameter keys & minimum mean values of those
          parameters, determining whether or not a given posterior is included (default: empty dict)
        :param keys_max_means: dict w/ keys & values = parameter keys & maximum mean values of those
          parameters, determining whether or not a given posterior is included (default: empty dict)
        :param keys_min_medians: dict w/ keys & values = parameter keys & minimum median values of those
          parameters, determining whether or not a given posterior is included (default: empty dict)
        :param keys_max_medians: dict w/ keys & values = parameter keys & maximum median values of those
          parameters, determining whether or not a given posterior is included (default: empty dict)
        :param weight_key: key to use as sample weights (default: None --> no additional weighting)
        :param fractions: list of fractions to show as contours in plots (MultiGrid initializer kwarg)
        :param scatter_points: DataFrame of samples to mark with lines in plots (MultiGrid.corner_plot() kwarg)
        :param grid_kws: additional kwargs passed on to grid.Grid
        :param multigrid_kws: additional kwargs passed on to grid.MultiGrid
        :param corner_plot_kws: additional kwargs passed on to grid.MultiGrid.corner_plot()
        :param extra_samples: list of additional sample DataFrame objects
        :param extra_samples_names: names of additional samples in extra_samples
        :param pe_names: list of same lenth as pe_codes with names of IAS posteriors in plot
          --> if no names given or list does not have one per plotted IAS posterior, use handle.name
        :param lvc_names: list of same lenth as lvc_codes with names of LVC posteriors in plot
          --> if no names given or list does not have one per plotted LVC posterior, use handle.name
        :param lvc_names_comoving: list of same lenth as lvc_codes_comoving with names of LVC comoving posteriors
          --> if no names given or list does not have one per plotted LVC comoving posterior, use handle.name
        **remaining params are for figure and label formatting** 
        """
        if pe_codes is None:
            pe_list = self.pe_list
        elif isinstance(pe_codes, int) or isinstance(pe_codes, str):
            if pe_codes == 'all':
                pe_list = self.pe_list
            else:
                pe_list = [self.get_handle(pe_codes)]
        elif isinstance(pe_codes, list) or isinstance(pe_codes, tuple):
            pe_list = [self.get_handle(key) for key in pe_codes]
        elif isinstance(pe_codes, dict):
            pe_list = [self.pe_list[k] for k in self.inds_by_attr(pe_codes, lvc=False)]
        else:
            raise ValueError(f'{pe_codes} is not a valid value for pe_codes')
        
        if lvc_codes is None:
            lvc_list = self.lvc_list
        elif isinstance(lvc_codes, int) or isinstance(lvc_codes, str):
            if lvc_codes == 'all':
                lvc_list = self.lvc_list
            else:
                lvc_list = [self.get_lvc(key=lvc_codes, samples_only=False)]
        elif isinstance(lvc_codes, list) or isinstance(lvc_codes, tuple):
            lvc_list = [self.get_lvc(key=key, samples_only=False) for key in lvc_codes]
        elif isinstance(lvc_codes, dict):
            lvc_list = [self.lvc_list[k] for k in self.inds_by_attr(lvc_codes, lvc=True, comoving=False)]
        else:
            raise ValueError(f'{lvc_codes} is not a valid value for lvc_codes')

        if lvc_codes_comoving is None:
            lvc_list_comoving = self.lvc_list_comoving
        elif isinstance(lvc_codes_comoving, int) or isinstance(lvc_codes_comoving, str):
            if lvc_codes_comoving == 'all':
                lvc_list_comoving = self.lvc_list_comoving
            else:
                lvc_list_comoving = [self.get_lvc(key=lvc_codes_comoving, samples_only=False)]
        elif isinstance(lvc_codes_comoving, list) or isinstance(lvc_codes_comoving, tuple):
            lvc_list_comoving = [self.get_lvc(key=key, samples_only=False) for key in lvc_codes_comoving]
        elif isinstance(lvc_codes_comoving, dict):
            lvc_list_comoving = [self.lvc_list_comoving[k] for k in \
                                 self.inds_by_attr(lvc_codes_comoving, lvc=True, comoving=True)]
        else:
            raise ValueError(f'{lvc_codes_comoving} is not a valid value for lvc_codes_comoving')

        # now remove posteriors with means and medians outside the range
        for k, v in keys_min_means.items():
            pe_list = [p for p in pe_list if p.means[k] > v]
            lvc_list = [p for p in lvc_list if p.means[k] > v]
            lvc_list_comoving = [p for p in lvc_list_comoving if p.means[k] > v]
        for k, v in keys_max_means.items():
            pe_list = [p for p in pe_list if p.means[k] < v]
            lvc_list = [p for p in lvc_list if p.means[k] < v]
            lvc_list_comoving = [p for p in lvc_list_comoving if p.means[k] < v]
        for k, v in keys_min_medians.items():
            pe_list = [p for p in pe_list if p.medians[k] > v]
            lvc_list = [p for p in lvc_list if p.medians[k] > v]
            lvc_list_comoving = [p for p in lvc_list_comoving if p.medians[k] > v]
        for k, v in keys_max_medians.items():
            pe_list = [p for p in pe_list if p.medians[k] < v]
            lvc_list = [p for p in lvc_list if p.medians[k] < v]
            lvc_list_comoving = [p for p in lvc_list_comoving if p.medians[k] < v]

        grids = []
        units, plabs = gw_utils.units, gw_utils.param_labels
        for k in pvkeys:
            if k not in units.keys():
                units[k] = ''
            if k not in plabs.keys():
                plabs[k] = k
        # IAS SAMPLES
        if pe_weight_keys is None:
            pe_weight_keys = [weight_key] * len(pe_list)
        elif isinstance(pe_weight_keys, str):
            pe_weight_keys = [pe_weight_keys] * len(pe_list)
        else:
            assert len(pe_weight_keys) == len(pe_list), \
                    'if giving pe_weight_keys list, must be same length as pe_codes'
        for i, p in enumerate(pe_list):
            labadd = ('' if no_add else f' ({p.approximant if add_approx else p.mdl})')
            pdfkey = (((p.label if labs else p.name) + labadd) if pe_names is None else pe_names[i])
            grids.append(gd.Grid.from_samples(pvkeys, p.samples, units=units, labels=plabs, pdf_key=pdfkey, \
                                              weights=p.samples.get(pe_weight_keys[i], None), **grid_kws))
        # LVC SAMPLES
        if lvc_weight_keys is None:
            lvc_weight_keys = [weight_key] * len(lvc_list)
        elif isinstance(lvc_weight_keys, str):
            lvc_weight_keys = [lvc_weight_keys] * len(lvc_list)
        else:
            assert len(lvc_weight_keys) == len(lvc_list), \
                    'if giving lvc_weight_keys list, must be same length as lvc_codes'
        for i, p in enumerate(lvc_list):
            pdfkey = ((p.label if labs else p.name) if lvc_names is None else lvc_names[i])
            if not (p.evname in pdfkey):
                pdfkey = p.evname + ': ' + pdfkey
            if not ('lvc' in pdfkey.lower()):
                pdfkey = '(LVC) ' + pdfkey
            grids.append(gd.Grid.from_samples(pvkeys, p.samples, units=units, labels=plabs, pdf_key=pdfkey, \
                                              weights=p.samples.get(lvc_weight_keys[i], None), **grid_kws))
        # LVC SAMPLES COMOVING
        if lvc_weight_keys_comoving is None:
            lvc_weight_keys_comoving = [weight_key] * len(lvc_list_comoving)
        elif isinstance(lvc_weight_keys_comoving, str):
            lvc_weight_keys_comoving = [lvc_weight_keys_comoving] * len(lvc_list_comoving)
        else:
            assert len(lvc_weight_keys_comoving) == len(lvc_list_comoving), \
                    'if giving lvc_weight_keys_comoving list, must be same length as lvc_codes_comoving'
        for i, p in enumerate(lvc_list_comoving):
            pdfkey = ((p.label if labs else p.name) if lvc_names_comoving is None else lvc_names_comoving[i])
            if not (p.evname in pdfkey):
                pdfkey = p.evname + ': ' + pdfkey
            if not ('lvc' in pdfkey.lower()):
                pdfkey = '(LVC) ' + pdfkey
            if not ('comov' in pdfkey.lower()):
                pdfkey += ' [comoving]'
            grids.append(gd.Grid.from_samples(pvkeys, p.samples, units=units, labels=plabs, pdf_key=pdfkey, \
                                              weights=p.samples.get(lvc_weight_keys_comoving[i], None), **grid_kws))
        # EXTRA SAMPLES
        if isinstance(extra_samples, pd.DataFrame):
            if extra_samples_weight_keys is None:
                extra_samples_weight_keys = weight_key
            grids.append(gd.Grid.from_samples(pvkeys, extra_samples, units=units, labels=plabs,
                                              pdf_key=(extra_samples_names or 'Extra samples'),
                                              weights=extra_samples.get(extra_samples_weight_keys, None),
                                              **grid_kws))
        elif isinstance(extra_samples, list):
            if extra_samples_weight_keys is None:
                extra_samples_weight_keys = [weight_key] * len(extra_samples)
            elif isinstance(extra_samples_weight_keys, str):
                extra_samples_weight_keys = [extra_samples_weight_keys] * len(extra_samples)
            else:
                assert len(extra_samples_weight_keys) == len(extra_samples), \
                    'if giving extra_samples_weight_keys list, must be same length as extra_samples'

            if extra_samples_names is None:
                extra_samples_names = 'Extra samples'
            if isinstance(extra_samples_names, str):
                extra_samples_names = [extra_samples_names + f' {i}' for i in range(len(extra_samples))]
            else:
                assert len(extra_samples_names) == len(extra_samples), \
                    'if giving extra_samples_names list, must be same length as extra_samples'

            grids += [gd.Grid.from_samples(pvkeys, s, pdf_key=nm, units=units, labels=plabs,
                                           weights=s.get(wk, None), **grid_kws)
                      for s, nm, wk in zip(extra_samples, extra_samples_names, extra_samples_weight_keys)]

        
        mg = gd.MultiGrid(grids, **multigrid_kws)
        mg.set_fractions(fractions, alphas)
        fig, ax = mg.corner_plot(set_legend=True, title=title, figsize=figsize,
                                 scatter_points=scatter_points, **corner_plot_kws)
        return ((fig, ax, mg) if return_multigrid else (fig, ax))
        

    def combined_hist(self, pe_codes=[], lvc_codes=[], xkey='mtot', ykey=None, lvc_codes_comoving=[],
                      keys_min_means={}, keys_max_means={}, keys_min_medians={}, keys_max_medians={},
                      samples_per_posterior=None, bins=100, cmap='rainbow', extra_samples=None,
                      extra_samples_names=None, title=None, figsize=(10, 10), xlim=None, ylim=None, **hist_kwargs):
        """
        make 1D or 2D histogram of combined posteriors
        :param pe_codes: list of indices/codes to plot from events (and variations) in self.pe_list
          --> use None or 'all' to get all IAS posteriors (empty list for no IAS posteriors)
        :param lvc_codes: list of indices/codes to plot from events (and approximants) in self.lvc_list
          --> use None or 'all' to get all LVC posteriors (empty list for no LVC posteriors)
        :param xkey: key for parameter to go on x-axis
        :param ykey: (optional) key for parameter to go on y-axis
        :param lvc_codes_comoving: list of indices/codes to plot from samples in self.lvc_list_comoving
          --> use None or 'all' to get all LVC comoving posteriors (empty list for none of them)
        :param keys_min_means: dict w/ keys & values = parameter keys & minimum mean values of those
          parameters, determining whether or not a given posterior is included (default: empty dict)
        :param keys_max_means: dict w/ keys & values = parameter keys & maximum mean values of those
          parameters, determining whether or not a given posterior is included (default: empty dict)
        :param keys_min_medians: dict w/ keys & values = parameter keys & minimum median values of those
          parameters, determining whether or not a given posterior is included (default: empty dict)
        :param keys_max_medians: dict w/ keys & values = parameter keys & maximum median values of those
          parameters, determining whether or not a given posterior is included (default: empty dict)
        :param bins: bins argument passed to hist or hist2d
        :param cmap: cmap argument passed to hist2d
        :param samples_per_posterior:
        :param extra_samples: list of additional sample DataFrame objects
        :param extra_samples_names: names of additional samples in extra_samples
        **remaining kwargs for hist or hist2d are for figure and label formatting** 
        """
        if pe_codes is None:
            pe_list = self.pe_list
        elif isinstance(pe_codes, int) or isinstance(pe_codes, str):
            if pe_codes == 'all':
                pe_list = self.pe_list
            else:
                pe_list = [self.get_handle(pe_codes)]
        elif isinstance(pe_codes, list) or isinstance(pe_codes, tuple):
            pe_list = [self.get_handle(key) for key in pe_codes]
        elif isinstance(pe_codes, dict):
            pe_list = [self.pe_list[k] for k in self.inds_by_attr(pe_codes, lvc=False)]
        else:
            raise ValueError(f'{pe_codes} is not a valid value for pe_codes')

        if lvc_codes is None:
            lvc_list = self.lvc_list
        elif isinstance(lvc_codes, int) or isinstance(lvc_codes, str):
            if lvc_codes == 'all':
                lvc_list = self.lvc_list
            else:
                lvc_list = [self.get_lvc(key=lvc_codes, samples_only=False)]
        elif isinstance(lvc_codes, list) or isinstance(lvc_codes, tuple):
            lvc_list = [self.get_lvc(key=key, samples_only=False) for key in lvc_codes]
        elif isinstance(lvc_codes, dict):
            lvc_list = [self.lvc_list[k] for k in self.inds_by_attr(lvc_codes, lvc=True, comoving=False)]
        else:
            raise ValueError(f'{lvc_codes} is not a valid value for lvc_codes')

        if lvc_codes_comoving is None:
            lvc_list_comoving = self.lvc_list_comoving
        elif isinstance(lvc_codes_comoving, int) or isinstance(lvc_codes_comoving, str):
            if lvc_codes_comoving == 'all':
                lvc_list_comoving = self.lvc_list_comoving
            else:
                lvc_list_comoving = [self.get_lvc(key=lvc_codes_comoving, samples_only=False)]
        elif isinstance(lvc_codes_comoving, list) or isinstance(lvc_codes_comoving, tuple):
            lvc_list_comoving = [self.get_lvc(key=key, samples_only=False) for key in lvc_codes_comoving]
        elif isinstance(lvc_codes_comoving, dict):
            lvc_list_comoving = [self.lvc_list_comoving[k] for k in \
                                 self.inds_by_attr(lvc_codes_comoving, lvc=True, comoving=True)]
        else:
            raise ValueError(f'{lvc_codes_comoving} is not a valid value for lvc_codes_comoving')

        # now remove posteriors with means and medians outside the range
        plot_list = pe_list + lvc_list + lvc_list_comoving
        for k, v in keys_min_means.items():
            plot_list = [p for p in plot_list if p.means[k] > v]
        for k, v in keys_max_means.items():
            plot_list = [p for p in plot_list if p.means[k] < v]
        for k, v in keys_min_medians.items():
            plot_list = [p for p in plot_list if p.medians[k] > v]
        for k, v in keys_max_medians.items():
            plot_list = [p for p in plot_list if p.medians[k] < v]
        
        print(' samples included in histogram:', '--------------------------------')
        for p in plot_list:
            if hasattr(p, 'name'):
                print(p.name)
        if extra_samples_names is not None:
            for nm in extra_samples_names:
                print(nm)

        plot_list = [p.samples for p in plot_list]
        if isinstance(extra_samples, pd.DataFrame):
            plot_list.append(extra_samples)
        elif isinstance(extra_samples, list):
            plot_list += extra_samples
        
        if samples_per_posterior is None:
            xarr = np.concatenate([np.asarray(s[xkey]) for s in plot_list])
        else:
            xarr = np.concatenate([np.asarray(s[xkey])[np.random.choice(np.arange(len(s)), \
                                                                        size=samples_per_posterior)] \
                                   for s in plot_list])
        try:
            xlab = gw_utils.param_labels[xkey]
        except:
            xlab = xkey

        # now make histogram (2d if ykey given, else 1d)
        plt.figure(figsize=figsize)
        plt.title(title)
        if isinstance(ykey, str):
            if samples_per_posterior is None:
                yarr = np.concatenate([np.asarray(s[ykey]) for s in plot_list])
            else:
                yarr = np.concatenate([np.asarray(s[ykey])[np.random.choice(np.arange(len(s)), \
                                                                            size=samples_per_posterior)] \
                                       for s in plot_list])
            try:
                ylab = gw_utils.param_labels[ykey]
            except:
                ylab = ykey
            hist_out = plt.hist2d(xarr, yarr, bins=bins, cmap=cmap, **hist_kwargs)
            plt.ylabel(ylab)
            plt.ylim(ylim)
        else:
            hist_out = plt.hist(xarr, bins=bins, **hist_kwargs)
        plt.xlim(xlim)
        plt.xlabel(xlab)
        return hist_out
        

######## CLASS ALIAS ########
POPPE = PopulationParameterEstimation
########################################################################################
