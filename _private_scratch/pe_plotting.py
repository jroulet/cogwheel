import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy as dcopy
import os
import h5py
from matplotlib.colors import Normalize as colorsNormalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d import Axes3D
import lalsimulation as lalsim

DICTLIKE_TYPES = [dict, type(pd.DataFrame([{0: 0}])), type(pd.Series({0: 0}))]
def is_dict(check):
    return any([isinstance(check, dtype) for dtype in DICTLIKE_TYPES])

from . import parameter_aliasing as aliasing
from . import standard_intrinsic_transformations as pxform
from . import parameter_label_formatting as label_formatting
PAR_LABELS = label_formatting.param_labels
PAR_UNITS = label_formatting.units

import sys
COGWHEEL_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'cogwheel'))
sys.path.append(COGWHEEL_PATH)
from cogwheel import utils
from cogwheel import gw_utils
from cogwheel import grid as gd
from cogwheel import cosmology as cosmo



def label_from_key(key):
    return PAR_LABELS.get(aliasing.PARKEY_MAP.get(key, key), key)

def printarr(arr, prec=4, pre='', post='', sep='  ', form='f'):
    print(pre + np.array2string(np.asarray(arr), separator=sep,
                                max_line_width=np.inf, threshold=np.inf,
                                formatter={'float_kind':lambda x: f"%.{prec}{form}" % x}) + post)

def fmt(num, prec=4, form='f'):
    formstr = '{:.' + str(prec) + form + '}'
    return formstr.format(num)

########################################
#### CORNER PLOTTING FUNCTIONS
def corner_plot_samples(samps, pvkeys=['mtot', 'q', 'chieff'], title=None,
                        figsize=(9,7), scatter_points=None, weights=None,
                        grid_kws={}, fig=None, ax=None, return_grid=False, **corner_plot_kws):
    """make corner plots"""
    units, plabs = PAR_UNITS, PAR_LABELS
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
    units, plabs = PAR_UNITS, PAR_LABELS
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
        if is_dict(dic):
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

#########################################
#### 4-DIMENSIONAL SAMPLE PLOTTING

def plot_loc3d(samples, title='flat', xlim='auto', ylim='auto', zlim='auto', nstep=2,
               ckey='lnL', clab=None, mask_keys_min={'lnL': 90}, mask_keys_max={},
               plot_kws=None, figsize=(14, 14), titlesize=20, colorbar_kws=None, units='Mpc',
               extra_point_dicts=[], fig=None, ax=None):
    x, y, z = xyzMpc_from_ra_dec_DL(samples['ra'].to_numpy(), samples['dec'].to_numpy(),
                                    samples['DL'].to_numpy())
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
        title = 'Posterior Samples from PE with Isotropic $\\vec{\\chi}_1, \\vec{\\chi}_2$ Prior'
    if (clab is None) or (clab == 'auto'):
        clab = label_from_key(ckey)
    fig, ax = scatter3d(x[mask][::nstep], y[mask][::nstep], z[mask][::nstep], clr[mask][::nstep],
                        title=title, xlab=f'X ({units})', ylab=f'Y ({units})', zlab=f'Z ({units})',
                        clab=clab, xlim=xlim, ylim=ylim, zlim=zlim, titlesize=titlesize,
                        figsize=figsize, plot_kws=plot_kws, colorbar_kws=colorbar_kws, fig=fig, ax=ax)
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
        xlab = label_from_key(xkey)
    if (ylab is None) or (ylab == 'auto'):
        ylab = label_from_key(ykey)
    if (zlab is None) or (zlab == 'auto'):
        zlab = label_from_key(zkey)
    if (clab is None) or (clab == 'auto'):
        clab = label_from_key(ckey)
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

def samples_with_ligo_angles(old_samps, f_ref, keep_new_spins=False):
    new_samps = old_samps.copy()
    ns = len(old_samps)
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

def samples_add_antenna_response(old_samps, det_chars='HLV', tgps=None):
    if tgps is None:
        tgps = np.asarray(old_samps['tgps'])
    ra, dec, psi = [np.asarray(old_samps[k]) for k in ['ra', 'dec', 'psi']]
    for d in det_chars:
        fp, fc = gw_utils.fplus_fcross_detector(d, ra, dec, psi, tgps)
        old_samps[f'fplus_{d}'] = fp
        old_samps[f'fcross_{d}'] = fc
        old_samps[f'antenna_{d}'] = fp**2 + fc**2
    return

def samples_add_cosmo_prior(old_samps):
    z, DL = old_samps['z'], old_samps['DL']
    old_samps['cosmo_prior'] = (1+z)**-4 * (1 - DL/(1+z)*cosmo.dz_dDL(DL))
    return


def combined_hist(samples_list, xkey='mtot', ykey=None, samples_per_posterior=None,
                  keys_min_means={}, keys_max_means={}, keys_min_medians={}, keys_max_medians={},
                  bins=100, cmap='rainbow', title=None, figsize=(10, 10), xlim=None, ylim=None,
                  fig=None, ax=None, **hist_kwargs):
    """
    make 1D or 2D histogram of combined posteriors
    :param samples_list: list with each element a DataFrame of samples
    :param xkey: key for parameter to go on x-axis
    :param ykey: (optional) key for parameter to go on y-axis
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
    :param samples_per_posterior: number of samples to randomly draw from each set of samples
      (default is to use all samples)
    **remaining kwargs for hist or hist2d are for figure and label formatting**
    """
    # remove posteriors with means and medians outside the range
    plot_list = samples_list
    for k, v in keys_min_means.items():
        plot_list = [p for p in plot_list if np.mean(p[k]) > v]
    for k, v in keys_max_means.items():
        plot_list = [p for p in plot_list if np.mean(p[k]) < v]
    for k, v in keys_min_medians.items():
        plot_list = [p for p in plot_list if np.median(p[k]) > v]
    for k, v in keys_max_medians.items():
        plot_list = [p for p in plot_list if np.median(p[k]) < v]

    if samples_per_posterior is None:
        xarr = np.concatenate([np.asarray(s[xkey]) for s in plot_list])
    else:
        xarr = np.concatenate([np.asarray(s[xkey])[
                    np.random.choice(np.arange(len(s)), size=samples_per_posterior)]
                               for s in plot_list])
    # now make histogram (2d if ykey given, else 1d)
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = plt.gca()
    if isinstance(ykey, str):
        if samples_per_posterior is None:
            yarr = np.concatenate([np.asarray(s[ykey]) for s in plot_list])
        else:
            yarr = np.concatenate([np.asarray(s[ykey])[np.random.choice(np.arange(len(s)), \
                                                                        size=samples_per_posterior)] \
                                   for s in plot_list])
        hist_out = ax.hist2d(xarr, yarr, bins=bins, cmap=cmap, **hist_kwargs)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.set_ylabel(label_from_key(ykey))
    else:
        hist_out = ax.hist(xarr, bins=bins, **hist_kwargs)
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.set_xlabel(label_from_key(xkey))
    if title is not None:
        ax.set_title(title)
    return fig, ax, hist_out


##################### DEFAULTS & LVC STUFF #######################
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

######### GETTING LVC SAMPLES ########

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



class LVCsampleHandle(object):
    """
    similar to ParameterEstimationHandle but for LVC sample releases
    for O2 posterior sample releases, maybe easier to add these manually to pe_list and plot via index
    when including in ParameterEstimationComparison (can try automated method, updates in progress)
    """
    default_approximant_priority = ['PrecessingSpinIMRHM', 'IMRPhenomPv3HM', 'NRSur7dq4',
                                    'SEOBNRv4PHM', 'IMRPhenomPv2_posterior']
    def __init__(self, lvc_h5, evname, dataset_name=None, keymap=None, tgps=None,
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
        self.obs_run = run_from_evname(self.evname)
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
                pxform.compute_samples_aux_vars(self.samples)
                if self.prior_samples is not None:
                    try:
                        pxform.compute_samples_aux_vars(self.prior_samples)
                    except:
                        print('posterior parameter conversion successful but failure in priors')
            except:
                print('ERROR: could not convert parameters with pxform.compute_samples_aux_vars')

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
        self.name, self.label = [f'LVC: {self.approximant}'] * 2
        if not no_print:
            print(self.Nsamples, 'samples from', dataset_name)
        return

    def sample_completion(self, new_samples=None, **gwpdic_kwargs):
        """use GWPD init completion to compute all parameter conversions"""
        # see if passing anything new, otherwise start with existing samples
        raise NotImplementedError('need to finish cogwheel version of gw_parameter_dictionary')
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
        #self.samples = GWPD(par_dic=new_samples, **kws).dataframe()


    def get_best_samples(self, key_rngs={}, get_best_inds=0, lnl_col='lnL', keys_to_get_as_par_dics=None):
        s = self.samples[np.isnan(self.samples[lnl_col]) == False]
        for k, rng in key_rngs.items():
            s = s[s[k] > rng[0]]
            s = s[s[k] < rng[1]]
        s = s.sort_values(lnl_col, ascending=False).reset_index().iloc[get_best_inds]
        if keys_to_get_as_par_dics is not None:
            s = s[keys_to_get_as_par_dics]
            if hasattr(get_best_inds, '__len__'):
                return [dict(idx_row[1]) for idx_row in s.iterrows()]
            return dict(s)
        return s

    @classmethod
    def from_evname(cls, evname, approximants=SLM_APPROX_LISTS['PHM'], samples_base_dir=None,
                    keymap=None, tgps=None, compute_aux_O3=False, convert_angles=False,
                    get_h5=False, gwpdic_keymap=aliasing.PARKEY_MAP, no_print=False,
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
            ret = cls(lvc_h5, evname, approximants, keymap=keymap, tgps=tgps,
                    compute_aux_O3=compute_aux_O3, convert_angles=convert_angles, gwpdic_keymap=gwpdic_keymap,
                    no_print=no_print, keep_unmatched_keys=keep_unmatched_keys, try_dataset_variants=True)
        else:
            dsns = approximants + ['C01:' + apr for apr in approximants if isinstance(apr, str)] \
                    + [apr + '_posterior' for apr in approximants if isinstance(apr, str)]
            ret = [cls(lvc_h5, evname, dsn, keymap=keymap, tgps=tgps,
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
        instance = cls(lvc_h5, evname, dataset_name, keymap=keymap,
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
        units, plabs = PAR_UNITS, PAR_LABELS
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