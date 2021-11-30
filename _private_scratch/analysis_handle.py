"""
Get samples, likelihood, and prior objects from parameter estimation run
after post-processing for data analysis and visualization.
"""
import numpy as np
import os
import pathlib
import sys
import json
import pandas as pd
from copy import deepcopy as dcopy

from . import parameter_aliasing as aliasing
from . import parameter_label_formatting as label_formatting
from . import standard_intrinsic_transformations as pxform
from . import pe_plotting as peplot

COGWHEEL_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'cogwheel'))
sys.path.append(COGWHEEL_PATH)
from cogwheel import utils
from cogwheel import sampling
from cogwheel import grid as gd
from cogwheel import cosmology as cosmo
from cogwheel import postprocessing

DEFAULT_PRIOR = 'IASPrior'
DEFAULT_PARENTDIR = '/data/srolsen/GW/cogwheel/o3a_cands/'

def key_rngs_mask(df_to_mask, key_rngs={}, keep_nans=False):
    """
    Mask for samples satisfying rng[0] < samples[k] < rng[1]
    for k, rng in key_rngs.items().
    If key_rngs is empty, return mask to select samples with no NaNs.
    Set keep_nans=True to allow samples with NaNs to be considered.
    """
    mask0 = (np.ones(len(df_to_mask), dtype=bool) if keep_nans
             else (df_to_mask.isna().any(axis=1) == False))
    if not key_rngs:
        return mask0
    samps = df_to_mask[mask0]
    mask = mask0.copy()
    for k, rng in key_rngs.items():
        mask[mask0] = (mask[mask0] & (samps[k] > rng[0])
                       & (samps[k] < rng[1]))
    return mask


class AnalysisHandle:
    """Class for analyzing posteriors."""
    LNL_COL = 'lnl'
    KEYMAP = aliasing.PARKEY_MAP
    PAR_LABELS = label_formatting.param_labels
    PAR_UNITS = label_formatting.units
    PAR_NAMES = label_formatting.param_names
    
    def __init__(self, rundir, name=None, separate_nans=True,
                 complete_samples=False, add_all=False):
        """
        If rundir has `run` in final path layer then will get samples
        Otherwise treat rundir as an eventdir and make the samples just
        a dataframe with the reference waveform's parameters.
        """
        super().__init__()
        # associate to run directory
        self.rundir = pathlib.Path(rundir)
        self.name = name or self.rundir.parts[-1]

        # load posterior attributes
        if 'run' not in self.rundir.parts[-1]:
            self.sampler = None
            self.posterior = utils.read_json(self.rundir/'Posterior.json')
        else:
            self.sampler = utils.read_json(self.rundir/sampling.Sampler.JSON_FILENAME)
            self.posterior = self.sampler.posterior
        # make these references for direct access
        self.likelihood = self.posterior.likelihood
        self.prior = self.posterior.prior
        self.evdata = self.likelihood.event_data
        self.wfgen = self.likelihood.waveform_generator
        self.evname = self.evdata.eventname

        # load samples
        if self.sampler is None:
            self.samples_path = None
            self.samples = pd.DataFrame([{self.LNL_COL: self.likelihood._lnl_0,
                                          **self.likelihood.par_dic_0,
                                          **self.prior.inverse_transform(
                                              **self.likelihood.par_dic_0)}])
        else:
            self.samples_path = self.rundir/sampling.SAMPLES_FILENAME
            self.samples = pd.read_feather(self.samples_path)
        if complete_samples or add_all:
            self.complete_samples(add_all=add_all)

        # check likelihood information
        if self.LNL_COL not in self.samples:
            self.LNL_COL = self.key(self.LNL_COL)
        if self.LNL_COL not in self.samples:
            print('WARNING: No likelihood information found in samples. Setting lnL = 0.')
            self.samples[self.LNL_COL] = np.zeros(len(self.samples))
            self.best_par_dic = None
        else:
            self.best_par_dic = self.get_best_par_dics()

        # separate samples with NaNs
        nan_mask = self.samples.isna().any(axis=1)
        self.nan_samples = dcopy(self.samples[nan_mask])
        if separate_nans:
            self.samples = self.samples[nan_mask == False].reset_index(drop=True)
            self.nan_samples = self.nan_samples.reset_index(drop=True)

        # see if the keymap is faithful to samples
        if not all([self.key(k) == k for k in self.samples.columns]):
            print('WARNING: Abandoning self.KEYMAP due to inconsistency with samples.')
            self.KEYMAP = {}

        #load test results if they're there
        self.tests_path = self.rundir/postprocessing.TESTS_FILENAME
        self.tests_dict = None
        if os.path.isfile(self.tests_path):
            self.tests_dict = json.load(open(self.tests_path, 'r'))

    @classmethod
    def from_evname(cls, evname, i_run=0, parentdir=DEFAULT_PARENTDIR,
                    prior_name=DEFAULT_PRIOR, **init_kwargs):
        evdir = utils.get_eventdir(parentdir=parentdir, prior_name=prior_name,
                                   eventname=evname)
        return cls(os.path.join(evdir, f'run_{i_run}'), **init_kwargs)

    #######################
    ##  KEYS and LABELS  ##
    #######################
    def key(self, key):
        return self.KEYMAP.get(key, key)

    def par_label(self, key):
        return self.PAR_LABELS.get(self.key(key), key)

    def par_name(self, key):
        return self.PAR_NAMES.get(self.key(key), key)

    def par_unit(self, key):
        return self.PAR_UNITS.get(self.key(key), key)

    #######################
    ##  MASKING SAMPLES  ##
    #######################
    def mask(self, key_rngs={}, keep_nans=False):
        """
        Mask for samples satisfying rng[0] < samples[self.key(k)] < rng[1]
        for k, rng in key_rngs.items().
        If key_rngs is empty, return mask to select samples with no NaNs.
        Set keep_nans=True to allow samples with NaNs to be considered.
        """
        return key_rngs_mask(self.samples,
                             {self.key(k): v for k, v in key_rngs.items()},
                             keep_nans)

    def masked_samples(self, key_rngs={}, keep_nans=False):
        """
        Get samples without NaNs (unless keep_nans=True) and
        satisfying rng[0] < samples[self.key(k)] < rng[1]
        for k, rng in key_rngs.items().
        """
        return self.samples[self.mask(key_rngs, keep_nans)]

    #######################
    ##  GETTING PAR_DIC  ##
    #######################
    def get_par_dic(self, par_dic=None):
        """
        Returns a dict with the keys in self.wfgen.params.
        Default par_dic=None will return self.likelihood.par_dic_0
        If par_dic is a dict-like object, the correct params are isolated
        Else can pass np.array([<values ordered as in self.wfgen.params>])
        Else can pass an integer to get row from self.samples
        NOTE if you pass a scalar that is not integer-like
        OR an iterable that is not a numpy array, it will be
        treated as a dict-like object and an error will occur.
        """
        if par_dic is None:
            return self.likelihood.par_dic_0
        if not hasattr(par_dic, '__len__'):
            return dict(self.samples[self.wfgen.params].iloc[par_dic])
        if isinstance(par_dic, np.ndarray):
            assert len(par_dic) == len(self.wfgen.params), \
                f'If par_dic is an array, form must follow {self.wfgen.params}'
            return dict(zip(self.wfgen.params, par_dic))
        return {k: par_dic[k] for k in self.wfgen.params}

    def get_best_par_dics(self, key_rngs={}, get_best_inds=0, as_par_dics=True):
        """
        Get index/indices in get_best_inds from samples sorted by likelihood.
        If as_par_dics=True, output will be dict(s) with the keys from self.wfgen.params,
        Otherwise output will be pandas Series/DataFrame with all columns.
        Use key_rngs[key] = (min_val_for_key, max_val_for_key) to filter the
        considered samples by parameter ranges.
        """
        s = self.masked_samples(key_rngs).sort_values(
                self.LNL_COL, ascending=False).reset_index(drop=True).iloc[get_best_inds]
        if as_par_dics:
            if hasattr(get_best_inds, '__len__'):
                return [self.get_par_dic(idx_row[1]) for idx_row in s.iterrows()]
            return self.get_par_dic(s)
        return s

    ##################
    ##  LIKELIHOOD  ##
    ##################
    def lnL(self, pdic_or_ind=None, use_relative_binning=False,
            bypass_relative_binning_tests=True):
        """
        Defaults to returning log likelihood (lnL) of reference waveform.
        If pdic_or_ind is any string, get array of lnL for all samples.
        If pdic_or_ind is an int, compute lnL for sample at that index.
        Otherwise pdic_or_ind can be a numpy array ordered as in
        self.wfgen.params or a dict-like containing at least those keys.
        This, i.e., self.get_par_dic(pdic_or_ind), will be passed to
        self.likelihood.lnlike() if use_relative_binning else
        self.likelihood.lnlike_fft().
        """
        if isinstance(pdic_or_ind, str):
            return self.samples[self.LNL_COL].to_numpy()
        if use_relative_binning:
            return self.likelihood.lnlike(self.get_par_dic(pdic_or_ind),
                bypass_tests=bypass_relative_binning_tests)
        return self.likelihood.lnlike_fft(self.get_par_dic(pdic_or_ind))

    def lnL_dets(self, pdic_or_ind):
        """
        Return log likelihood at all detectors of parameters for
        par_dic = self.get_par_dic(pdic_or_ind)
        WITHOUT applying any ASD drift.
        See self.get_par_dic for accepted input formats.
        """
        h_f = self.likelihood._get_h_f(self.get_par_dic(pdic_or_ind))
        h_h = self.likelihood._compute_h_h(h_f)
        d_h = self.likelihood._compute_d_h(h_f)
        return d_h - .5*h_h

    #########################
    ##  SAMPLE COMPLETION  ##
    #########################
    def complete_samples(self, antenna=False, cosmo_weights=False,
                         ligo_angles=False, add_all=False):
        """
        Complete samples with self.add_source_parameters() and other options
        (self.add_ligo_angles, self.add_antenna, self.add_cosmo_weights).
        TODO: still need a way to get these peplot functions to use self.KEYMAP
        """
        self.add_source_parameters()
        if add_all:
            antenna, cosmo_weights, ligo_angles = True, True, True
        if ligo_angles:
            self.add_ligo_angles()
        if antenna:
            self.add_antenna()
        if cosmo_weights:
            self.add_cosmo_weights()

    def write_complete_samples(self, fname=None, overwrite=False, antenna=False,
                               cosmo_weights=False, ligo_angles=False):
        """
        Complete samples with self.add_source_parameters() and other options,
        then write new samples to path given by fname (Defaults to self.samples_path).
        """
        if fname is None:
            fname = self.samples_path
        if os.path.exists(fname) and (not overwrite):
            raise FileExistsError(f'Set overwrite=True to overwrite {fname}')
        self.complete_samples(antenna=antenna, cosmo_weights=cosmo_weights, ligo_angles=ligo_angles)
        self.samples.to_feather(self.samples_path)

    def add_source_parameters(self, redshift_key=None, mass_keys=['m1', 'm2', 'mtot', 'mchirp']):
        """
        Add _source version of each mass in mass_keys using *= 1+self.samples[redshift_key].
        If redshift_key is None, do intrinsic parameter completion with pxform.compute_samples_aux_vars().
        """
        if redshift_key is None:
            # this completes intrinsic parameter space and adds redshift and source frame information
            pxform.compute_samples_aux_vars(self.samples)
            return
        rkey = self.key(redshift_key)
        if rkey not in self.samples:
            self.samples[rkey] = cosmo.z_of_DL_Mpc(self.samples['d_luminosity'])
        for k in mass_keys:
            self.samples[self.key(k)+'_source'] = self.samples[self.key(k)] / (1+self.samples[rkey])

    def add_ligo_angles(self, keep_new_spins=False):
        """
        Add angular variables from lalsimulation.SimInspiralTransformPrecessingWvf2PE
        with the option to also keep the spin variables output by this function
        (which should be equal to the existing spin parameters if conventions align).
        Keys are `thetaJN`, `phiJL`, `phi12`.
        If keep_new_spins=True, will also replace `s1`, `s1theta`, `s2`, `s2theta`.
        """
        self.samples = peplot.samples_with_ligo_angles(self.samples, self.wfgen.f_ref,
                                                       keep_new_spins=keep_new_spins)

    def add_antenna(self):
        """
        Add antenna responses F_+, F_x, F_+^2 + F_x^2 for each detector.
        Keys are `fplus_{det_char}`, `fcross_{det_char}`, `antenna_{det_char}`.
        """
        peplot.samples_add_antenna_response(self.samples, det_chars=self.evdata.detector_names,
                                            tgps=self.evdata.tgps)

    def add_cosmo_weights(self):
        """
        Add weights for prior reweighting from uniform luminosity volume to comoving volume.
        Key is `cosmo_weight`.
        """
        peplot.samples_add_cosmo_weight(self.samples)

    #########################
    ##  WAVEFORM PLOTTING  ##
    #########################
    def plot_psd(self, ax=None, fig=None, label=None, plot_type='loglog',
                 weights=None, plot_asd=False, xlim=None, ylim=None, title=None,
                 figsize=None, use_fmask=False, **plot_kws):
        """Plot PSD at all detectors"""
        msk = (self.evdata.fslice if use_fmask else slice(None))
        dets_xplot = self.evdata.frequencies[msk]
        dets_yplot = self.evdata.psd[..., msk]
        ylabel = 'Power Spectral Density'
        if plot_asd:
            dets_yplot = np.sqrt(dets_yplot)
            ylabel = 'Amplitude Spectral Density'
        if weights is not None:
            dets_yplot *= weights[msk]
            ylabel = 'Weighted ' + ylabel
        return peplot.plot_at_dets(dets_xplot, dets_yplot, ax=ax, fig=fig, label=label,
                                   xlabel='Frequency (Hz)', ylabel=ylabel, plot_type=plot_type,
                                   xlim=xlim, ylim=ylim, title=title, det_names=self.evdata.detector_names,
                                   figsize=figsize, **plot_kws)

    def plot_wf_amp(self, par_dic=None, whiten=True, by_m=False, ax=None, fig=None, label=None,
                    plot_type='loglog', weights=None, xlim=None, ylim=None,
                    title=None, figsize=None, use_fmask=False, **plot_kws):
        """Plot waveform amplitude at all detectors"""
        msk = (self.evdata.fslice if use_fmask else slice(None))
        dets_xplot = self.evdata.frequencies[msk]
        h_f = self.likelihood._get_h_f(par_dic, by_m=by_m)
        if whiten:
            h_f = self.evdata.dt * np.fft.rfft(self.likelihood._get_whitened_td(h_f), axis=-1)
        if weights is not None:
            h_f *= weights
        if by_m:
            for j, lmlist in enumerate(self.wfgen._harmonic_modes_by_m.values()):
                dets_yplot = np.abs(h_f[j, :, msk])
                fig, ax = peplot.plot_at_dets(dets_xplot, dets_yplot, ax=ax, fig=fig, label=str(lmlist),
                    xlabel='Frequency (Hz)', ylabel='Waveform Amplitude', plot_type=plot_type,
                    xlim=xlim, ylim=ylim, title=title, det_names=self.evdata.detector_names,
                    figsize=figsize, **plot_kws)
            return fig, ax
        return peplot.plot_at_dets(dets_xplot, np.abs(h_f[:, msk]), ax=ax, fig=fig, label=label,
                                   xlabel='Frequency (Hz)', ylabel='Waveform Amplitude',
                                   plot_type=plot_type, xlim=xlim, ylim=ylim, title=title,
                                   det_names=self.evdata.detector_names, figsize=figsize, **plot_kws)

    def plot_wf_phase(self, par_dic, unwrap=True, by_m=False, ax=None, fig=None, label=None,
                      plot_type='linear', weights=None, xlim=None, ylim=None,
                      title=None, figsize=None, use_fmask=False, **plot_kws):
        """Plot waveform phase at all detectors"""
        msk = (self.evdata.fslice if use_fmask else slice(None))
        dets_xplot = self.evdata.frequencies[msk]
        h_f = self.likelihood._get_h_f(par_dic, by_m=by_m)
        if weights is not None:
            h_f *= weights
        func = (np.unwrap if unwrap else (lambda x: x))
        if by_m:
            for j, lmlist in enumerate(self.wfgen._harmonic_modes_by_m.values()):
                fig, ax = peplot.plot_at_dets(dets_xplot, func(np.angle(h_f[j, :, msk])), ax=ax, fig=fig,
                                label=str(lmlist), xlabel='Frequency (Hz)', ylabel='Waveform Phase (rad)',
                                plot_type=plot_type, xlim=xlim, ylim=ylim, title=title,
                                det_names=self.evdata.detector_names, figsize=figsize, **plot_kws)
            return fig, ax
        return peplot.plot_at_dets(dets_xplot, func(np.angle(h_f[:, msk])), ax=ax, fig=fig, label=label,
                                   xlabel='Frequency (Hz)', ylabel='Waveform Phase (rad)',
                                   plot_type=plot_type, xlim=xlim, ylim=ylim, title=title,
                                   det_names=self.evdata.detector_names, figsize=figsize, **plot_kws)

    def plot_whitened_wf(self, par_dic=None, trng=(-.7, .1), **kwargs):
        """
        par_dic can be None (take self.likelihood.par_dic_0), dict-like,
        int (index in self.samples), or array (form as in self.wfgen.params).
        """
        return self.likelihood.plot_whitened_wf(self.get_par_dic(par_dic), trng=trng, **kwargs)

    #######################
    ##  CORNER PLOTTING  ##
    #######################
    def corner_plot(self, parkeys=['mchirp', 'q', 'chieff'], weights=None,
                    key_rngs={}, extra_grid_kwargs={}, **corner_plot_kwargs):
        """
        Make corner plot of self.samples for the parameter keys in parkeys.

        **corner_plot_kwargs can include anything (except pdf) from
          Grid.corner_plot(pdf=None, title=None, subplot_size=2., fig=None, ax=None,
                figsize=None, nbins=6, set_legend=True, save_as=None, y_title=.98,
                plotstyle=None, show_titles_1d=True, scatter_points=None, **kwargs)
        --> NAMELY, pass fig=myfig, ax=myax to plot with existing axes

        weights can be an array of weights or a key to use from self.samples
        scatter_points (corner_plot_kwargs) can be DataFrame of extra samples to plot
        """
        samps = self.masked_samples(key_rngs)
        if isinstance(weights, str):
            weights = samps[self.key(weights)]
        pdfnm = f'{self.evname}: {self.name}\n{len(samps)} samples'
        corner_plot_kwargs['set_legend'] = corner_plot_kwargs.get('set_legend', True)
        if 'title' not in corner_plot_kwargs:
            corner_plot_kwargs['title'] = pdfnm
            pdfnm = None
        return gd.Grid.from_samples([self.key(k) for k in parkeys], samps,
                                    pdf_key=pdfnm, units=self.PAR_UNITS, labels=self.PAR_LABELS, weights=weights,
                                    **extra_grid_kwargs).corner_plot(pdf=pdfnm, **corner_plot_kwargs)

    def corner_plot_comparison(self, compare_posteriors=[], compare_names=[],
                               parkeys=['mtot', 'q', 'chieff'], weight_key=None,
                               key_rngs={}, extra_grid_kwargs={}, multigrid_kwargs={},
                               return_grid=False, **corner_plot_kwargs):
        """
        Make corner plot for the parameter keys in parkeys comparing self.samples
        with the posteriors in compare_posteriors (list of pd.DataFrame objects),
        labeling them by the corresponding elements of compare_names.

        **corner_plot_kwargs can include anything (except pdf) from
          Grid.corner_plot(pdf=None, title=None, subplot_size=2., fig=None, ax=None,
                figsize=None, nbins=6, set_legend=True, save_as=None, y_title=.98,
                plotstyle=None, show_titles_1d=True, scatter_points=None, **kwargs)
        --> NAMELY, pass fig=myfig, ax=myax to plot with existing axes

        weight_key is key (str) to use for reweighting samples.
        scatter_points (corner_plot_kwargs) can be DataFrame of extra samples to plot.
        return_grid=True will return the multigrid along with the figure and axes.
        multigrid_kwargs will be unpacked into MultiGrid.from_grids().
        extra_grid_kwargs will be unpacked into Grid.from_samples().
        """
        comp_names = dcopy(compare_names)
        if len(comp_names) < len(compare_posteriors):
            comp_names += [''] * (len(compare_posteriors) - len(comp_names))
        for j in range(len(compare_posteriors)):
            if isinstance(compare_posteriors[j], AnalysisHandle):
                if comp_names[j] == '':
                    comp_names[j] = compare_posteriors[j].name
                compare_posteriors[j] = compare_posteriors[j].samples
        return peplot.corner_plot_list(([self.masked_samples(key_rngs)] +
                                        [s[key_rngs_mask(s, key_rngs)] for s in compare_posteriors]),
                                       [self.name] + comp_names, pvkeys=parkeys, weight_key=weight_key,
                                       grid_kws=extra_grid_kwargs, multigrid_kws=multigrid_kwargs,
                                       return_grid=return_grid, **corner_plot_kwargs)

    ###############################################################
    ##  SCATTER PLOTTING WITH COLOR+SIZE+TRANSPARENCY GRADIENTS  ##
    ###############################################################
    def plot_2d_color(self, xkey='chieff', ykey='q', ckey='lnl', extra_posteriors=[], key_rngs=None,
                      samples_per_posterior=None, fig=None, ax=None, figsize=(8, 8), title=None,
                      titlesize=20, xlim='auto', ylim='auto', clim=None, size_key=None, size_scale=1,
                      alpha_key=None, alpha_scale=1, colorbar_kws=None, colorsMap='jet', **plot_kws):
        """Make two-dimensional scatter plot with colorbar for visualizing third dimension."""
        return peplot.plot_samples2d_color(([self.masked_samples(key_rngs)] +
                                            [s[key_rngs_mask(s, key_rngs)] for s in extra_posteriors]),
            xkey=self.key(xkey), ykey=self.key(ykey), ckey=self.key(ckey), fig=fig, ax=ax, figsize=figsize,
            samples_per_posterior=samples_per_posterior, colorbar_kws=colorbar_kws, colorsMap=colorsMap,
            title=title, titlesize=titlesize, xlim=xlim, ylim=ylim, clim=clim, size_key=size_key,
            size_scale=size_scale, alpha_key=alpha_key, alpha_scale=alpha_scale, **plot_kws)

    def plot_3d_color(self, xkey='chieff', ykey='q', zkey='mtot', ckey='lnl', key_rngs=None,
                      nstep=1, fig=None, ax=None, xlim='auto', ylim='auto', zlim='auto',
                      xlab='auto', ylab='auto', zlab='auto', clab='auto', title=None,
                      plot_kws=None, figsize=(8, 8), titlesize=20, colorbar_kws=None,
                      extra_point_dicts=[], size_key=None, size_scale=1):
        """
        Make three-dimensional scatter plot with colorbar for visualizing fourth dimension.
        Additional gradient dimensions are size (size_key)
        """
        return peplot.plot_samples4d(self.masked_samples(key_rngs), xkey=self.key(xkey), ykey=self.key(ykey),
            zkey=self.key(zkey), ckey=self.key(ckey), xlim=xlim, ylim=ylim, zlim=zlim,
            nstep=nstep, title=title, xlab=xlab, ylab=ylab, zlab=zlab, clab=clab, fig=fig, ax=ax,
            figsize=figsize, titlesize=titlesize, extra_point_dicts=extra_point_dicts,
            size_key=size_key, size_scale=size_scale, plot_kws=plot_kws, colorbar_kws=colorbar_kws)

    #####################
    ##  SPIN PLOTTING  ##
    #####################
    def plot_inplane_spin(self, color_key='lnl', use_V3=False, secondary_spin=False,
                          key_rngs=None, fractions=[.5, .95], plotstyle_color='r', scatter_alpha=.5,
                          figsize=None, title=None, tight=False, **colorbar_kws):
        """
        Plot constituent spin posterior projected onto the plane of the orbit with colorbar.
        Defaults to primary BH, use secondary_spin=True to plot spin of the secondary BH.
        """
        return peplot.plot_inplane_spin(self.masked_samples(key_rngs), color_key=self.key(color_key),
                                        use_V3=use_V3, secondary_spin=secondary_spin, fractions=fractions,
                                        plotstyle_color=plotstyle_color, scatter_alpha=scatter_alpha,
                                        figsize=figsize, title=title, tight=tight, **colorbar_kws)

    def plot_3d_spin(self, ckey='lnl', use_V3=False, secondary_spin=False, sign_or_scale=True,
                     key_rngs=None, fig=None, ax=None, xkey='s1x', ykey='s1y', zkey='s1z',
                     nstep=1, title=None, xlab='auto', ylab='auto', zlab='auto', clab='auto',
                     plotlim=[-1.01, 1.01], plot_kws=None, figsize=(8, 8), titlesize=20,
                     colorbar_kws=None, extra_point_dicts=[(0, 0, 0)],
                     marker_if_not_dict='o', size_if_not_dict=20, color_if_not_dict='k', ):
        """
        Plot constituent spin posterior in three-dimensional space with colorbar
        and unit sphere wire frame option (default).
        Defaults to primary BH, use secondary_spin=True to plot spin of the secondary BH.
        """
        return peplot.plot_spin4d(self.masked_samples(key_rngs), use_V3=use_V3, secondary_spin=secondary_spin,
            sign_or_scale=sign_or_scale, xkey=self.key(xkey), ykey=self.key(ykey), plotlim=plotlim,
            zkey=self.key(zkey), ckey=self.key(ckey), nstep=nstep, title=title, titlesize=titlesize,
            xlab=xlab, ylab=ylab, zlab=zlab, clab=clab, fig=fig, ax=ax, extra_point_dicts=extra_point_dicts,
            figsize=figsize, plot_kws=plot_kws, colorbar_kws=colorbar_kws, marker_if_not_dict=marker_if_not_dict,
            size_if_not_dict=size_if_not_dict, color_if_not_dict=color_if_not_dict)

    #########################
    ##  LOCATION PLOTTING  ##
    #########################
    def plot_3d_location(self, fig=None, ax=None, ckey='lnl', key_rngs=None, nstep=1,
                         clab=None, extra_point_dicts=[], title=None, units='Mpc',
                         figsize=(8, 8), xlim='auto', ylim='auto', zlim='auto',
                         titlesize=20, plot_kws=None, colorbar_kws=None):
        """
        Plot posteriors in physical space using luminosity distance and RA/DEC.
        Color points by value of samples[ckey].
        Use key_rngs={k: (vlo, vhi), ...} to put bounds (vlo, vhi) on samples[k].
        """
        return peplot.plot_loc3d(self.masked_samples(key_rngs), title=title, xlim=xlim, ylim=ylim, zlim=zlim,
                                 nstep=nstep, ckey=ckey, clab=clab, plot_kws=plot_kws, figsize=figsize,
                                 titlesize=titlesize, colorbar_kws=colorbar_kws, units=units,
                                 extra_point_dicts=extra_point_dicts, fig=fig, ax=ax)


