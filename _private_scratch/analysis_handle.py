"""
Get samples, likelihood, and prior objects from parameter estimation run
after post-processing for data analysis and visualization.
"""
import numpy as np
import os
import pathlib
import sys
import pandas as pd
from copy import deepcopy as dcopy

from matplotlib.colors import Normalize as colorsNormalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d import Axes3D
#import h5py


import parameter_aliasing as aliasing

COGWHEEL_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'cogwheel'))
sys.path.append(COGWHEEL_PATH)
from cogwheel import utils
from cogwheel import sampling
from cogwheel import grid as gd

class AnalysisHandle:
    """Class for analyzing posteriors."""
    LNL_COL = 'lnl'

    KEYMAP = aliasing.PARKEY_MAP
    PAR_LABELS = aliasing.param_labels
    PAR_UNITS = aliasing.units
    PAR_NAMES = aliasing.param_names
    
    def __init__(self, rundir, name=None):
        super().__init__()
        # associate to run directory
        self.rundir = pathlib.Path(rundir)
        self.name = name or self.rundir.parts[-1]
        # load samples
        self.samples = pd.read_feather(self.rundir/sampling.SAMPLES_FILENAME)
        if self.LNL_COL not in self.samples:
            self.LNL_COL = self.key(self.LNL_COL)
        if self.LNL_COL not in self.samples:
            print('WARNING: No likelihood information found in samples.')
        if not all([self.key(k) == k for k in self.samples.columns]):
            print('WARNING: Abandoning self.KEYMAP due to inconsistency with samples.')
            self.KEYMAP = {}
        
        # load posterior attributes
        sampler = utils.read_json(self.rundir/sampling.Sampler.JSON_FILENAME)
        self.likelihood = dcopy(sampler.posterior.likelihood)
        self.prior = dcopy(sampler.posterior.prior)
        # make these references for direct access
        self.evdata = self.likelihood.event_data
        self.wfgen = self.likelihood.waveform_generator
        self.evname = self.evdata.eventname

    def key(self, key):
        return self.KEYMAP.get(key, key)

    def par_label(self, key):
        return self.PAR_LABELS.get(self.key(key), key)

    def par_name(self, key):
        return self.PAR_NAMES.get(self.key(key), key)

    def par_unit(self, key):
        return self.PAR_UNITS.get(self.key(key), key)

    def get_best_par_dics(self, key_rngs={}, get_best_inds=0, as_par_dics=True):
        s = self.samples[np.isnan(self.samples[self.LNL_COL]) == False]
        for k, rng in key_rngs.items():
            s = s[s[self.key(k)] > rng[0]]
            s = s[s[self.key(k)] < rng[1]]
        s = s.sort_values(self.LNL_COL, ascending=False).reset_index().iloc[get_best_inds]
        if as_par_dics:
            s = s[[self.key(k) for k in self.wfgen.params]]
            if hasattr(get_best_inds, '__len__'):
                return [dict(idx_row[1]) for idx_row in s.iterrows()]
            return dict(s)
        return s

    def corner_plot(self, parkeys=['mchirp', 'q', 'chieff'],
                    weights=None, scatter_points=None,
                    extra_grid_kwargs={}, **corner_plot_kwargs):
        """
        Make corner plot of self.samples for the parameter keys in parkeys.

        **corner_plot_kwargs can include anything (except pdf) from
          Grid.corner_plot(pdf=None, title=None, subplot_size=2., fig=None, ax=None,
                figsize=None, nbins=6, set_legend=True, save_as=None, y_title=.98,
                plotstyle=None, show_titles_1d=True, scatter_points=None, **kwargs)
        --> NAMELY, pass fig=myfig, ax=myax to plot with existing axes

        weights can be an array of weights or a key to use from self.samples
        scatter_points can be a dataframe of extra samples to plot
        """
        if isinstance(weights, str):
            weights = self.samples[self.key(weights)]
        pdfnm = f'{self.evname}: {self.name}\n{len(self.samples)} samples'
        corner_plot_kwargs['set_legend'] = corner_plot_kwargs.get('set_legend', True)
        if 'title' not in corner_plot_kwargs:
            corner_plot_kwargs['title'] = pdfnm
            pdfnm = None
        return gd.Grid.from_samples([self.key(k) for k in parkeys],
            self.samples, pdf_key=pdfnm, units=self.PAR_UNITS,
            labels=self.PAR_LABELS, weights=weights,
            **extra_grid_kwargs).corner_plot(pdf=pdfnm, **corner_plot_kwargs)

    



