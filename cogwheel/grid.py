from copy import copy, deepcopy
import json
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

# -----------------------------------------------------------
# Plot settings

class PlotStyle1d:
    """Arguments for 1d histograms in a corner plot's diagonal."""
    DEFAULT_KWARGS = {'color': 'C0'}
    def __init__(self, alpha_vlines=.5, lw_vlines=1, alpha_fill=.1,
                 step=False, **kwargs):
        self.kwargs = self.DEFAULT_KWARGS | kwargs
        self.alpha_vlines = alpha_vlines
        self.lw_vlines = lw_vlines
        self.alpha_fill = alpha_fill
        self.step = step


class PlotStyle2d:
    """Arguments for 2d histograms in a corner plot's off-diagonal."""
    DEFAULT_FRACTIONS = .5, .9

    def __init__(self, color='k', linestyles='-', linewidths=1,
                 fill='gradient', fractions=None, show_cl=False,
                 clabel_fs=8, clabel_colors=None, contour_kwargs=None):
        """
        Store the 2d plot settings.
        2d pdfs are displayed as surface density (fill) and
        probability contours.

        Parameters:
        -----------
        color: Color of both the fill and contours
        linestyles: of the contours
        linewidths: of the contours
        alphas: of the filling, when `fill` is 'flat'
        fill: 'gradient', 'flat' or 'none'.
            'gradient' displays the pdf using a transparency gradient
            'flat' fills the contours with a flat color
            'none' shows just the contours
        fractions: list of fraction of the pdf to enclose by the
                   contours when plotting.
        show_cl: Flag, whether to show the value of the confidence level
                 on the contour
        clabel_fs: Fontsize of the contour label, used if
                   `show_cl == True`.
        clabel_colors: Color of the contour label, used if
                       `show_cl == True`.
        """
        self.color = color
        self.linestyles = linestyles
        self.linewidths = linewidths
        self.fill = fill
        self.alphas = None  # Set by `fractions` by default
        self.fractions = fractions
        self.show_cl = show_cl
        self.clabel_fs = clabel_fs
        self.clabel_colors = clabel_colors
        self.contour_kwargs = contour_kwargs or {}

    @property
    def fractions(self):
        return self._fractions

    @fractions.setter
    def fractions(self, fractions):
        fractions = self.DEFAULT_FRACTIONS if fractions is None else fractions
        self._fractions = sorted(fractions, reverse=True)
        self.alphas = [1 - fraction for fraction in self.fractions]


class PlotStyle:
    def __init__(self, plotstyle_1d, plotstyle_2d):
        self.plotstyle_1d = plotstyle_1d
        self.plotstyle_2d = plotstyle_2d

    @classmethod
    def get_many(cls, number, fractions=None):
        linestyles = cls._gen_linestyles(number)
        colors = cls._gen_colors(number)

        return [cls(PlotStyle1d(color=color, alpha_vlines=0, alpha_fill=0),
                    PlotStyle2d(color, linestyle, fill='flat',
                                fractions=fractions))
                for color, linestyle in zip(colors, linestyles)]

    @classmethod
    def _gen_linestyles(cls, number):
        if number <= len(linestyles := ['-', '--', '-.', ':']):
            return linestyles[:number]
        return ['-'] + [(0, tuple([2, 2]*i + [7, 2])) for i in range(number-1)]

    @classmethod
    def _gen_colors(cls, number):
        colors = (mpl.cm.get_cmap('tab20').colors
                  + mpl.cm.get_cmap('tab20b').colors)
        colors = colors[::2] + colors[1::2]
        return [colors[i] for i in np.arange(number) % len(colors)]

DEFAULT_PLOTSTYLE1D = PlotStyle1d()
DEFAULT_PLOTSTYLE2D = PlotStyle2d()
DEFAULT_PLOTSTYLE = PlotStyle(DEFAULT_PLOTSTYLE1D, DEFAULT_PLOTSTYLE2D)


def parenthesized_unit(unit):
    return f' ({unit})' if unit else ''


def get_grid_dic(params, arrs_1d):
    if isinstance(arrs_1d, dict):
        arrs_1d = [arrs_1d[par] for par in params]
    return dict(zip(params, np.meshgrid(*arrs_1d, indexing='ij')))


# Functions related to pdfs:
# -----------------------------------------------------------

def get_levels(pdf, fractions):
    """
    Return the values of P for which the
    sum(pdf[pdf > P]) == f
    for f in fractions. fractions can be array or scalar.
    """
    sorted_pdf = np.sort(pdf.ravel())
    fraction_below = np.cumsum(sorted_pdf)
    fraction_below /= fraction_below[-1]
    fraction_above = 1 - fraction_below
    return np.interp(fractions, fraction_above[::-1], sorted_pdf[::-1])

def get_minimal_interval(arr, pdf, confidence_level):
    """
    Given a 1 dimensional pdf, find smallest interval
    enclosing the central given fraction of the population.
    Warning: Can give a larger interval if the pdf is multimodal.
    Parameters
    ----------
    arr: array with the values of the variable,
         should be uniformly spaced.
    pdf: probability density function evaluated on arr
    confidence_level: float between 0 and 1, fraction of the
                      pdf enclosed in the returned interval.
    Returns
    -------
    (a, b): bounds of the variable
    """
    pdf = pdf / pdf.sum()
    p0 = get_levels(pdf, confidence_level)

    if np.allclose(p0, pdf):  # Handle uniform pdfs
        r = (1 - confidence_level) / 2
        return np.interp([r, 1-r], [0, 1], [arr[0], arr[-1]])

    i_left, i_right = np.where(pdf > p0)[0][[0, -1]]
    if i_left == 0:
        a = arr[0]
    else:
        a = np.interp(p0, pdf[i_left-1 : i_left+1], arr[i_left-1 : i_left+1])

    if i_right == len(arr)-1:
        b = arr[-1]
    else:
        b = np.interp(p0, pdf[i_right : i_right+2], arr[i_right : i_right+2])

    return a, b


def get_median_and_central_interval(arr, pdf, confidence_level):
    """
    Given a 1 dimensional pdf, find median and interval
    enclosing the central given fraction of the population.
    Parameters
    ----------
    arr: array with the values of the variable,
         should be uniformly spaced.
    pdf: probability density function evaluated on arr
    confidence_level: float between 0 and 1, fraction of the
                      pdf enclosed in the returned interval.
    Returns
    -------
    (median, a, b): median and bounds of the variable
    """
    edges = get_edges(arr)
    pdf = pdf / pdf.sum()
    cum_P = np.concatenate(([0], np.cumsum(pdf)))
    r = (1 - confidence_level) / 2
    return np.interp([.5, r, 1-r], cum_P, edges)


def latex_val_err(value, error):
    """
    Pass a value and its uncertainty, return a latex string
    '$value_{-err_minus}^{+err_plus}$' with the significant
    figures given by the uncertainties.
    Parameters
    ----------
    value: float
    error: [err_minus, err_plus]
    """
    error =  np.abs(error)
    if np.min(error) == 0:
        return f'${value:.2g}$'

    n_decimals = max(*np.ceil(-np.log10(error)).astype(int))
    if f'{np.min(error):e}'[0] == '1':
        n_decimals += 1

    def truncate(x):
        rounded = round(x, n_decimals)
        if n_decimals > 0:
            return rounded
        return int(rounded)

    truncated_value = truncate(value)
    err_plus = truncate(value + error[1] - truncated_value)
    err_minus = truncate(value - error[0] - truncated_value)
    n_decimals = max(0, n_decimals)
    return (fr'${truncated_value:.{n_decimals}f}'
            fr'_{{{err_minus:.{n_decimals}f}}}'
            fr'^{{{err_plus:+.{n_decimals}f}}}$')

def get_midpoints(x):
    """Get bin midpoints from edges"""
    return (x[1:] + x[:-1]) / 2

def get_edges(x):
    """Get bin edges from midpoints."""
    half_dx = (x[1] - x[0]) / 2
    return np.concatenate(([x[0] - half_dx], x + half_dx))


# Grid classes:
# -----------------------------------------------------------------

class Grid1D(dict):
    DEFAULT_INTERVAL_TYPE = 'central'  # 'minimal' or 'central'
    DEFAULT_CONFIDENCE_LEVEL = .9
    def __init__(self, dic, param, arr, pdfs, labels, units, density=True):
        """
        Parameters
        ----------
        """
        super().__init__(dic)
        self.param = param
        self.arr = arr
        self.pdfs = pdfs
        self.labels = labels
        self.confidence_level = self.DEFAULT_CONFIDENCE_LEVEL
        self.interval_type = self.DEFAULT_INTERVAL_TYPE
        self.estimates = {pdf: self.get_estimate(pdf) for pdf in pdfs}
        self.units = units
        self.dx = arr[1] - arr[0]
        if density:
            for pdf in self.pdfs:
                self[pdf] /= self[pdf].sum() * self.dx

    def get_estimate(self, pdf):
        median, a, b = get_median_and_central_interval(
            self.arr, self[pdf], self.confidence_level)
        if self.interval_type == 'central':
            return median, a, b
        a2, b2 = get_minimal_interval(self.arr, self[pdf],
                                      self.confidence_level)
        return median, a2, b2

    def plot_pdf(self, pdf, ax, set_label=False, set_title=False,
                 style=DEFAULT_PLOTSTYLE1D, title_label=True):
        if style.step:
            ax.step(self.arr, self[pdf], label=self.labels[pdf],
                    lw=style.linewidth, **style.kwargs, where='mid')
        else:
            ax.plot(self.arr, self[pdf], label=self.labels[pdf],
                    lw=style.linewidth, **style.kwargs)
        for val in self.estimates[pdf]:
            ax.plot([val]*2, [0, np.interp(val, self.arr, self[pdf])],
                    alpha=style.alpha_vlines, lw=style.lw_vlines,
                    **style.kwargs)
        span = np.linspace(*self.estimates[pdf][1:])
        ax.fill_between(
            span, 0, np.interp(span, self.arr, self[pdf]),
            alpha=style.alpha_fill, color=style.kwargs['color'])
        if self.labels[self.param] is not None and set_label:
            ax.set_xlabel(self.labels[self.param]
                          + parenthesized_unit(self.units[self.param]))

        if set_title:
            median, low, high = self.estimates[pdf]
            err = np.array([median - low, high - median])
            title = ''
            if title_label:
                title += f'{self.labels[self.param]}${{}}={{}}$'
            title += f'{latex_val_err(median, err)}{self.units[self.param]}'
            ax.set_title(title)
        ax.set_xlim(self.arr[[0, -1]])
        ax.set_ylim(0)

class Grid2D(dict):
    def __init__(self, dic, params, pdfs, labels, units):
        """
        Two-dimensional regular grid with marginalized probability
        density functions, useful for plotting e.g. corner
        plots.

        Parameters
        ----------
        dic: dictionary of 2D arrays defined over the 2D grid.
             Two should have the grid variables (a meshgrid).
             Typically the other entries will be pdfs.
        params: list of len=2, names of the grid variables.
        pdfs: list of names of the probability density functions.
        """
        super().__init__(dic)
        self.params = params
        self.pdfs = pdfs
        self.labels = labels
        self.units = units


    def plot_pdf(self, pdf, ax, set_labels=False,
                 style=DEFAULT_PLOTSTYLE2D):
        levels = list(get_levels(self[pdf], style.fractions))
        contour = ax.contour(*[self[par] for par in self.params], self[pdf],
                             levels=levels, colors=[style.color],
                             linestyles=style.linestyles,
                             linewidths=style.linewidths,
                             **style.contour_kwargs)
        if style.show_cl:
            clabels = {lev: fr'{100*f:.0f}%' for lev, f in zip(levels, style.fractions)}
            plt.clabel(contour, fontsize=style.clabel_fs, fmt=clabels,
                       colors=style.clabel_colors)
        legend_handle = contour.legend_elements()[0][0]
        if hasattr(ax, 'legend_handles'):
            ax.legend_handles.append(legend_handle)
        else:
            ax.legend_handles = [legend_handle]
        rgba = np.array(mpl.colors.to_rgba(style.color))
        if style.fill == 'gradient':
            img = rgba * np.ones_like(self[pdf].T[:, :, np.newaxis])
            img[:, :, 3] *= self[pdf].T / self[pdf].max()
            ax.imshow(img, origin='lower', aspect='auto',
                      extent=[m(self[par]) for par in self.params
                              for m in (np.min, np.max)])
        elif style.fill == 'flat':
            if isinstance(style.alphas, (int, float)):
                style.alphas = [style.alphas] * len(levels)
            levels = levels + [np.inf]
            for i in range(len(levels[:-1])):
                level = levels[i]
                next_level = levels[i+1]
                ax.contourf(*[self[par] for par in self.params], self[pdf],
                            levels=[level, next_level], colors=[style.color],
                            linestyles=style.linestyles, alpha=style.alphas[i])
        if self.labels[self.params[0]] is not None and set_labels:
            ax.set_xlabel(self.labels[self.params[0]]
                          + parenthesized_unit(self.units[self.params[0]]))
        if self.labels[self.params[1]] is not None and set_labels:
            ax.set_ylabel(self.labels[self.params[1]]
                          + parenthesized_unit(self.units[self.params[1]]))

class Grid(dict):
    """
    A regular, rectangular grid in some set of coordinates, which
    can additionally have multiple fields defined over
    those coordinates.
    """
    def __init__(self, dic, shape, params,
                 pdfs=None, labels=None, units=None,
                 plotstyle=DEFAULT_PLOTSTYLE,
                 compute_grids_1d_2d=True, density=True):
        """
        Parameters
        ----------
        dic: dictionary of arrays of shape <shape>, they are the grid
             parameters, likelihood and priors.
        shape: shape of the grid.
        params: list of keys in dic that correspond to grid parameters.
        priors: list of keys that correspond to grid priors.
        labels: dictionary where the keys are (any subset of) the
                keys in dic and the values are latex strings,
                used for plotting.
        units: dictionary where the keys are (any subset of) the
               params and the values are units in latex.
        """
        super().__init__(dic)
        self.params = params
        self.shape = shape
        self.pdfs = pdfs or []
        self.plotstyle = plotstyle
        self.labels = labels or {}
        self.update_labels()
        self.units = units or {}
        self.update_units()
        self.shape_dic = dict(zip(self.params, self.shape))
        self.density = density

        self.slices = {par: [0 for _ in self.params] for par in self.params}
        for i, par in enumerate(self.params):
            self.slices[par][i] = slice(None)
            self.slices[par] = tuple(self.slices[par])

        if compute_grids_1d_2d:
            self.compute_grids_1d()
            self.compute_grids_2d()

    @classmethod
    def from_samples(cls, params, samples_dic, bins=40, pdf_key='posterior',
                     plotstyle=DEFAULT_PLOTSTYLE, labels=None, units=None,
                     hollow='auto', weights=None, range=None, density=True):
        n_samples = len(samples_dic[params[0]])
        if hollow == 'auto':
            hollow = len(params) > 4
        if hollow:
            # Only compute the 2d grids to save memory
            params_2d = [(x_par, y_par)
                         for i, x_par in enumerate(params)
                         for y_par in params[i+1:]]
            grids_1d = {}
            grids_2d = {}
            for xy_pars in params_2d:
                aux_grid = Grid.from_samples(
                    params=xy_pars, samples_dic=samples_dic, bins=bins,
                    pdf_key=pdf_key, plotstyle=plotstyle, labels=labels,
                    units=units, hollow=False, weights=weights,
                    density=density)
                grids_2d[xy_pars] = aux_grid.grids_2d[xy_pars]
                for par in xy_pars:
                    grids_1d[par] = aux_grid.grids_1d[par]
            arrs_1d = [grids_1d[par].arr for par in params]
            shape = [len(arr) for arr in arrs_1d]
            dic = {}  # Full grid will be empty to save memory
            instance = cls(dic, shape, params, pdfs=[pdf_key],
                           labels=labels, units=units,
                           compute_grids_1d_2d=False, density=density)
            instance.grids_1d = grids_1d
            instance.grids_2d = grids_2d
            instance.n_samples = n_samples
            return instance

        # Else, full-dimensional grid
        h = np.histogramdd([samples_dic[p] for p in params], bins=bins,
                           density=True, weights=weights, range=range)
        arrs_1d = [get_midpoints(edges) for edges in h[1]]
        dic = get_grid_dic(params, arrs_1d)
        dic[pdf_key] = h[0] * n_samples
        shape = [len(arr) for arr in arrs_1d]
        instance = cls(dic, shape, params, pdfs=[pdf_key],
                       labels=labels, units=units, density=density)
        instance.n_samples = n_samples
        return instance

    def to_json(self, dname, fname='metadata', permissions='644'):
        # Make directory
        if not os.path.isdir(dname):
            os.mkdir(dname)
            os.system(f'chmod 755 {dname}')
        # Save large data to npy files
        fnames = {}
        for k, v in self.items():
            fnames[k] = k + '.npy'  # Relative path, so directory can be moved
            abs_fname = os.path.join(dname, fnames[k])
            np.save(abs_fname, v)
            os.system(f'chmod {permissions} {abs_fname}')
        # Save small data to json dictionary
        metadata = {'fnames': fnames}
        for k in ['params', 'shape', 'units', 'labels', 'pdfs', 'density']:
            metadata[k] = getattr(self, k)
        fname_metadata = os.path.join(dname, fname)
        with open(fname_metadata, 'w') as f:
            json.dump(metadata, f, indent=2)
            f.write("\n")
        os.system(f'chmod {permissions} {fname_metadata}')

        self.fname_metadata = fname_metadata

    @classmethod
    def from_json(cls, fname_metadata):
        """
        Instantiate class from json file created by previous run
        :param fname_metadata: json file with grid information
        :return: Instance of Grid
        """
        dname = os.path.dirname(fname_metadata)
        with open(fname_metadata, 'r') as f:
            metadata = json.load(f)

        dic = {}
        for k, fname in metadata['fnames'].items():
            if not os.path.isabs(fname):
                fname = os.path.join(dname, fname)
            dic[k] = np.load(fname)

        del metadata['fnames']  # The rest are kwargs to Grid.__init__()
        instance = cls(dic, **metadata)
        return instance

    def add_pdf(self, key, val, label=None, normalize=True):
        """
        Keeps the list of pdfs up to date.

        key: name of the pdf
        val: array of grid.shape with the pdf values
        """
        if normalize:
            val /= val.sum()
        self[key] = val
        self.pdfs.append(key)
        if label is not None:
            self.labels[key] = label
        self.update_labels()
        self.compute_grids_1d()
        self.compute_grids_2d()

    def update_units(self):
        units = copy(self.units)
        self.units.update({p: '' for p in self.params})
        self.units.update(units)  # Don't let old units be overriden

    def update_labels(self):
        labels = copy(self.labels)
        self.labels.update({p: p for p in self.params})
        self.labels.update({k: k for k in self})
        self.labels.update(labels)  # Don't let old labels be overriden

    def compute_grids_1d(self):
        self.grids_1d = {}
        for par in self.params:
            arr_1d = self.get_arr_1d(par)
            pdfs_1d = {pdf: self.marginalize(pdf, kept_params=[par])
                       for pdf in self.pdfs}
            self.grids_1d[par] = Grid1D(
                pdfs_1d, par, arr_1d, list(pdfs_1d),
                labels=self.labels, units=self.units, density=self.density)

    def compute_grids_2d(self):
        self.params_2d = [(x_par, y_par)
                          for i, x_par in enumerate(self.params)
                          for y_par in self.params[i+1:]]
        self.grids_2d = {}
        for xy_pars in self.params_2d:
            dic = get_grid_dic(xy_pars,
                               [self.get_arr_1d(par) for par in xy_pars])
            dic.update({pdf: self.marginalize(pdf, kept_params=xy_pars)
                        for pdf in self.pdfs})
            self.grids_2d[xy_pars] = Grid2D(
                dic, xy_pars, self.pdfs, self.labels, self.units)

    def marginalize(self, pdf, kept_params):
        js = tuple(j for j, y_par in enumerate(self.params)
                   if not y_par in kept_params)  # Marginalize over these axes
        dy = np.prod([np.diff(self[y_par][self.slices[y_par]])[0]
                      for y_par in self.params if not y_par in kept_params])
        return self[pdf].sum(axis=js) * dy

    def marginal_maximize(self, pdf, kept_params):
        """
        Some non-standard but prior-agnostic way to 'marginalize' the
        likelihood: maximize w.r.t. the parameters that you are
        not interested in.
        """
        js = tuple(j for j, y_par in enumerate(self.params)
                   if not y_par in kept_params)  # Maximize over these axes
        return self[pdf].max(axis=js)

    def get_arr_1d(self, par):
        """
        Return an array with the values that a given variable
        can take on the grid.
        """
        return self[par][self.slices[par]]

    def change_resolution(self, zoom=2):
        self.update({k: ndimage.zoom(self[k], zoom, order=1) for k in self})
        self.shape = np.array(zoom) * self.shape
        self.shape_dic = dict(zip(self.params, self.shape))
        self.compute_grids_1d()
        self.compute_grids_2d()

    def gaussian_filter(self, sigma=1):
        self.update({k: ndimage.gaussian_filter(self[k], sigma)
                     for k in self.pdfs})
        self.compute_grids_1d()
        self.compute_grids_2d()

    def interp_other(self, other_grid, key):
        """
        Interpolates on self a scalar function defined on another
        grid that has the same params.
        """
        return ndimage.map_coordinates(
            other_grid[key],
            [(other_grid.shape_dic[par] - 1)
             * (self[par] - other_grid[par].min())
             / (other_grid[par].max() - other_grid[par].min())
             for par in self.params], order=1)

    def interp_self(self, key, coords):
        """
        Interpolates a scalar function (already defined
        on the grid) on arbitrary values provided.
        """
        if len(np.shape(coords)) == 1:
            return self.interp_self(key, np.array(coords)[:, np.newaxis])[0]
        return ndimage.map_coordinates(
            self[key], [(self.shape_dic[par] - 1) * (c - self[par].min())
                        / (self[par].max() - self[par].min())
                        for c, par in zip(coords, self.params)], order=1)

    def change_variables(self, new_params, new_shape, new_extent, new2old_func,
                         new_labels=None, new_units=None):
        dic = get_grid_dic(new_params, [np.linspace(*ext, num) for ext, num
                                        in zip(new_extent, new_shape)])
        coords = new2old_func(*[dic[par] for par in new_params])
        for k in self:
            # params is useless and pdfs do not transform like scalars:
            if not k in self.params + self.pdfs:
                dic[k] = self.interp_self(k, coords)
        if new_labels is None:
            new_labels = self.labels
        if new_units is None:
            new_units = self.units

        return Grid(dic, np.array(new_shape), new_params, [],
                    new_labels, new_units, self.plotstyle)

    def corner_plot(
            self, pdf=None, title=None, subplot_size=2., fig=None, ax=None,
            figsize=None, nbins=6, set_legend=False, save_as=None, y_title=.98,
            plotstyle=None, show_titles_1d=True, scatter_points=None,
            title_label=True, legend_title=None, plot_params=None, **kwargs):
        if pdf is None:
            if len(self.pdfs) == 1:
                pdf = self.pdfs[0]
            else:
                raise ValueError(f'Specify a pdf key from {self.pdfs}.')

        if plotstyle is None:
            plotstyle = self.plotstyle

        plot_params = plot_params or self.params

        n_rows = n_cols = len(plot_params)
        if fig is None or ax is None:
            if figsize is None:
                figsize = (subplot_size * n_cols, subplot_size * n_rows + .3)
            fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize)
            if n_rows == 1:
                ax = np.array([[ax]])
        if title is not None:
            plt.suptitle(title, y=y_title)

        # Plot 2D pdfs (off-diagonal in the plot grid)
        for row, y_par in list(enumerate(plot_params))[1:]:
            for col, x_par in list(enumerate(plot_params))[:row]:
                self.grids_2d[x_par, y_par].plot_pdf(pdf, ax[row][col],
                                                     style=plotstyle.plotstyle_2d)
        # Plot 1D pdf (diagonal)
        for i, par in enumerate(plot_params):
            self.grids_1d[par].plot_pdf(pdf, ax[i][i], set_title=show_titles_1d,
                                        style=plotstyle.plotstyle_1d,
                                        title_label=title_label)
            ax[i][i].autoscale()
            ax[i][i].set_ylim(0)

        # Embellishment:
        if set_legend:
            handles_1d, labels_1d = ax[0][0].get_legend_handles_labels()
            fig.legends = []
            fig.legend(handles_1d, labels_1d, loc='upper right',
                       bbox_to_anchor=(1, .95), frameon=False, title=legend_title)
        fig, ax = self.embellish_plot(fig, ax, nbins=nbins, pdf=pdf,
                                      plot_params=plot_params, **kwargs)

        if scatter_points is not None:
            colors = PlotStyle._gen_colors(len(scatter_points))
            for index, (_, row) in enumerate(scatter_points.iterrows()):
                for i, xpar in enumerate(plot_params):
                    ax[i][i].axvline(row[xpar], color=colors[index])
                    for j, ypar in enumerate(plot_params):
                        if j > i:
                            ax[j][i].scatter(row[xpar], row[ypar],
                                             color=colors[index])

        if save_as is not None:
            plt.savefig(save_as, bbox_inches='tight')

        return fig, ax

    def embellish_plot(self, fig, ax, lim={}, tight=False, tightness=.95, nbins=6,
                       pdf=None, plot_params=None, **kwargs):
        """
        Handle the limits, labels, etc of the axes.
        fig, ax: figure and axes returned by self.corner_plot()
        lim: optional dict of (par: [xmin, xmax]) pairs
        """
        n_rows, n_cols = ax.shape
        if pdf is None:
            if tight and len(self.pdfs) != 1:
                raise ValueError('Please specify a pdf key.')
            pdf = self.pdfs[0]
        plot_params = plot_params or self.params

        override_lim = deepcopy(lim)
        if tight:
            lim = {par: get_median_and_central_interval(
                self.grids_1d[par].arr, self.grids_1d[par][pdf].copy(), tightness)[1:]
                   for par in self.params}
        else:
            lim = {par: [self.grids_1d[par].arr.min(), self.grids_1d[par].arr.max()]
                   for par in self.params}
        lim.update(override_lim)

        # Force xlim
        for col, x_par in enumerate(plot_params):
            for row in range(n_rows):
                ax[row][col].xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins))
                ax[row][col].set_xlim(lim[x_par])
        # Force ylim
        for row, y_par in list(enumerate(plot_params))[1:]:
            for col in range(row - 1):
                ax[row][col].yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins))
                ax[row][col].set_ylim(lim[y_par])
        # Set x labels & ticks
        for col, x_par in enumerate(plot_params):
            ax[n_rows-1][col].set_xlabel(
                self.labels[x_par] + parenthesized_unit(self.units[x_par]),
                size='large')
            plt.setp(ax[n_rows-1][col].get_xticklabels(), rotation=45)
            ax[n_rows-1][col].locator_params(nbins=nbins)
            for row in range(n_rows-1):
                ax[row][col].tick_params(labelbottom=False)
        # Set y labels & ticks
        for row, y_par in list(enumerate(plot_params))[1:]:
            ax[row][0].set_ylabel(
                self.labels[y_par] + parenthesized_unit(self.units[y_par]),
                size='large')
            plt.setp(ax[row][0].get_yticklabels(), rotation=45)
            ax[row][0].locator_params(nbins=nbins)
            for col in range(1, n_cols):
                ax[row][col].tick_params(labelleft=False)
        # Disable upper triangle
        for row in range(n_rows-1):
            for col in range(row+1, n_cols):
                ax[row][col].axis('off')
        # Share x axes & ticks
        for col in range(n_cols):
            for row in range(0, n_rows-1):
                ax[-1][col].get_shared_x_axes().join(ax[-1][col],
                                                     ax[row][col])
            for row in range(0, n_rows):
                ax[row][col].set_xticks(ax[n_rows-1][col].get_xticks())
        # Share y axes & ticks
        for row in range(1, n_rows):
            for col in range(1, row):
                ax[row][0].get_shared_y_axes().join(ax[row][0], ax[row][col])
                ax[row][col].set_yticks(ax[row][0].get_yticks())
        for i in range(n_rows):
            ax[i][i].tick_params(axis='y', left=False, labelleft=False)
        # Set top and right ticks
        for row in range(1, n_rows):
            for col in range(row):
                ax[row][col].tick_params(which='both', direction='in',
                                         right=True, top=True)
        for i in range(n_rows):
            ax[i][i].tick_params(axis='y', left=False, labelleft=False)
            ax[i][i].tick_params(which='both', direction='in', top=True)
        # Force xlim
        for col, x_par in enumerate(plot_params):
            for row in range(n_rows):
                ax[row][col].set_xlim(lim[x_par])
        # Force ylim
        for row, y_par in list(enumerate(plot_params))[1:]:
            for col in range(row):
                ax[row][col].set_ylim(lim[y_par])
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.subplots_adjust(hspace=0.04, wspace=0.04)
        return fig, ax

    def plot_line(self, ax, xpar, ypar, xmin=None, xmax=None,
                  plotstr='--k', yfunc=lambda x: x, plot_params=None):
        """
        Plot a function y(x) in the corresponding subplot of a corner plot.

        Parameters
        ----------
            ax: Corner plot axes as outputted by `self.corner_plot()`.
            xpar: Parameter name from `self.params`, independent variable of `yfunc`
                  (need not actually be the horizontal axis in the plot).
            ypar: Parameter name of dependent variable in `yfunc`.
            yfunc: Function to plot.

        """
        plot_params = plot_params or self.params
        if xmin is None:
            xmin = self.get_arr_1d(xpar)[0]
        if xmax is None:
            xmax = self.get_arr_1d(xpar)[-1]

        x = np.linspace(xmin, xmax, 200)
        y = yfunc(x)

        i = plot_params.index(xpar)
        j = plot_params.index(ypar)
        if i < j:
            i, j = j, i
            x, y = y, x
        xlim = ax[i][j].get_xlim()
        ylim = ax[i][j].get_ylim()
        ax[i][j].plot(x, y, plotstr)
        ax[i][j].set_xlim(xlim)
        ax[i][j].set_ylim(ylim)

    def shade_region(self, ax, xpar, ypar, xmin=None, xmax=None,
                     ymin_func=lambda x: x, ymax_func=None,
                     hatch='/////', hatch_lw=.2, facecolor='lightgray',
                     plot_params=None):
        """
        Shade a region between functions ymin(x) and ymax(x) in the
        corresponding subplot of a corner plot. E.g. to highlight a
        forbidden region of parameter space.

        Parameters
        ----------
            ax: Corner plot axes as outputted by `self.corner_plot()`.
            xpar: Parameter name from `self.params`, independent variable of
                  `ymin_func` and `ymax_func` (need not actually be the
                  horizontal axis in the plot).
            ypar: Parameter name of dependent variable in `ymin_func` and
                  `ymax_func`.
            ymin_func, ymax_func: Functions of x specifying the boundaries of
                                  the region to shade. They default to the
                                  min and max values of y.
        """
        mpl.rcParams['hatch.linewidth'] = hatch_lw
        plot_params = plot_params or self.params
        if xmin is None:
            xmin = self.get_arr_1d(xpar)[0]
        if xmax is None:
            xmax = self.get_arr_1d(xpar)[-1]
        if ymin_func is None:
            ymin_func = lambda x: self.get_arr_1d(ypar)[0]
        if ymax_func is None:
            ymax_func = lambda x: self.get_arr_1d(ypar)[-1]
        x = np.linspace(xmin, xmax, 200)
        ymin = ymin_func(x)
        ymax = ymax_func(x)

        i = plot_params.index(xpar)
        j = plot_params.index(ypar)
        if i < j:
            a = ax[j][i]
            fill_between = a.fill_between

        else:
            a = ax[i][j]
            fill_between = a.fill_betweenx
        xlim = a.get_xlim()
        ylim = a.get_ylim()
        fill_between(x, ymin, ymax, hatch=hatch, facecolor=facecolor)
        a.set_xlim(xlim)
        a.set_ylim(ylim)


class MultiGrid:
    def __init__(self, grids, plotstyles=None):
        self.grids = grids
        self.ngrids = len(self.grids)
        self.params = grids[0].params
        assert all(g.params == self.params for g in grids[1:])

        self.plotstyles = plotstyles or PlotStyle.get_many(len(grids))

        self.plot_line = self.grids[0].plot_line
        self.shade_region = self.grids[0].shade_region

    def corner_plot(self, pdfs=None, grid_inds=None, tight=False, tightness=.99,
                    override_lim=None, fig=None, ax=None, show_titles_1d=False,
                    plot_params=None, **kwargs):
        if grid_inds is None:
            grid_inds = range(self.ngrids)
        if not hasattr(show_titles_1d, '__len__'):
            show_titles_1d = [show_titles_1d] * len(grid_inds)
        if pdfs is None:
            if any(len(g.pdfs) != 1 for g in self.grids):
                raise ValueError('Please specify pdfs')
            pdfs = [g.pdfs[0] for g in self.grids]
        elif isinstance(pdfs, str):
            pdfs = [pdfs] * len(grid_inds)
        plot_params = plot_params or self.params
        for i in grid_inds:
            fig, ax = self.grids[i].corner_plot(
                pdf=pdfs[i], fig=fig, ax=ax,
                plotstyle=self.plotstyles[i],
                show_titles_1d=show_titles_1d[i],
                plot_params=plot_params, **kwargs)
        if tight:
            lims = {par: np.array([get_median_and_central_interval(
                self.grids[i].grids_1d[par].arr,
                self.grids[i].grids_1d[par][pdfs[i]].copy(), tightness)[1:]
                                   for i in grid_inds])
                    for par in self.params}
            lim = {k: [np.min(v), np.max(v)] for k, v in lims.items()}
        else:
            lim = {par: [np.min([self.grids[i].grids_1d[par].arr.min()
                                 for i in grid_inds]),
                         np.max([self.grids[i].grids_1d[par].arr.max()
                                 for i in grid_inds])]
                   for par in self.params}
        if override_lim:
            lim.update(override_lim)
        self.grids[0].embellish_plot(fig, ax, lim=lim, plot_params=plot_params,
                                     **kwargs)
        return fig, ax

    def plot_2d(self, xpar, ypar, pdfs=None, grid_inds=None,
                xlim=None, ylim=None, set_labels=True, figsize=None,
                set_legend=False, fig=None, ax=None):
        """
        Just 'one panel' of the corner plot.
        """
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        if grid_inds is None:
            grid_inds = range(self.ngrids)
        if pdfs is None:
            if any(len(grid.pdfs) != 1 for grid in self.grids):
                raise ValueError('Specify pdfs')
            pdfs = [grid.pdfs[0] for grid in self.grids]
        elif isinstance(pdfs, str):
            pdfs = [pdfs] * len(grid_inds)
        for i in grid_inds:
            self.grids[i].grids_2d[xpar, ypar].plot_pdf(
                pdf=pdfs[i], ax=ax, style=self.plotstyles[i].plotstyle_2d,
                set_labels=set_labels)
        if set_legend:
            ax.legend(ax.legend_handles, pdfs)
        ax.tick_params(which='both', direction='in', top=True, right=True)
        # Default xlim, ylim
        xlim_, ylim_ = [[m([m(self.grids[i].grids_1d[par].arr)
                            for i in grid_inds])
                         for m in [np.min, np.max]] for par in [xpar, ypar]]
        xlim = xlim or xlim_
        ylim = ylim or ylim_
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        return fig, ax
