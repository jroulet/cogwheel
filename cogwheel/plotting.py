"""Basic corner plots."""

import itertools
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Union
import scipy.interpolate
import scipy.ndimage
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib as mpl
import pandas as pd

from cogwheel import utils


class LatexLabels(dict):
    """
    Helper class for plotting parameter labels.

    Example
    -------
    >>> labels = LatexLabels(labels={'m': '$m$', 'theta': r'$\theta$'},
    ...                      units={'m': 'kg'})
    >>> labels['m']
    '$m$'
    >>> labels.units['m']
    'kg'
    >>> labels.units['theta']
    ''
    >>> labels.with_units('m')
    '$m$ (kg)'
    >>> labels.with_units('theta')
    '$\\theta$'
    >>> labels.with_units('x')
    'x'
    """
    def __init__(self, labels: dict = None, units: dict = None):
        super().__init__(labels or {})

        self.units = defaultdict(str, units or {})

    def with_units(self, par):
        """Return string of the form '{label} ({unit})'."""
        if self.units[par]:
            return self[par] + f' ({self.units[par]})'
        return self[par]

    def __missing__(self, par):
        if isinstance(par, str) and par.startswith('folded_'):
            label = self[par.removeprefix('folded_')]
            return rf'{label}$^{{\rm folded}}$'.replace('$$', '')
        return par


# ----------------------------------------------------------------------
# Plot settings

def gen_linestyles(number):
    """
    Return list of linestyles.

    Parameters
    ----------
    number: int
        How many linestyles are desired.
    """
    if number <= len(few_linestyles := ['-', '--', '-.', ':']):
        return few_linestyles[:number]
    return ['-'] * number


def gen_colors(number):
    """
    Return list of colors.

    Parameters
    ----------
    number: int
        How many colors are desired.
    """
    colors = (mpl.cm.get_cmap('tab20').colors
              + mpl.cm.get_cmap('tab20b').colors)
    colors = colors[::2] + colors[1::2]
    return [colors[i] for i in np.arange(number) % len(colors)]


@dataclass
class PlotStyle:
    """
    Class that encapsulates plotting choices (colors, linestyles, etc.)
    for a corner plot of a distribution.

    Attributes
    ----------
    confidence_level: float between 0 and 1, or ``None``
        Determines the reported confidence interval around the
        median (highlighted band in 1-d marginal probability and
        numeric values in the subplot titles). If ``None``, both
        the numerical values and highlighted bands are removed.

    contour_fractions: sequence of floats
        Fractions of the distribution to enclose by 2-d contours.

    bins: int | {'rice', 'sturges', 'sqrt'}
        How many histogram bins to use, the same for all parameters.

    color_2d: str, RGB tuple, etc.
        Color used for the 2-d marginal distributions.

    contour_kwargs: dict
        Keyword arguments to `plt.contour` and `plt.contourf`

    vline_kwargs: dict
        Keyword arguments to `plt.plot` for the vertical lines
        signaling medians and 1-d confidence intervals.

    vfill_kwargs: dict
        Keyword arguments to `plt.fill_between` for 1-d plots.

    kwargs_1d: dict
        Keyword arguments to `plt.plot` for 1-d plots.

    clabel_kwargs: dict, optional
        Keyword arguments for contour labels. Pass an empty `dict`
        to use defaults. ``None`` draws no contour labels.

    fill: {'gradient', 'flat', 'none'}
        How to display 2-d marginal distributions:
        'gradient' displays the 2-d pdf with a transparency gradient
        'flat' fills the contours with a flat transparent color
        'none' shows just the contours

    smooth: float
        Smooth the 2d histograms by convolving them with a Gaussian
        kernel with this standard deviation in pixel units.
        0 (default) does no smoothing.

    density: bool
        Whether to normalize the 1-d histograms to integrate to 1.

    tail_probability: float between 0 and 1
        Disregard `tail_probability / 2` of the distribution to
        either side in the plots. Used as an automatic way of
        zooming in on the interesting part of the distribution if
        there are a few outlier samples.
        0 (default) includes all samples.

    See Also
    --------
    CornerPlot, MultiCornerPlot
    """
    # Defaults:
    KWARGS_1D = {'color': 'C0'}
    VLINE_KWARGS = {'alpha': .5, 'linewidth': 1}
    VFILL_KWARGS = {'alpha': .1}

    confidence_level: float = 0.9
    contour_fractions: tuple[float] = (0.5, 0.9)
    bins: Union[int, str] = 'rice'
    color_2d: str = 'k'
    contour_kwargs: dict = field(default_factory=dict)
    vline_kwargs: dict = field(default_factory=dict)
    vfill_kwargs: dict = field(default_factory=dict)
    kwargs_1d: dict = field(default_factory=dict)
    clabel_kwargs: dict = None
    fill: str = 'gradient'
    smooth: float = 0.0
    density: bool = True
    tail_probability: float = 0.0

    @property
    def decreasing_contour_fractions(self):
        """In decreasing order so that levels are increasing."""
        return np.sort(np.atleast_1d(self.contour_fractions))[::-1]

    def __post_init__(self):
        self.vline_kwargs = self.VLINE_KWARGS | self.vline_kwargs
        self.kwargs_1d = self.KWARGS_1D | self.kwargs_1d
        self.vfill_kwargs = self.VFILL_KWARGS | self.vfill_kwargs

    def get_contour_kwargs(self):
        """Keyword arguments to `plt.contour` and `plt.contourf`."""
        return {'colors': [self.color_2d]} | self.contour_kwargs

    def get_vline_kwargs(self):
        """
        Keyword arguments to `plt.plot` for the vertical lines signaling
        medians and 1-d confidence intervals.
        """
        return self.kwargs_1d | self.vline_kwargs

    def get_vfill_kwargs(self):
        """Keyword arguments to `plt.fill_between` for 1-d plots."""
        return {'color': self.kwargs_1d['color']} | self.vfill_kwargs

    @classmethod
    def get_many(cls, number, **kwargs):
        """
        Return list of plostyles with different colors and
        linestyles.
        """
        linestyles = gen_linestyles(number)
        colors = gen_colors(number)

        # This logic lets user `kwargs` override automatic choices:
        contour_kwargs = kwargs.pop('contour_kwargs', {})
        kwargs_1d = kwargs.pop('kwargs_1d', {})
        kwargs = {'fill': 'flat', 'confidence_level': None} | kwargs
        for color, linestyle in zip(colors, linestyles):
            yield cls(
                contour_kwargs={'linestyles': [linestyle]} | contour_kwargs,
                kwargs_1d={'color': color, 'linestyle': linestyle} | kwargs_1d,
                **{'color_2d': color} | kwargs)


def get_transparency_colormap(color):
    """Return a colormap of a given color with varying transparency."""
    rgb = mpl.colors.to_rgb(color)
    rgba = [rgb + (alpha,) for alpha in np.linspace(0, 1)]
    return ListedColormap(rgba)


# ----------------------------------------------------------------------
# Functions related to pdfs:

def latex_val_err(value, error):
    """
    Pass a value and its uncertainty, return a latex string
    '$value_{-err_minus}^{+err_plus}$' with the significant figures
    given by the uncertainties.

    Parameters
    ----------
    value: float
    error: (err_minus, err_plus)
    """
    error = np.abs(error)
    min_error = np.min(error)

    if min_error == 0:
        return f'${value:.2g}$'

    n_decimals = np.ceil(-np.log10(min_error)).astype(int)
    if f'{min_error:e}'.startswith('1'):
        n_decimals += 1

    def truncate(val):
        rounded = round(val, n_decimals)
        if n_decimals > 0:
            return rounded
        return int(rounded)

    truncated_value = truncate(value)
    err_plus = truncate(value + error[1] - truncated_value)
    err_minus = truncate(value - error[0] - truncated_value)
    n_decimals = max(0, n_decimals)
    return (fr'${truncated_value+0:.{n_decimals}f}'  # +0 so -0.0 -> 0.0
            fr'_{{{err_minus:.{n_decimals}f}}}'
            fr'^{{{err_plus:+.{n_decimals}f}}}$')


def get_midpoints(arr):
    """Get bin midpoints from edges."""
    return (arr[1:] + arr[:-1]) / 2


def get_contour_fractions(sigmas):
    """
    Enclosed probability within a number of sigmas for a 2d Gaussian.
    """
    return 1 - np.exp(-np.square(sigmas) / 2)


# ----------------------------------------------------------------------
# Plotting class

class CornerPlot:
    """
    Class for making a corner plot of a multivariate distribution if you
    have samples from the distribution.

    Methods
    -------
    plot: Make a corner plot.
    scatter_points: Plot points on a corner plot (e.g. "truths").
    set_lims: Edit the limits of a corner plot.
    plot_2d: Make one panel of a corner plot.
    """
    DEFAULT_LATEX_LABELS = LatexLabels()
    MARGIN_INCHES = .8

    def __init__(self, samples: pd.DataFrame, params=None,
                 plotstyle=None, weights_col='weights',
                 latex_labels=None, **plotstyle_kwargs):
        """
        Parameters
        ----------
        samples: pandas.DataFrame
            Columns correspond to parameters to plot and rows correspond
            to samples.

        params: list of str, optional
            Subset of columns present in all dataframes, to plot a
            reduced number of parameters.

        plotstyle: PlotStyle
            Determines the colors, linestyles, etc.

        weights_col: str, optional
            If existing, use a column with this name to set weights for
            the samples. Pass `None` to ignore, if you have a column
            named 'weights' that is not to be interpreted as weights.

        latex_labels: LatexLabels
            Maps column names to latex strings and assigns units.

        **plotstyle_kwargs
            Passed to ``PlotStyle`` constructor, ignored if `plotstyle`
            is passed.

        Other parameters
        ----------------
        confidence_level: float between 0 and 1, or ``None``
            Determines the reported confidence interval around the
            median (highlighted band in 1-d marginal probability and
            numeric values in the subplot titles). If ``None``, both
            the numerical values and highlighted bands are removed.

        contour_fractions: sequence of floats
            Fractions of the distribution to enclose by 2-d contours.

        bins: int | {'rice', 'sturges', 'sqrt'}
            How many histogram bins to use, the same for all parameters.

        color_2d: str, RGB tuple, etc.
            Color used for the 2-d marginal distributions.

        contour_kwargs: dict
            Keyword arguments to `plt.contour` and `plt.contourf`

        vline_kwargs: dict
            Keyword arguments to `plt.plot` for the vertical lines
            signaling medians and 1-d confidence intervals.

        vfill_kwargs: dict
            Keyword arguments to `plt.fill_between` for 1-d plots.

        kwargs_1d: dict
            Keyword arguments to `plt.plot` for 1-d plots.

        clabel_kwargs: dict, optional
            Keyword arguments for contour labels. Pass an empty `dict`
            to use defaults. ``None`` draws no contour labels.

        fill: {'gradient', 'flat', 'none'}
            How to display 2-d marginal distributions:
            'gradient' displays the 2-d pdf with a transparency gradient
            'flat' fills the contours with a flat transparent color
            'none' shows just the contours

        smooth: float
            Smooth the 2d histograms by convolving them with a Gaussian
            kernel with this standard deviation in pixel units.
            0 (default) does no smoothing.

        density: bool
            Whether to normalize the 1-d histograms to integrate to 1.

        tail_probability: float between 0 and 1
            Disregard `tail_probability / 2` of the distribution to
            either side in the plots. Used as an automatic way of
            zooming in on the interesting part of the distribution if
            there are a few outlier samples.
            0 (default) includes all samples.

        See Also
        --------
        MultiCornerPlot
        """
        self.samples = samples.dropna()
        self.latex_labels = latex_labels or self.DEFAULT_LATEX_LABELS
        self.plotstyle = plotstyle or PlotStyle(**plotstyle_kwargs)
        self.weights_col = weights_col
        self.params = params or [col for col in samples if col != weights_col]
        if missing := (set(self.params) - set(self.samples)):
            raise ValueError(f'`samples` missing key(s) {missing}')

        self.fig = None
        self.axes = None

    def plot(self, fig=None, title=None, max_figsize=10., max_n_ticks=4,
             label=None, legend_title=None):
        """
        Make a corner plot of the distribution.

        Parameters
        ----------
        fig: ``matplotlib.Figure``, optional
            If provided, will reuse this figure to make the plot. Must
            have axes matching the number of parameters.

        title: str, optional
            Figure title.

        max_figsize: float
            Maximum size in inches of a side of the square figure.
            Ignored if `fig` is passed.

        max_n_ticks: int
            Determines the number of ticks in each subplot. Ignored if
            `fig` is passed.

        label: str, optional
            Legend label.

        legend_title: str, optional
            Legend title.
        """
        self._setup_fig(fig, max_figsize, max_n_ticks=max_n_ticks)

        for par in self.params:
            self._plot_1d(par, label)

        for xpar, ypar in itertools.combinations(self.params, 2):
            self.plot_2d(xpar, ypar, ax=self.axes[self.params.index(ypar),
                                                  self.params.index(xpar)])

        y_title = 1 - .9 * self.MARGIN_INCHES / self.fig.get_figheight()
        self.fig.suptitle(title, y=y_title, verticalalignment='bottom')

        self.axes[0][-1].legend(
            *self.axes[0][0].get_legend_handles_labels(),
            bbox_to_anchor=(1, 1), frameon=False, loc='upper right',
            borderaxespad=0, borderpad=0, title=legend_title)

    def scatter_points(self, scatter_points, colors=None,
                       adjust_lims=False, **kwargs):
        """
        Add scatter points to an existing corner plot.
        For every point passed, one vertical line in the diagonal panels
        and one dot in each off-diagonal panel will be plotted.

        Parameters
        ----------
        scatter_points: pandas.DataFrame or dict
            Columns are parameter names, must contain ``self.params``.
            Rows correspond to parameter values. A dict is acceptable to
            plot a single point.

        colors: iterable, optional
            Colors corresponding to each scatter point passed.

        adjust_lims: bool
            Whether to adjust x and y limits in the case that the
            scatter points lie outside or very near the current limits.

        **kwargs:
            Passed to ``matplotlib.axes.Axes.scatter``.
        """
        if self.axes is None:
            raise RuntimeError('There is no plot to scatter points on. '
                               'Call ``plot`` before ``scatter_points``.')

        if isinstance(scatter_points, dict):
            scatter_points = pd.DataFrame(scatter_points, index=[0])

        colors = colors or gen_colors(len(scatter_points))

        lims = self.get_current_lims()

        for color, (_, point) in zip(colors, scatter_points.iterrows()):
            for ax, par in zip(np.diagonal(self.axes), self.params):
                try:
                    ax.axvline(point[par], color=color)
                except KeyError:  # `scatter_points` may lack params
                    pass

            for row, col in zip(*np.tril_indices_from(self.axes, -1)):
                try:
                    self.axes[row][col].scatter(point[self.params[col]],
                                                point[self.params[row]],
                                                color=color, **kwargs)
                except KeyError:  # `scatter_points` may lack params
                    pass

        if not adjust_lims:
            self.set_lims(**lims)

    def plot_2d(self, xpar, ypar, ax=None):
        """
        Plot just one panel of the corner plot.

        Parameters
        ----------
        xpar, ypar: str
            Parameters in ``self.params`` to plot in the x and y axes.

        ax: matplotlib.axes.Axes or None
            Axes on which to draw the figure. ``None`` makes new axes.

        label: str
            Add a legend element with this label.
        """
        if ax is None:
            _, ax = plt.subplots()
            ax.set_xlabel(self.latex_labels.with_units(xpar))
            ax.set_ylabel(self.latex_labels.with_units(ypar))

        pdf, extent = self._get_pdf_2d(xpar, ypar)
        levels = self._get_levels(pdf)
        contour = ax.contour(pdf, extent=extent, levels=levels,
                             **self.plotstyle.get_contour_kwargs())

        if self.plotstyle.clabel_kwargs is not None:
            clabels = [
                f'{fraction:.0%}'
                for fraction in self.plotstyle.decreasing_contour_fractions]
            plt.clabel(contour, fmt=dict(zip(levels, clabels)),
                       **self.plotstyle.clabel_kwargs)

        if self.plotstyle.fill == 'gradient':
            cmap = get_transparency_colormap(self.plotstyle.color_2d)
            ax.imshow(pdf, origin='lower', aspect='auto', extent=extent,
                      cmap=cmap, interpolation='bicubic', vmin=0)

        elif self.plotstyle.fill == 'flat':
            alphas = 1 - self.plotstyle.decreasing_contour_fractions
            next_levels = *levels[1:], np.inf
            for *level_edges, alpha in zip(levels, next_levels, alphas):
                ax.contourf(pdf, extent=extent, levels=level_edges,
                            alpha=alpha, **self.plotstyle.get_contour_kwargs())

    def _get_pdf_2d(self, xpar, ypar):
        mask = (self._get_tail_probability_mask(xpar)
                & self._get_tail_probability_mask(ypar))
        samples = self.samples[mask]
        hist2d, xedges, yedges = np.histogram2d(
            samples[xpar], samples[ypar], bins=self._get_bins(),
            weights=samples.get(self.weights_col))

        hist2d = scipy.ndimage.gaussian_filter(hist2d, self.plotstyle.smooth)

        # We want the pdf at edges, not midpoints:
        pdf = scipy.interpolate.RectBivariateSpline(
            get_midpoints(xedges), get_midpoints(yedges), hist2d
            )(xedges, yedges).T

        extent = (samples[xpar].min(), samples[xpar].max(),
                  samples[ypar].min(), samples[ypar].max())
        return pdf, extent

    def _plot_1d(self, par, label):
        ax = np.diagonal(self.axes)[self.params.index(par)]
        mask = self._get_tail_probability_mask(par)
        samples = self.samples[par][mask]
        weights = self.samples[mask].get(self.weights_col)

        hist, edges = np.histogram(samples, bins=self._get_bins(),
                                   weights=weights,
                                   density=self.plotstyle.density)

        hist = scipy.ndimage.gaussian_filter(hist, self.plotstyle.smooth)

        # We want the pdf at edges, not midpoints:
        pdf = np.array([hist[0], *(hist[1:] + hist[:-1]) / 2, hist[-1]])

        # Plot pdf
        ax.plot(edges, pdf, label=label, **self.plotstyle.kwargs_1d)

        if self.plotstyle.confidence_level is not None:
            # Plot vlines
            median, *span = self._get_median_and_central_interval(par)
            for val in (median, *span):
                ax.plot([val] * 2, [0, np.interp(val, edges, pdf)],
                        **self.plotstyle.get_vline_kwargs())

            # Plot vfill
            xvalues = np.linspace(*span, 100)
            yvalues = np.interp(xvalues, edges, pdf)
            ax.fill_between(xvalues, 0, yvalues,
                            **self.plotstyle.get_vfill_kwargs())

            # Set subplot title
            ax.set_title((self.latex_labels[par] + '${}={}$'
                          + latex_val_err(median, np.subtract(median, span))
                          + self.latex_labels.units[par]),
                         loc='left')

        # Set plot limits
        ax.autoscale(axis='x', tight=True)
        ax.autoscale(axis='y')
        ax.set_ylim(0, None)

    def _get_bins(self):
        """Implement rules for choosing bins for weighted samples."""
        if isinstance(self.plotstyle.bins, str):
            weights = self.samples.get(self.weights_col)
            if weights is None:
                n_effective = len(self.samples)
            else:
                n_effective = utils.n_effective(weights)

            if self.plotstyle.bins == 'sturges':
                return int(np.ceil(np.log2(n_effective))) + 1
            if self.plotstyle.bins == 'rice':
                return int(np.ceil(2*np.cbrt(n_effective)))
            if self.plotstyle.bins == 'sqrt':
                return int(np.ceil(np.sqrt(n_effective)))

        return self.plotstyle.bins

    def _get_tail_probability_mask(self, par):
        probabilities = (self.plotstyle.tail_probability / 2,
                         1 - self.plotstyle.tail_probability / 2)

        qmin, qmax = utils.quantile(self.samples[par], probabilities,
                                    self.samples.get(self.weights_col))
        return (self.samples[par] >= qmin) & (self.samples[par] <= qmax)

    def _get_levels(self, pdf):
        """
        Return the values of P for which
            sum(pdf[pdf > P]) == f
        for f in decreasing_contour_fractions.
        """
        sorted_pdf = [0.] + sorted(pdf.ravel())
        cdf = np.cumsum(sorted_pdf)
        cdf /= cdf[-1]
        ccdf = 1 - cdf
        return np.interp(self.plotstyle.decreasing_contour_fractions,
                         ccdf[::-1], sorted_pdf[::-1])

    def _get_median_and_central_interval(self, par):
        """
        Find median and interval enclosing the central fraction of the
        distribution given by ``.plotstyle.confidence_level``.

        Parameters
        ----------
        par: str
            Parameter name from ``self.params``.

        Returns
        -------
        (median, a, b): median and bounds of the variable.
        """
        tail_prob = (1 - self.plotstyle.confidence_level) / 2
        return utils.quantile(self.samples[par],
                              (.5, tail_prob, 1 - tail_prob),
                              self.samples.get(self.weights_col))

    def _setup_fig(self, fig=None, max_figsize=10.,
                   max_subplot_size=1.5, max_n_ticks=4):
        """Setup attributes ``.fig`` and ``.axes``."""
        n_params = len(self.params)

        if fig is not None:
            self.axes = np.reshape(fig.axes, (n_params, n_params))
            self.fig = fig
            return

        self.fig, self.axes = plt.subplots(
            n_params, n_params, squeeze=False, sharex='col',
            **self._get_subplot_kwargs(max_figsize, max_subplot_size))

        # Share y-axes in the lower triangle
        for i_row in range(n_params):
            for i_col in range(1, i_row):
                self.axes[i_row, i_col].sharey(self.axes[i_row, 0])

        for ax in self.axes.flat:
            ax.label_outer()

        # Diagonal:
        for ax in np.diagonal(self.axes):
            ax.tick_params(axis='x', direction='in', top=True, rotation=45)
            ax.tick_params(axis='y', left=False, right=False, labelleft=False)

        # Upper triangle:
        for ax in self.axes[np.triu_indices_from(self.axes, 1)]:
            ax.axis('off')

        # Lower triangle
        for ax in self.axes[np.tril_indices_from(self.axes, -1)]:
            ax.tick_params(which='both', direction='in', right=True, top=True,
                           rotation=45)

        # Left column and bottom row
        for ax in (*self.axes[-1, :], *self.axes[:, 0]):
            for axis in ax.xaxis, ax.yaxis:
                axis.set_major_locator(mpl.ticker.MaxNLocator(max_n_ticks))

        for i, par in enumerate(self.params):
            label = self.latex_labels.with_units(par)
            if i > 0:
                self.axes[i, 0].set_ylabel(label)
            self.axes[-1, i].set_xlabel(label)

    def _get_subplot_kwargs(self, max_figsize=10., max_subplot_size=1.5,
                            space=.04):
        """Return dictionary with ``figsize`` and ``gridspec_kw``."""
        box_side = min(
            max_subplot_size * ((1 + space) * len(self.params) - space),
            max_figsize - 2 * self.MARGIN_INCHES)

        side = box_side + 2 * self.MARGIN_INCHES
        margin_fraction = self.MARGIN_INCHES / side

        return {'figsize': (side, side),
                'gridspec_kw': {'wspace': space,
                                'hspace': space,
                                'bottom': margin_fraction,
                                'top': 1 - margin_fraction,
                                'left': margin_fraction,
                                'right': 1 - margin_fraction}}

    def get_current_lims(self):
        """
        Return dictionary of the form ``{par: (vmin, vmax)}`` with
        current plot limits for all parameters in ``self.params``.
        """
        return {par: ax.get_xlim()
                for par, ax in zip(self.params, self.axes[-1])}

    def set_lims(self, **lims):
        """
        Set x and y limits of the plots.

        Parameters
        ----------
        **lims
            Keyword arguments of the form ``par=(vmin, vmax)`` for those
            parameters whose limits that are to be adjusted.
        """
        for ax, par in zip(self.axes[-1], self.params):
            if par in lims:
                ax.set_xlim(lims[par])

        for ax, par in zip(self.axes[1:, 0], self.params[1:]):
            if par in lims:
                ax.set_ylim(lims[par])


class MultiCornerPlot:
    """
    Class for overlaying multiple distributions on the same corner plot
    given samples from each of the distributions.

    Methods
    -------
    plot: Make a corner plot.
    scatter_points: Plot points on a corner plot (e.g. "truths").
    set_lims: Edit the limits of a corner plot.
    plot_2d: Make one panel of a corner plot.
    """
    corner_plot_cls = CornerPlot  # Can be overriden by subclasses

    def __init__(self, dataframes, params=None, plotstyles=None,
                 weights_col='weights', labels=None,
                 **plotstyle_kwargs):
        """
        Parameters
        ----------
        dataframes: sequence of ``pandas.DataFrame`` instances
            Samples from the distributions to be plotted.

        params: list of str, optional
            Subset of columns present in all dataframes, to plot a
            reduced number of parameters.

        plotstyles: sequence of ``PlotStyle`` instances
            Determines the colors, linestyles, etc. Must have the same
            length as `dataframes`. ``None`` (default) makes automatic
            choices.

        weights_col: str, optional
            If existing, use a column with this name to set weights for
            the samples. Pass `None` to ignore, if you have a column
            named 'weights' that is not to be interpreted as weights.

        labels: sequence of strings, optional
            Legend labels corresponding to the different distributions.

        **plotstyle_kwargs:
            Passed to ``PlotStyle`` constructor to override defaults.
            Ignored if `plotstyles` is passed.

        Other parameters
        ----------------
        confidence_level: float between 0 and 1, or ``None``
            Determines the reported confidence interval around the
            median (highlighted band in 1-d marginal probability and
            numeric values in the subplot titles). If ``None``, both
            the numerical values and highlighted bands are removed.

        contour_fractions: sequence of floats
            Fractions of the distribution to enclose by 2-d contours.

        bins: int | {'rice', 'sturges', 'sqrt'}
            How many histogram bins to use, the same for all parameters.

        color_2d: str, RGB tuple, etc.
            Color used for the 2-d marginal distributions.

        contour_kwargs: dict
            Keyword arguments to `plt.contour` and `plt.contourf`

        vline_kwargs: dict
            Keyword arguments to `plt.plot` for the vertical lines
            signaling medians and 1-d confidence intervals.

        vfill_kwargs: dict
            Keyword arguments to `plt.fill_between` for 1-d plots.

        kwargs_1d: dict
            Keyword arguments to `plt.plot` for 1-d plots.

        clabel_kwargs: dict, optional
            Keyword arguments for contour labels. Pass an empty `dict`
            to use defaults. ``None`` draws no contour labels.

        fill: {'gradient', 'flat', 'none'}
            How to display 2-d marginal distributions:
            'gradient' displays the 2-d pdf with a transparency gradient
            'flat' fills the contours with a flat transparent color
            'none' shows just the contours

        smooth: float
            Smooth the 2d histograms by convolving them with a Gaussian
            kernel with this standard deviation in pixel units.
            0 (default) does no smoothing.

        density: bool
            Whether to normalize the 1-d histograms to integrate to 1.

        tail_probability: float between 0 and 1
            Disregard `tail_probability / 2` of the distribution to
            either side in the plots. Used as an automatic way of
            zooming in on the interesting part of the distribution if
            there are a few outlier samples.
            0 (default) includes all samples.
        """
        if labels is None:
            labels = [None] * len(dataframes)
        if len(labels) != len(dataframes):
            raise ValueError(
                '`dataframes` and `labels` have different lengths.')

        if plotstyles is None:
            plotstyles = PlotStyle.get_many(len(dataframes),
                                            **plotstyle_kwargs)
        elif len(plotstyles) != len(dataframes):
            raise ValueError(
                '`dataframes` and `plotstyles` have different lengths.')

        self.labels = labels

        if params is None:
            params = [par for par in dataframes[0]
                      if par in set.intersection(*map(set, dataframes))]

        self.corner_plots = [
            self.corner_plot_cls(samples, plotstyle=plotstyle, params=params,
                                 weights_col=weights_col)
            for samples, plotstyle in zip(dataframes, plotstyles)]

        # Delegate these methods
        self.set_lims = self.corner_plots[0].set_lims
        self.scatter_points = self.corner_plots[0].scatter_points

    def plot(self, max_figsize=10., max_n_ticks=4, title=None,
             legend_title=None):
        """
        Make a corner plot with all distributions overlaid.

        Parameters
        ----------
        max_figsize: float
            Maximum size in inches of a side of the square figure.

        max_n_ticks: int
            Determines the number of ticks in each subplot.

        title: str, optional
            Figure title.

        legend_title: str, optional
            Legend title.
        """
        fig = None
        for corner_plot, label in zip(self.corner_plots, self.labels):
            corner_plot.plot(fig=fig, max_figsize=max_figsize,
                             max_n_ticks=max_n_ticks, label=label, title=title,
                             legend_title=legend_title)
            fig = corner_plot.fig

    def plot_2d(self, xpar, ypar, ax=None):
        """
        Plot just one panel of the corner plot.

        Parameters
        ----------
        xpar, ypar: str
            Parameters in ``self.params`` to plot in the x and y axes.

        ax: matplotlib.axes.Axes or None
            Axes on which to draw the figure. ``None`` makes new axes.
        """
        if ax is None:
            _, ax = plt.subplots()
            get_label = self.corner_plots[0].latex_labels.with_units
            ax.set_xlabel(get_label(xpar))
            ax.set_ylabel(get_label(ypar))

        for corner_plot in self.corner_plots:
            corner_plot.plot_2d(xpar, ypar, ax)
