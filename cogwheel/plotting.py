"""Basic corner plots."""

import itertools
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib as mpl
import pandas as pd


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
    return ['-'] + [(0, tuple([2, 2]*i + [7, 2])) for i in range(number-1)]


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


class PlotStyle:
    """
    Class that encapsulates plotting choices (colors, linestyles, etc.)
    for a corner plot of a distribution.
    """
    # Defaults:
    KWARGS_1D = dict(color='C0')
    VLINE_KWARGS = dict(alpha=.5, linewidth=1)
    VFILL_KWARGS = dict(alpha=.1)

    def __init__(self, confidence_level=.9, contour_fractions=(.5, .9),
                 color_2d='k', contour_kwargs=None, vline_kwargs=None,
                 vfill_kwargs=None, kwargs_1d=None, clabel_kwargs=None,
                 fill='gradient'):
        """
        Parameters
        ----------
        confidence_level: float between 0 and 1, or ``None``
            Determines the reported confidence interval around the
            median (highlighted band in 1-d marginal probability and
            numeric values in the subplot titles). If ``None``, both
            the numerical values and highlighted bands are removed.

        contour_fractions: sequence of floats
            Fractions of the distribution to enclose by 2-d contours.

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
        """
        self.confidence_level = confidence_level
        self.contour_fractions = contour_fractions
        self.color_2d = color_2d
        self.contour_kwargs = contour_kwargs or {}
        self.vline_kwargs = self.VLINE_KWARGS | (vline_kwargs or {})
        self.clabel_kwargs = clabel_kwargs
        self.kwargs_1d = self.KWARGS_1D | (kwargs_1d or {})
        self.fill = fill
        self.vfill_kwargs = self.VFILL_KWARGS | (vfill_kwargs or {})

    @property
    def contour_fractions(self):
        """List of fraction of the pdf to enclose by the contours."""
        return self._contour_fractions

    @contour_fractions.setter
    def contour_fractions(self, contour_fractions):
        """
        Store `contour_fractions` in decreasing order, so contour levels
        are increasing.
        """
        self._contour_fractions = sorted(contour_fractions, reverse=True)

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
        Return generator of plostyles with different colors and
        linestyles.
        """
        linestyles = gen_linestyles(number)
        colors = gen_colors(number)
        kwargs = dict(fill='flat') | kwargs
        for color, linestyle in zip(colors, linestyles):
            yield cls(contour_kwargs={'linestyles': [linestyle]},
                      color_2d=color,
                      kwargs_1d={'color': color, 'linestyle': linestyle},
                      confidence_level=None, **kwargs)



def get_transparency_colormap(color):
    """Return a colormap of a given color with varying transparency."""
    rgb = mpl.colors.to_rgb(color)
    rgba = [rgb + (alpha,) for alpha in np.linspace(0, 1)]
    return ListedColormap(rgba)


# ----------------------------------------------------------------------
# Functions related to pdfs:

def get_levels(pdf, contour_fractions):
    """
    Return the values of P for which the
        sum(pdf[pdf > P]) == f
    for f in contour_fractions.
    `contour_fractions` can be array or scalar.
    """
    sorted_pdf = [0.] + sorted(pdf.ravel())
    cdf = np.cumsum(sorted_pdf)
    cdf /= cdf[-1]
    ccdf = 1 - cdf
    return np.interp(contour_fractions, ccdf[::-1], sorted_pdf[::-1])


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
    if np.min(error) == 0:
        return f'${value:.2g}$'

    n_decimals = max(*np.ceil(-np.log10(error)).astype(int))
    if f'{np.min(error):e}'[0] == '1':
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


def get_edges(arr):
    """Get bin edges from midpoints."""
    half_dx = (arr[1] - arr[0]) / 2
    return np.concatenate(([arr[0] - half_dx], arr + half_dx))


# ----------------------------------------------------------------------
# Plotting class

class CornerPlot:
    """
    Class for making a corner plot of a multivariate distribution if you
    have samples from the distribution.
    """
    DEFAULT_LATEX_LABELS = LatexLabels()
    MARGIN_INCHES = .8

    def __init__(self, samples: pd.DataFrame, plotstyle=None, bins=None,
                 density=True, weights=None, latex_labels=None):
        """
        Parameters
        ----------
        samples: pandas DataFrame
            Columns determine the parameters to plot and rows correspond
            to samples.

        plotstyle: PlotStyle instance, optional
            Determines the colors, linestyles, etc.

        bins: int
            How many histogram bins to use, the same for all parameters.

        density: bool
            Whether to normalize the 1-d histograms to integrate to 1.

        weights: sequence of floats, optional
            An array of weights, of the same length as `samples`.
            Each value in a only contributes its associated weight
            towards the bin count (instead of 1).

        latex_labels: `LatexLabels` instance, optional.
            Maps column names to latex strings and assigns units.
        """
        self.latex_labels = latex_labels or self.DEFAULT_LATEX_LABELS
        self.plotstyle = plotstyle or PlotStyle()

        bin_edges = {}
        self.arrs_1d = {}
        self.pdfs_1d = {}
        self.pdfs_2d = {}

        if bins is None:
            if weights is None:
                bins = 'sturges'
            else:
                bins = 40

        for par, values in samples.items():
            self.pdfs_1d[par], bin_edges[par] = np.histogram(
                values, bins=bins, density=density, weights=weights)
            self.arrs_1d[par] = get_midpoints(bin_edges[par])

        for xpar, ypar in itertools.combinations(samples, 2):
            histogram_2d, _, _ = np.histogram2d(
                samples[xpar], samples[ypar],
                bins=(bin_edges[xpar], bin_edges[ypar]), weights=weights)
            # Jitter to break contour degeneracy if we have few samples:
            histogram_2d += np.random.normal(scale=1e-10,
                                             size=histogram_2d.shape)
            self.pdfs_2d[xpar, ypar] = histogram_2d.T  # Cartesian convention

        self.fig = None
        self.axes = None

    @property
    def params(self):
        """List of parameter values."""
        return list(self.arrs_1d)

    def plot(self, fig=None, title=None, max_figsize=10., max_n_ticks=4,
             tightness=None, label=None):
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

        tightness: float between 0 and 1
            Fraction of the 1-d marginal posterior to enclose. Useful
            to zoom-in around the areas with non-negligible probability
            density.

        label: str, optional
            Legend label.
        """
        self._setup_fig(fig, max_figsize, max_n_ticks=max_n_ticks)

        for par in self.params:
            self._plot_1d(par, label)

        for xpar, ypar in itertools.combinations(self.params, 2):
            self._plot_2d(xpar, ypar)

        if tightness:
            self.set_lims(**self.get_lims(tightness))

        y_title = 1 - .9 * self.MARGIN_INCHES / self.fig.get_figheight()
        self.fig.suptitle(title, y=y_title, verticalalignment='bottom')

        self.axes[0][-1].legend(
            *self.axes[0][0].get_legend_handles_labels(),
            bbox_to_anchor=(1, 1), frameon=False,
            loc='upper right', borderaxespad=0, borderpad=0)

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
                ax.axvline(point[par], color=color)

            for row, col in zip(*np.tril_indices_from(self.axes, -1)):
                self.axes[row][col].scatter(point[self.params[col]],
                                            point[self.params[row]],
                                            color=color, **kwargs)

        if not adjust_lims:
            self.set_lims(**lims)

    def _plot_2d(self, xpar, ypar):
        ax = self.axes[self.params.index(ypar), self.params.index(xpar)]
        xarr = self.arrs_1d[xpar]
        yarr = self.arrs_1d[ypar]
        pdf = self.pdfs_2d[xpar, ypar]

        levels = get_levels(pdf, self.plotstyle.contour_fractions)
        contour = ax.contour(xarr, yarr, pdf, levels=levels,
                             **self.plotstyle.get_contour_kwargs())

        if self.plotstyle.clabel_kwargs is not None:
            clabels = [fr'{100 * fraction :.0f}%'
                       for fraction in self.plotstyle.contour_fractions]
            plt.clabel(contour, fmt=dict(zip(levels, clabels)),
                       **self.plotstyle.clabel_kwargs)

        if self.plotstyle.fill == 'gradient':
            cmap = get_transparency_colormap(self.plotstyle.color_2d)
            ax.imshow(pdf, origin='lower', aspect='auto',
                      extent=(xarr.min(), xarr.max(), yarr.min(), yarr.max()),
                      cmap=cmap, interpolation='bicubic', vmin=0)

        elif self.plotstyle.fill == 'flat':
            alphas = np.subtract(1, self.plotstyle.contour_fractions)
            next_levels = list(levels[1:]) + [np.inf]
            for *level_edges, alpha in zip(levels, next_levels, alphas):
                ax.contourf(xarr, yarr, pdf, levels=level_edges, alpha=alpha,
                            **self.plotstyle.get_contour_kwargs())

    def _plot_1d(self, par, label):
        ax = np.diagonal(self.axes)[self.params.index(par)]

        # Plot pdf
        ax.plot(self.arrs_1d[par], self.pdfs_1d[par], label=label,
                **self.plotstyle.kwargs_1d)

        if self.plotstyle.confidence_level is not None:
            # Plot vlines
            median, *span = self.get_median_and_central_interval(par)
            for val in (median, *span):
                ax.plot([val]*2, [0, np.interp(val, self.arrs_1d[par],
                                               self.pdfs_1d[par])],
                        **self.plotstyle.get_vline_kwargs())

            # Plot vfill
            xvalues = np.linspace(*span, 100)
            yvalues = np.interp(xvalues, self.arrs_1d[par], self.pdfs_1d[par])
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

    def get_median_and_central_interval(self, par, confidence_level=None):
        """
        Given a 1 dimensional pdf, find median and interval enclosing
        the central given fraction of the population.

        Parameters
        ----------
        par: str
            Parameter name from ``self.params``.

        confidence_level: float between 0 and 1, optional
            Determines the reported confidence interval around the
            median (highlighted band in 1-d marginal probability and
            numeric values in the subplot titles). If ``None``, both
            the numerical values and highlighted bands are removed.

        Returns
        -------
        (median, a, b): median and bounds of the variable.
        """
        confidence_level = confidence_level or self.plotstyle.confidence_level
        edges = get_edges(self.arrs_1d[par])
        cum_prob = np.concatenate(([0.], np.cumsum(self.pdfs_1d[par])))
        cum_prob /= cum_prob[-1]
        tail_prob = (1 - confidence_level) / 2
        return np.interp((.5, tail_prob, 1 - tail_prob), cum_prob, edges)

    def _setup_fig(self, fig=None, max_figsize=10.,
                   max_subplot_size=1.5, max_n_ticks=4):
        n_params = len(self.params)

        if fig is not None:
            self.axes = np.reshape(fig.axes, (n_params, n_params))
            self.fig = fig
            return

        self.fig, self.axes = plt.subplots(
            n_params, n_params, squeeze=False, sharex='col', sharey='row',
            **self._get_subplot_kwargs(max_figsize, max_subplot_size))

        # Diagonal:
        for ax in np.diagonal(self.axes):
            ax.get_shared_y_axes().remove(ax)
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
        for ax in np.r_[self.axes[-1, :], self.axes[:, 0]]:
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
                'gridspec_kw': dict(wspace=space, hspace=space,
                                    bottom=margin_fraction,
                                    top=1 - margin_fraction,
                                    left=margin_fraction,
                                    right=1 - margin_fraction)}

    def get_lims(self, tightness=1.):
        """
        Return dictionary of the form ``{par: (vmin, vmax)}`` with
        limits for plotting for all parameters in ``self.params``.

        Parameters
        ----------
        tightness: float between 0 and 1
            Fraction of the 1-d marginal posterior to enclose. Useful
            to zoom-in around the areas with non-negligible probability
            density.
        """
        return {par: self.get_median_and_central_interval(par, tightness)[1:]
                for par in self.params}

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
        **lims:
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
    if you have samples from each of the distributions.
    """
    corner_plot_cls = CornerPlot  # Can be overriden by subclasses

    def __init__(self, dataframes, labels=None, bins=40, params=None,
                 density=True, weights_col='weights',
                 **plotstyle_kwargs):
        """
        Parameters
        ----------
        dataframes: sequence of ``pandas.DataFrame`` instances
            Samples from the distributions to be plotted.

        labels: sequence of strings, optional
            Legend labels corresponding to the different distributions.

        bins: int
            How many histogram bins to use, the same for all parameters
            and all distributions.

        params: list of str, optional
            Subset of columns present in all dataframes, to plot a
            reduced number of parameters.

        density: bool
            Whether to normalize the 1-d histograms to integrate to 1.

        weights_col: str, optional
            If existing, use a column with this name to set weights for
            the samples. Pass `None` to ignore, if you have a column
            named 'weights' that is not to be interpreted as weights.

        **plotstyle_kwargs:
            Passed to ``PlotStyle`` constructor to override defaults.
        """
        if labels is None:
            labels = [None for _ in dataframes]

        if len(labels) != len(dataframes):
            raise ValueError(
                '`dataframes` and `labels` have different lengths.')

        self.labels = labels

        params = params or slice(None)
        plotstyles = PlotStyle.get_many(len(dataframes),
                                        **plotstyle_kwargs)
        self.corner_plots = [
            self.corner_plot_cls(samples[params], next(plotstyles), bins,
                                 density, weights=samples.get(weights_col))
            for samples in dataframes]

    def plot(self, max_figsize=10., max_n_ticks=4, tightness=None,
             title=None):
        """
        Make a corner plot with all distributions overlaid.

        Parameters
        ----------
        max_figsize: float
            Maximum size in inches of a side of the square figure.

        max_n_ticks: int
            Determines the number of ticks in each subplot.

        tightness: float between 0 and 1
            Fraction of the 1-d marginal posterior to enclose. Useful
            to zoom-in around the areas with non-negligible probability
            density.
        """
        fig = None
        for corner_plot, label in zip(self.corner_plots, self.labels):
            corner_plot.plot(fig=fig, max_figsize=max_figsize,
                             max_n_ticks=max_n_ticks, label=label, title=title)
            fig = corner_plot.fig

        if tightness:
            self.set_lims(**self.get_lims(tightness))

    def get_lims(self, tightness=1.):
        """
        Return dictionary of the form ``{par: (vmin, vmax)}`` with
        limits for plotting for all parameters in ``self.params``.

        Parameters
        ----------
        tightness: float between 0 and 1
            Fraction of the 1-d marginal posterior to enclose. Useful
            to zoom-in around the areas with non-negligible probability
            density.
        """
        all_lims = [corner_plot.get_lims(tightness)
                    for corner_plot in self.corner_plots]
        lims = {}
        for par in self.corner_plots[0].params:
            values = [value for lim in all_lims for value in lim[par]]
            lims[par] = (min(values), max(values))

        return lims

    def set_lims(self, **lims):
        """
        Set x and y limits of the plots.

        Parameters
        ----------
        **lims:
            Keyword arguments of the form ``par=(vmin, vmax)`` for those
            parameters whose limits that are to be adjusted.
        """
        self.corner_plots[0].set_lims(**lims)

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
        self.corner_plots[0].scatter_points(scatter_points, colors,
                                            adjust_lims, **kwargs)
