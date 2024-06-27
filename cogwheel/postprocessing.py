"""
Post-process parameter estimation samples:
    * Diagnostics for sampler convergence
    * Diagnostics for robustness against ASD-drift choice
    * Diagnostics for relative-binning accuracy

The function `postprocess_rundir` is used to process samples from a
single parameter estimation run.
"""

import argparse
import copy
import inspect
import json
import pathlib
from pstats import Stats
from scipy.cluster.vq import kmeans
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

from cogwheel import gw_plotting
from cogwheel.likelihood import RelativeBinningLikelihood
from cogwheel import utils
from cogwheel import sampling
from cogwheel import prior

TESTS_FILENAME = 'postprocessing_tests.json'


def postprocess_rundir(rundir, relative_binning_boost=4):
    """
    Postprocess posterior samples from a single run.

    This computes:
        * Columns for standard parameters
        * Column for log likelihood
        * Auxiliary columns for log likelihood (by detector, at high
          relative binning resolution and with no ASD-drift correction
          applied)
        * Tests for log likelihood differences arising from reference
          waveform choice for setting ASD-drift
        * Tests for log likelihood differences arising from relative-
          binning accuracy.
    """
    RundirPostprocessor(rundir, relative_binning_boost).process_samples()


class RundirPostprocessor:
    """
    Postprocess posterior samples from a single run.

    The method `process_samples` executes all the functionality of the
    class. It is suggested to use the top-level function
    `postprocess_rundir` for simple usage.
    """
    LNL_COL = 'lnl'

    def __init__(self, rundir, relative_binning_boost: int = 4):
        super().__init__()

        self.rundir = pathlib.Path(rundir)
        self.relative_binning_boost = relative_binning_boost

        likelihood = utils.read_json(
            self.rundir/sampling.Sampler.JSON_FILENAME).posterior.likelihood

        if not isinstance(likelihood, RelativeBinningLikelihood):
            init_dict = likelihood.get_init_dict()
            keys = inspect.signature(RelativeBinningLikelihood).parameters
            likelihood = RelativeBinningLikelihood(
                **{key: init_dict[key] for key in keys})

        self.likelihood = likelihood
        self.samples_path = self.rundir/sampling.SAMPLES_FILENAME
        self.samples = pd.read_feather(self.samples_path)

        try:
            with open(self.rundir/TESTS_FILENAME, encoding='utf-8') as file:
                self.tests = json.load(file)
        except FileNotFoundError:
            self.tests = {'asd_drift': [],
                          'relative_binning': {},
                          'lnl_max': None,
                          'lnl_0': self.likelihood.lnlike(
                              self.likelihood.par_dic_0)}

        self._lnl_aux_cols = self.get_lnl_aux_cols(
            self.likelihood.event_data.detector_names)

        self._asd_drifts_subset = None

    @staticmethod
    def get_lnl_aux_cols(detector_names):
        """Return names of auxiliary log likelihood columns."""
        return [f'lnl_aux_{det}' for det in detector_names]

    def process_samples(self):
        """
        Call the various methods of the class sequentially, then save
        the results. This computes:
            * Columns for standard parameters
            * Column for log likelihood
            * Auxiliary columns for log likelihood (by detector, at high
              relative binning resolution and with no ASD-drift
              correction applied)
            * Tests for log likelihood differences arising from
              reference waveform choice for setting ASD-drift
            * Tests for log likelihood differences arising from
              relative binning accuracy.
        """
        print(f'Processing {self.rundir}')

        self.tests['lnl_max'] = max(self.samples['lnl'])
        print(' * Computing auxiliary likelihood products...')
        self.compute_lnl_aux()
        print(' * Testing ASD-drift correction...')
        self.test_asd_drift()
        print(' * Testing relative binning...')
        self.test_relative_binning()
        self.save_tests_and_samples()

    def compute_lnl_aux(self):
        """
        Add columns `self._lnl_aux_cols` to `self.samples` with log
        likelihood computed by detector, at high relative binning
        resolution, with no ASD-drift correction applied.
        """
        # Increase the relative-binning frequency resolution:
        try:  # few seconds faster...
            likelihood = copy.deepcopy(self.likelihood)
        except TypeError:  # ...but likelihood might be un-pickleable
            likelihood = self.likelihood.reinstantiate()

        if likelihood.pn_phase_tol:
            likelihood.pn_phase_tol /= self.relative_binning_boost
        else:
            num = self.relative_binning_boost * (len(likelihood.fbin) - 1) + 1
            likelihood.fbin = np.interp(
                np.linspace(0, 1, num),
                np.linspace(0, 1, len(likelihood.fbin)),
                likelihood.fbin)

        lnl_aux = pd.DataFrame(map(likelihood.lnlike_detectors_no_asd_drift,
                                   self._standard_samples()),
                               columns=self._lnl_aux_cols)
        utils.update_dataframe(self.samples, lnl_aux)

    def test_asd_drift(self):
        """
        Compute typical and worse-case log likelihood differences
        arising from the choice of somewhat-parameter-dependent
        asd_drift correction. Store in `self.tests['asd_drift']`.
        """
        ref_lnl = self._apply_asd_drift(self.likelihood.asd_drift)
        for asd_drift in self._get_representative_asd_drifts():
            lnl = self._apply_asd_drift(asd_drift)
            # Difference in log likelihood from changing asd_drift:
            dlnl = lnl - lnl.mean() - (ref_lnl - ref_lnl.mean())
            weights = self.samples.get(utils.WEIGHTS_NAME)
            _, dlnl_std = utils.weighted_avg_and_std(dlnl, weights=weights)
            self.tests['asd_drift'].append({'asd_drift': asd_drift,
                                            'dlnl_std': dlnl_std,
                                            'dlnl_max': np.max(np.abs(dlnl))})

    def test_relative_binning(self):
        """
        Compute typical and worst-case errors in log likelihood due to
        relative binning. Store in `self.tests['relative_binning']`.
        If the samples are weighted, the weights are considered in the
        standard deviation of the errors but ignored in the maximum.
        """
        dlnl = (self.samples[self.LNL_COL]
                - self._apply_asd_drift(self.likelihood.asd_drift))
        weights = self.samples.get(utils.WEIGHTS_NAME)
        _, dlnl_std = utils.weighted_avg_and_std(dlnl, weights=weights)
        self.tests['relative_binning'] = {'dlnl_std': dlnl_std,
                                          'dlnl_max': np.max(np.abs(dlnl))}

    def save_tests_and_samples(self):
        """Save `self.tests` and `self.samples` in `self.rundir`."""
        with open(self.rundir/TESTS_FILENAME, 'w', encoding='utf-8') as file:
            json.dump(self.tests, file, cls=utils.NumpyEncoder)

        self.samples.to_feather(self.rundir/sampling.SAMPLES_FILENAME)

    def _get_representative_asd_drifts(self, n_kmeans=5, n_subset=100,
                                       decimals=3):
        """
        Return `n_kmeans` sets of `asd_drift` generated with via k-means
        from the asd_drift of `n_subset` random samples.
        Each asd_drift is a float array of length n_detectors.
        asd_drifts are rounded to `decimals` places.
        """
        if (self._asd_drifts_subset is None
                or len(self._asd_drifts_subset) != n_subset):
            self._gen_asd_drifts_subset(n_subset)

        return np.round(kmeans(self._asd_drifts_subset, n_kmeans)[0], decimals)

    def _apply_asd_drift(self, asd_drift):
        """
        Return series of length n_samples with log likelihood for the
        provided `asd_drift`.

        Parameters
        ----------
        asd_drift: float array of length n_detectors.
        """
        return self.samples[self._lnl_aux_cols] @ asd_drift**-2

    def _gen_asd_drifts_subset(self, n_subset):
        """
        Compute asd_drifts for a random subset of the samples, store
        them in `self._asd_drifts_subset`.
        """
        subset = self.samples.sample(
            n_subset, weights=self.samples.get(utils.WEIGHTS_NAME))
        self._asd_drifts_subset = [
            self.likelihood.compute_asd_drift(sample)
            for sample in self._standard_samples(subset)]

    def _standard_samples(self, samples=None):
        """Iterator over standard parameter samples."""
        samples = samples if samples is not None else self.samples
        return (dict(sample) for _, sample in samples[
            self.likelihood.waveform_generator.params].iterrows())


def postprocess_eventdir(eventdir, reference_rundir=None, outfile=None):
    """
    Make diagnostics plots aggregating multiple runs of an event and
    save them to pdf format.
    These include a summary table of the parameters of the runs,
    number of samples vs time to completion, and corner plots comparing
    each run to a reference one.

    Parameters
    ----------
    eventdir: os.PathLike
        Path to eventdir where all the rundirs to compare are located.

    reference_rundir: os.PathLike
        Path to rundir used as reference against which to overplot
        samples. Defaults to the first rundir by name.

    outfile: os.PathLike
        Path to save output as pdf. Defaults to
        `{eventdir}/{EventdirPostprocessor.DIAGNOSTICS_FILENAME}`.
    """
    EventdirPostprocessor(eventdir, reference_rundir
                         ).postprocess_eventdir(outfile)


class EventdirPostprocessor:
    """
    Class to gather information from multiple runs of an event and
    exporting summary to pdf file.

    The method `postprocess_eventdir` executes all the functionality of the
    class. It is suggested to use the top-level function
    `postprocess_eventdir` for simple usage.
    """
    DIAGNOSTICS_FILENAME = 'diagnostics.pdf'
    DEFAULT_TOLERANCE_PARAMS = {'asd_drift_dlnl_std': .1,
                                'asd_drift_dlnl_max': .5,
                                'lnl_max_exceeds_lnl_0': 15.,
                                'lnl_0_exceeds_lnl_max': .1,
                                'relative_binning_dlnl_std': .05,
                                'relative_binning_dlnl_max': .25}
    _LABELS = {
      'n_effective': r'$N^\mathrm{samples}_\mathrm{eff}$',
      'runtime': 'Runtime (h)',
      'asd_drift_dlnl_std': r'$\sigma(\Delta\ln\mathcal{L}_{\rm ASD\,drift})$',
      'asd_drift_dlnl_max': r'$\max|\Delta\ln\mathcal{L}_{\rm ASD\,drift}|$',
      'lnl_max': r'$\max \ln\mathcal{L}$',
      'lnl_0': r'$\ln\mathcal{L}_0$',
      'relative_binning_dlnl_std': r'$\sigma(\Delta\ln\mathcal{L}_{\rm RB})$',
      'relative_binning_dlnl_max': r'$\max|\Delta\ln\mathcal{L}_{\rm RB}|$'}

    def __init__(self, eventdir, reference_rundir=None,
                 tolerance_params=None):
        """
        Parameters
        ----------
        eventdir: os.PathLike
            Path to directory containing rundirs.

        reference_rundir: os.PathLike, optional
            Path to reference run directory. Defaults to the first (by
            name) rundir in `eventdir`.

        tolerance_params: dict
            Items to update defaults from `DEFAULT_TOLERANCE_PARAMS`.
            Values higher than their tolerance are highlighted in the
            table. Keys include:

            * 'asd_drift_dlnl_std'
                Tolerable standard deviation of log likelihood
                fluctuations due to choice of reference waveform for
                ASD-drift.

            * 'asd_drift_dlnl_max'
                Tolerable maximum log likelihood fluctuation due to
                choice of reference waveform for ASD-drift.

            * 'lnl_max_exceeds_lnl_0'
                Tolerable amount by which the log likelihood of the best
                sample may exceed that of the reference waveform.

            * 'lnl_0_exceeds_lnl_max'
                Tolerable amount by which the log likelihood of the
                reference waveform may exceed that of the best sample.

            * 'relative_binning_dlnl_std'
                Tolerable standard deviation of log likelihood
                fluctuations due to the relative binning approximation.

            * 'relative_binning_dlnl_max'
                Tolerable maximum log likelihood fluctuation due to
                the relative binning approximation.
        """
        self.eventdir = pathlib.Path(eventdir)
        self.rundirs = self.get_rundirs()
        self.table = self.make_table()
        self.reference_rundir = reference_rundir

        tolerance_params = tolerance_params or {}
        if extra_keys := (tolerance_params.keys()
                          - self.DEFAULT_TOLERANCE_PARAMS.keys()):
            raise ValueError(
                f'Extraneous tolerance key(s) {extra_keys}.\n'
                f'Allowed keys are {self.DEFAULT_TOLERANCE_PARAMS.keys()}')
        self.tolerance_params = (self.DEFAULT_TOLERANCE_PARAMS
                                 | tolerance_params)

    def postprocess_eventdir(self, outfile=None):
        """
        Make diagnostics plots aggregating multiple runs of an event and
        save them to pdf format in `{eventdir}/{DIAGNOSTICS_FILENAME}`.
        These include a summary table of the parameters of the runs,
        number of samples vs time to completion, and corner plots
        comparing each run to a reference one.
        """
        outfile = outfile or self.eventdir/self.DIAGNOSTICS_FILENAME
        print(f'Diagnostic plots will be saved to "{outfile}"...')

        if self.reference_rundir:
            # Move reference_rundir to front:
            self.rundirs.insert(0, self.rundirs.pop(self.rundirs.index(
                pathlib.Path(self.reference_rundir))))

        with PdfPages(outfile) as pdf:
            self._display_table()
            plt.title(self.eventdir)
            pdf.savefig(bbox_inches='tight')

            self._scatter_nsamples_vs_runtime()
            pdf.savefig(bbox_inches='tight')

            refdir, *otherdirs = self.rundirs

            sampler = utils.read_json(refdir/sampling.Sampler.JSON_FILENAME)
            sampled_params = sampler.posterior.prior.sampled_params
            for par in sampler.posterior.prior.folded_params:
                sampled_params[sampled_params.index(par)] = f'folded_{par}'

            try:
                sampled_par_dic_0 = sampler.posterior.prior.inverse_transform(
                    **sampler.posterior.likelihood.par_dic_0)
            except prior.PriorError:
                sampled_par_dic_0 = None

            ref_samples = pd.read_feather(refdir/'samples.feather')
            for otherdir in otherdirs:
                other_samples = pd.read_feather(otherdir/'samples.feather')
                cornerplot = gw_plotting.MultiCornerPlot(
                    [ref_samples, other_samples],
                    labels=[refdir.name, otherdir.name],
                    params=sampled_params,
                    weights_col=utils.WEIGHTS_NAME,
                    tail_probability=1e-4)
                cornerplot.plot(max_n_ticks=3)
                if sampled_par_dic_0:
                    cornerplot.scatter_points(sampled_par_dic_0,
                                              adjust_lims=True)
                pdf.savefig(bbox_inches='tight')

    def get_rundirs(self):
        """
        Return a list of rundirs in `self.eventdir` for which sampling
        has completed. Ignores incomplete runs, printing a warning.
        """
        rundirs = []
        for rundir in utils.sorted_rundirs(
                self.eventdir.glob(f'{utils.RUNDIR_PREFIX}*')):
            if (rundir/TESTS_FILENAME).exists():
                rundirs.append(rundir)
            else:
                print(f'{rundir} was not post-processed, excluding.')
        return rundirs

    def make_table(self, rundirs=None):
        """
        Return a pandas DataFrame with a table that summarizes the
        different runs in `rundirs`.
        The columns report differences in the samplers' `run_kwargs`,
        plus the runtime and number of samples of each run.

        Parameters
        ----------
        rundirs: sequence of `pathlib.Path`s
            Run directories.
        """
        rundirs = rundirs or self.rundirs

        table = pd.DataFrame()
        table['run'] = [x.name for x in rundirs]
        utils.update_dataframe(table, self._collect_run_kwargs(rundirs))

        table['n_effective'] = [round(self._get_n_effective(rundir))
                                for rundir in rundirs]
        table['runtime'] = [
            Stats(str(rundir/sampling.Sampler.PROFILING_FILENAME)).total_tt
            / 3600
            for rundir in rundirs]
        utils.update_dataframe(table, self._collect_tests(rundirs))

        return table

    @staticmethod
    def _get_n_effective(rundir):
        samples = pd.read_feather(rundir/sampling.SAMPLES_FILENAME)
        weights = samples.get(utils.WEIGHTS_NAME, np.ones(len(samples)))
        return utils.n_effective(weights)

    @staticmethod
    def _collect_run_kwargs(rundirs):
        """
        Return a DataFrame aggregating run_kwargs used in sampling.
        """
        run_kwargs = []
        for rundir in rundirs:
            with open(rundir/sampling.Sampler.JSON_FILENAME,
                      encoding='utf-8') as sampler_file:
                dic = json.load(sampler_file)
                sampler_cls = utils.class_registry[dic['__cogwheel_class__']]
                init_kwargs = dic['init_kwargs']
                settings = {key: val
                            for key, val in init_kwargs['run_kwargs'].items()
                            if val != sampler_cls.DEFAULT_RUN_KWARGS.get(key)}
                run_kwargs.append({'sampler': sampler_cls.__name__,
                                   **settings})

        run_kwargs = pd.DataFrame(run_kwargs)
        const_cols = [col for col, (first, *others) in run_kwargs.items()
                      if all(first == other for other in others)]
        drop_cols = const_cols + ['outputfiles_basename', 'wrapped_params',
                                  'filepath']
        return run_kwargs.drop(columns=drop_cols, errors='ignore')

    @staticmethod
    def _collect_tests(rundirs):
        """Return a DataFrame aggregating postprocessing tests."""
        tests = []
        for rundir in rundirs:
            with open(rundir/TESTS_FILENAME, encoding='utf-8') as tests_file:
                dic = json.load(tests_file)

                asd_drift_dlnl_std = np.sqrt(np.mean(
                    [val['dlnl_std']**2 for val in dic['asd_drift']]))

                asd_drift_dlnl_max = max(
                    val['dlnl_max'] for val in dic['asd_drift'])

                tests.append({'lnl_max': dic['lnl_max'],
                              'lnl_0': dic['lnl_0'],
                              'asd_drift_dlnl_std': asd_drift_dlnl_std,
                              'asd_drift_dlnl_max': asd_drift_dlnl_max,
                              'relative_binning_dlnl_std':
                                  dic['relative_binning']['dlnl_std'],
                              'relative_binning_dlnl_max':
                                  dic['relative_binning']['dlnl_max']})
        return pd.DataFrame(tests)

    def _display_table(self, cell_size=(1., .3)):
        """Make a matplotlib figure and display the table in it."""
        cell_colors = self.table.astype(str)
        cell_colors[::2] = 'whitesmoke'
        cell_colors[1::2] = 'w'
        for key in ['asd_drift_dlnl_std',
                    'asd_drift_dlnl_max',
                    'relative_binning_dlnl_std',
                    'relative_binning_dlnl_max']:
            cell_colors[key] = self._test_color(key, self.table[key])
        dlnl_max = self.table['lnl_max'] - self.table['lnl_0']
        cell_colors['lnl_0'] = self._test_color('lnl_max_exceeds_lnl_0',
                                                dlnl_max)
        cell_colors['lnl_max'] = self._test_color('lnl_0_exceeds_lnl_max',
                                                  - dlnl_max)

        nrows, ncols = self.table.shape
        _, ax = plt.subplots(figsize=np.multiply((ncols, nrows+1), cell_size))
        ax.axis([0, 1, nrows, -1])

        tab = plt.table(
            self.table.round(3).to_numpy(),
            colLabels=self.table.rename(columns=self._LABELS).columns,
            loc='center',
            cellColours=cell_colors.to_numpy(),
            bbox=[0, 0, 1, 1])

        for cell in tab._cells.values():
            cell.set_edgecolor(None)

        tab.auto_set_column_width(range(ncols))

        plt.axhline(0, color='k', lw=1)
        plt.axis('off')
        plt.tight_layout()

    def _test_color(self, key, values):
        """
        Return a list of colors depending on the value/tolerance ratio.
        green = 0, yellow = tolerance, red = 2x tolerance.
        """
        return list(mpl.cm.RdYlGn_r(values / self.tolerance_params[key] / 2,
                                    alpha=.3))

    def _scatter_nsamples_vs_runtime(self):
        """Scatter plot number of samples vs runtime from `table`."""
        plt.figure()
        xpar, ypar = 'runtime', 'n_effective'
        plt.scatter(self.table[xpar], self.table[ypar])
        for run, *x_y in self.table[['run', xpar, ypar]].to_numpy():
            plt.annotate(run.lstrip(utils.RUNDIR_PREFIX), x_y,
                         fontsize='large')
        plt.grid()
        plt.xlim(0)
        plt.ylim(0)
        plt.xlabel(self._LABELS[xpar])
        plt.ylabel(self._LABELS[ypar])


def submit_postprocess_rundir_slurm(
        rundir, job_name=None, n_hours_limit=2, stdout_path=None,
        stderr_path=None, sbatch_cmds=('--mem-per-cpu=16G',),
        batch_path=None):
    """
    Submit a slurm job to postprocess a run directory where a
    `sampling.Sampler` has been run.
    Note this may not be necessary if the parameter estimation run was
    done through `sampling.main` with `postprocess=True`.
    """
    rundir = pathlib.Path(rundir)
    job_name = job_name or f'{rundir.name}_postprocessing'
    stdout_path = stdout_path or rundir/'postprocessing.out'
    stderr_path = stderr_path or rundir/'postprocessing.err'
    args = f'--rundir {rundir.resolve()}'
    utils.submit_slurm(job_name, n_hours_limit, stdout_path, stderr_path, args,
                       sbatch_cmds, batch_path)


def submit_postprocess_eventdir_slurm(
        eventdir, job_name=None, n_hours_limit=2, stdout_path=None,
        stderr_path=None, sbatch_cmds=(), batch_path=None):
    """
    Submit a slurm job to postprocess an event directory containing
    postprocessed rundirs.
    This will generate a pdf file in `eventdir` with diagnostic plots.
    """
    eventdir = pathlib.Path(eventdir)
    job_name = job_name or f'{eventdir.name}_diagnostics'
    stdout_path = stdout_path or eventdir/'diagnostics.out'
    stderr_path = stderr_path or eventdir/'diagnostics.err'
    args = f'--eventdir {eventdir.resolve()}'
    utils.submit_slurm(job_name, n_hours_limit, stdout_path, stderr_path, args,
                       sbatch_cmds, batch_path)


def main(*, rundir=None, eventdir=None):
    """
    Postprocess a run directory or an event directory.

    Typically, an event directory will contain many run directories
    corresponding to different sampler settings.
    Processing a run directory generates a file
    {rundir}/{TESTS_FILENAME} with test results.
    Processing an event directory aggregates these files and generates
    diagnostic plots summarizing them.

    Parameters
    ----------
    rundir: path to a run directory to postprocess, can't be set
            simultaneously with `eventdir` or a `ValueError` is raised.

    eventdir: path to an event directory to postprocess, can't be set
              simultaneously with `rundir` or a `ValueError` is raised.
    """
    if (rundir is None) == (eventdir is None):
        raise ValueError('Pass exactly one of `rundir` or `eventdir`.')

    if rundir:
        postprocess_rundir(rundir)
    else:
        postprocess_eventdir(eventdir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='postprocess either a rundir or an eventdir.')
    parser.add_argument('--rundir', help='''path to a run directory where a
                                            `sampling.Sampler` was run.''')
    parser.add_argument('--eventdir',
                        help='''path to an event directory containing
                                postprocessed rundirs.''')
    main(**vars(parser.parse_args()))
