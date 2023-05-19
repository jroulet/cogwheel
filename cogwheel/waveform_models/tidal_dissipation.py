"""
Necessary classes and functions to run parameter estimation with tidal
dissipation parameters.

Example usage
-------------
```
import pandas as pd

from cogwheel import data
from cogwheel import gw_plotting
from cogwheel import posterior
from cogwheel import sampling
import cogwheel.waveform_models.tidal_dissipation

parentdir = 'example'  # Parameter estimation runs will be saved here
eventname = 'GW151226'
mchirp_guess = data.EVENTS_METADATA['mchirp'][eventname]
post = posterior.Posterior.from_event(
    eventname,
    mchirp_guess,
    approximant='tidal_dissipation',
    prior_class='IntrinsicTidalDissipationsPrior',
    prior_kwargs={'tidal_dissipation_rng': (-3, 3)})

pym = sampling.PyMultiNest(post, run_kwargs={'n_live_points': 256})
rundir = pym.get_rundir(parentdir)
pym.run(rundir)  # Will take a bit

samples = pd.read_feather(rundir/sampling.SAMPLES_FILENAME)
gw_plotting.CornerPlot(samples[post.prior.sampled_params]).plot()
```
"""
from cogwheel import prior
from cogwheel import waveform
from cogwheel.likelihood import MarginalizedExtrinsicLikelihoodQAS
from cogwheel.gw_prior import (UniformDetectorFrameMassesPrior,
                               UniformEffectiveSpinPrior,
                               ZeroTidalDeformabilityPrior,
                               FixedReferenceFrequencyPrior,
                               CombinedPrior,
                               RegisteredPriorMixin)


DEFAULT_PARS = waveform.DEFAULT_PARS | {'h1': 1., 'h2': 1.}


def compute_hplus_hcross_tidal_dissipation(
        f, par_dic, approximant='tidal_dissipation',
        harmonic_modes=None, lal_dic=None):
    """
    Generate frequency domain waveform with tidal dissipation
    parameters.
    Return hplus, hcross evaluated at f.

    Parameters
    ----------
    f: 1d array of type float
        Frequency array in Hz

    par_dic: dict
        Source parameters. Needs to have these keys:
            * m1, m2: component masses (Msun)
            * d_luminosity: luminosity distance (Mpc)
            * iota: inclination (rad)
            * phi_ref: phase at reference frequency (rad)
            * f_ref: reference frequency (Hz)
        plus, optionally:
            * h1, h2: component dissipation numbers
            * s1z, s2z: dimensionless spins

    approximant: 'tidal_dissipation'
        ``ValueError`` is raised if it is not 'tidal_dissipation'.

    harmonic_modes: {[(2, 2)], None}
        Ignored (unless it is not one of the accepted values, in which
        case ``ValueError`` is raised).

    lal_dic:
        Ignored.
    """
    del lal_dic
    if approximant != 'tidal_dissipation':
        raise ValueError('`approximant` must be "tidal_dissipation"`')

    if harmonic_modes not in ([(2, 2)], None):
        raise ValueError('`harmonic_modes` can only be [(2, 2)]')

    par_dic = DEFAULT_PARS | par_dic

    # TODO
    # hplus_hcross = ...
    # return hplus_hcross
    raise NotImplementedError


waveform.APPROXIMANTS['tidal_dissipation'] = waveform.Approximant(
    hplus_hcross_func=compute_hplus_hcross_tidal_dissipation)


class TidalDissipationWaveformGenerator(waveform.WaveformGenerator):
    """
    Similar to WaveformGenerator but expects tidal dissipation
    parameters in addition.
    """
    params = sorted(waveform.WaveformGenerator.params + ['h1', 'h2'])


class TidalDissipationMarginalizedExtrinsicLikelihoodQAS(
        MarginalizedExtrinsicLikelihoodQAS):
    """
    Similar to MarginalizedExtrinsicLikelihoodQAS but expects tidal
    dissipation parameters in addition.
    """
    params = sorted(MarginalizedExtrinsicLikelihoodQAS.params + ['h1', 'h2'])

    @classmethod
    def from_reference_waveform_finder(
            cls, reference_waveform_finder, approximant,
            fbin=None, pn_phase_tol=.05, spline_degree=3, **kwargs):
        """
        Instantiate with help from a `ReferenceWaveformFinder` instance,
        which provides `waveform_generator`, `event_data` and
        `par_dic_0` objects.

        Parameters
        ----------
        reference_waveform_finder: Instance of
                ``cogwheel.likelihood.ReferenceWaveformFinder``.

        approximant: str
            Approximant name.

        fbin: 1-d array or None
            Array with edges of the frequency bins used for relative
            binning [Hz]. Alternatively, pass `pn_phase_tol`.

        pn_phase_tol: float or None
            Tolerance in the post-Newtonian phase [rad] used for
            defining frequency bins. Alternatively, pass `fbin`.

        spline_degree: int
            Degree of the spline used to interpolate the ratio between
            waveform and reference waveform for relative binning.

        **kwargs:
            Keyword arguments, in case a subclass needs them.

        Return
        ------
        Instance of ``cls``.
        """
        waveform_generator = TidalDissipationWaveformGenerator(
            **reference_waveform_finder.waveform_generator.get_init_dict()
            | {'harmonic_modes': None, 'approximant': approximant})

        return cls(event_data=reference_waveform_finder.event_data,
                   waveform_generator=waveform_generator,
                   par_dic_0=reference_waveform_finder.par_dic_0,
                   fbin=fbin,
                   pn_phase_tol=pn_phase_tol,
                   spline_degree=spline_degree,
                   **kwargs)


class UniformTidalDissipationsPrior(prior.UniformPriorMixin,
                                prior.IdentityTransformMixin,
                                prior.Prior):
    """Uniform prior for tidal dissipations h1, h2."""
    range_dic = {'h1': None,
                 'h2': None}

    def __init__(self, *, tidal_dissipation_rng, **kwargs):
        """
        Parameters
        ----------
        tidal_dissipation_rng: (float, float)
            Range of tidal dissipation parameters h1, h2.
        """
        self.range_dic = {'h1': tidal_dissipation_rng,
                          'h2': tidal_dissipation_rng}
        super().__init__(**kwargs)

    def get_init_dict(self):
        """Dictionary with arguments to reproduce class instance."""
        return {'tidal_dissipation_rng': self.range_dic['h1']}


class IntrinsicTidalDissipationsPrior(RegisteredPriorMixin,
                                      CombinedPrior):
    """
    Prior for usage with
    ``TidalDissipationMarginalizedExtrinsicLikelihoodQAS``.
    Intrinsic parameters only, aligned spins, uniform in effective spin
    and detector frame component masses, uniform in tidal dissipation
    parameters, no tides.
    """
    default_likelihood_class \
        = TidalDissipationMarginalizedExtrinsicLikelihoodQAS

    prior_classes = [UniformDetectorFrameMassesPrior,
                     UniformEffectiveSpinPrior,
                     UniformTidalDissipationsPrior,
                     ZeroTidalDeformabilityPrior,
                     FixedReferenceFrequencyPrior]
