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
import numpy as np
from cogwheel import prior
from cogwheel.prior import Prior, FixedPrior, UniformPriorMixin
from cogwheel import waveform
from cogwheel.likelihood import MarginalizedExtrinsicLikelihoodQAS
from cogwheel.gw_prior import (UniformDetectorFrameMassesPrior,
                               UniformEffectiveSpinPrior,
                               ZeroTidalDeformabilityPrior,
                               FixedReferenceFrequencyPrior,
                               UniformTidalDeformabilitiesBNSPrior,
                               CombinedPrior,
                               RegisteredPriorMixin)
from cogwheel.waveform_models.pe_waveform_cogwheel_qdol import gen_taylorF2_qdol_polar


DEFAULT_PARS = waveform.DEFAULT_PARS | {'kappa1': 1., 'kappa2': 1.} | {'h1': 1., 'h2': 1.}  | {'lambda1': 1., 'lambda2': 1.}


def compute_hplus_hcross_quadrupole_dissipation_octupole_love(
        f, par_dic, approximant='quadrupole_dissipation_octupole_love',
        harmonic_modes=None, lal_dic=None):
    """
    Generate frequency domain waveform with spin-induced quadrupole, 
    tidal dissipation, spin-induced octupole and tidal deformability parameters.
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
            * s1z, s2z: dimensionless spins
            * kappa1, kappa2: component spin-induced quadrupoles
            * h1, h2: component dissipation numbers
            * lambda1, lambda2: component spin-induced octupoles
            * l1, l2: component Love numbers
            

    harmonic_modes: {[(2, 2)], None}
        Ignored (unless it is not one of the accepted values, in which
        case ``ValueError`` is raised).

    lal_dic:
        Ignored.
    """
    del lal_dic
    if approximant != 'quadrupole_dissipation_octupole_love':
        raise ValueError('`approximant` must be "quadrupole_dissipation_octupole_love"`')

    if harmonic_modes not in ([(2, 2)], None):
        raise ValueError('`harmonic_modes` can only be [(2, 2)]')

    par_dic = DEFAULT_PARS | par_dic
    
    eta = par_dic['m1']*par_dic['m2'] / (par_dic['m1']+par_dic['m2'])**2
    Mc = (par_dic['m1']+par_dic['m2'])*eta**(3/5)
    
    # Waveform generator code must follow this ordering:
    wf_params = np.array([Mc, eta, par_dic['s1z'], par_dic['s2z'], par_dic['kappa1'], par_dic['kappa2'], 
                          par_dic['h1'], par_dic['h2'],  par_dic['lambda1'], par_dic['lambda2'], par_dic['l1'], par_dic['l2'],
                          par_dic['d_luminosity'], 0, par_dic['phi_ref'], par_dic['iota']]) # tc in gen_taylorF2_qdol_polar set to zero
    f_ref = par_dic['f_ref'] 
    
    # Avoid divergence in amplitude at f=0
    f0_is_0 = f[0] == 0 
    if f0_is_0:
        f[0] = 0.0001 # Prevent divergence in amplitude
    
    hplus, hcross = gen_taylorF2_qdol_polar(f, wf_params, f_ref)
    hplus, hcross = np.array(hplus), np.array(hcross)
    hplus_hcross = np.stack([hplus, hcross])    
    return hplus_hcross
    

waveform.APPROXIMANTS['quadrupole_dissipation_octupole_love'] = waveform.Approximant(
    hplus_hcross_func=compute_hplus_hcross_quadrupole_dissipation_octupole_love)


class QuadrupoleDissipationOctupoleLoveWaveformGenerator(waveform.WaveformGenerator):
    """
    Similar to WaveformGenerator but expects additional finite-size effect parameters.
    """
    params = sorted(waveform.WaveformGenerator.params + ['kappa1', 'kappa2', 'h1', 'h2', 'lambda1', 'lambda2'])


class QuadrupoleDissipationOctupoleLoveMarginalizedExtrinsicLikelihoodQAS(
        MarginalizedExtrinsicLikelihoodQAS):
    """
    Similar to MarginalizedExtrinsicLikelihoodQAS but expects additional
    finite-size effect parameters.
    """
    params = sorted(MarginalizedExtrinsicLikelihoodQAS.params + ['kappa1', 'kappa2', 'h1', 'h2', 'lambda1', 'lambda2'])

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
        waveform_generator = QuadrupoleDissipationOctupoleLoveWaveformGenerator(
            **reference_waveform_finder.waveform_generator.get_init_dict()
            | {'harmonic_modes': None, 'approximant': approximant})

        return cls(event_data=reference_waveform_finder.event_data,
                   waveform_generator=waveform_generator,
                   par_dic_0=reference_waveform_finder.par_dic_0,
                   fbin=fbin,
                   pn_phase_tol=pn_phase_tol,
                   spline_degree=spline_degree,
                   **kwargs)

    
################## Prior Classes for Various Finite Size Effects ##################


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
    
    
class UniformSpinQuadrupolePrior(prior.UniformPriorMixin,
                                    prior.IdentityTransformMixin,
                                    prior.Prior):
    """Uniform prior for spin-induced quadrupoles kappa1, kappa2."""
    range_dic = {'kappa1': None,
                 'kappa2': None}

    def __init__(self, *, spin_quadrupole_rng, **kwargs):
        """
        Parameters
        ----------
        spin_quadrupole_rng: (float, float)
            Range of spin-induced quadrpole parameters kappa1, kappa2.
        """
        self.range_dic = {'kappa1': spin_quadrupole_rng,
                          'kappa2': spin_quadrupole_rng}
        super().__init__(**kwargs)

    def get_init_dict(self):
        """Dictionary with arguments to reproduce class instance."""
        return {'spin_quadrupole_rng': self.range_dic['kappa1']}
    
    
class UniformSpinOctupolePrior(prior.UniformPriorMixin,
                                    prior.IdentityTransformMixin,
                                    prior.Prior):
    """Uniform prior for spin-induced octupoles lambda1, lambda2."""
    range_dic = {'lambda1': None,
                 'lambda2': None}

    def __init__(self, *, spin_octupole_rng, **kwargs):
        """
        Parameters
        ----------
        spin_quadrupole_rng: (float, float)
            Range of spin-induced octupole parameters kappa1, kappa2.
        """
        self.range_dic = {'lambda1': spin_octupole_rng,
                          'lambda2': spin_octupole_rng}
        super().__init__(**kwargs)

    def get_init_dict(self):
        """Dictionary with arguments to reproduce class instance."""
        return {'spin_octupole_rng': self.range_dic['lambda1']}
    

class BHTidalDissipationsPrior(FixedPrior):
    """Set tidal dissipation parameters to BH value."""
    standard_par_dic = {'h1': 1,
                        'h2': 1} # TODO
    
    
class BHSpinQuadrupolePrior(FixedPrior):
    """Set spin-induced quadrupole parameters to BH value."""
    standard_par_dic = {'kappa1': 1,
                        'kappa2': 1} 

    
class BHSpinOctupolePriorPrior(FixedPrior):
    """Set spin-induced octupole parameters to BH value."""
    standard_par_dic = {'lambda': 1,
                        'lambda': 1} 

    
class IntrinsicQDOLPrior(RegisteredPriorMixin,
                                      CombinedPrior):
    """
    Prior for usage with
    ``QuadrupoleDissipationOctupoleLoveMarginalizedExtrinsicLikelihoodQAS``.
    Intrinsic parameters only, aligned spins, uniform in effective spin
    and detector frame component masses, uniform in tidal dissipation
    parameters, no tides.
    """
    default_likelihood_class \
        = QuadrupoleDissipationOctupoleLoveMarginalizedExtrinsicLikelihoodQAS

    prior_classes = [UniformDetectorFrameMassesPrior,
                     UniformEffectiveSpinPrior,
                     UniformSpinQuadrupolePrior,
                     UniformSpinOctupolePrior,
                     # BHSpinOctupolePriorPrior,
                     UniformTidalDissipationsPrior,
                     # ZeroTidalDeformabilityPrior,
                     UniformTidalDeformabilitiesBNSPrior,
                     FixedReferenceFrequencyPrior]
