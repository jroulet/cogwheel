"""
Configuration file for injections.

You can copy-paste this file to some other directory of its own and edit
that copy to make a new configuration. Note: eventually that directory
will contain a lot of data.

Load using ``cogwheel.validation.load_config``.
"""
from pathlib import Path

from cogwheel import sampling
from cogwheel.validation.injection_prior import IASInjectionPrior

# Set to an integer, it is desirable to change SEED for every config so
# that every injection campaign gets different noise realizations.
# Note: the seed to the i-th injection's EventData is ``(SEED, i)``, so
# different injections within a campaign always get different data.
SEED = 0

INJECTION_DIR = Path(__file__).parent
INJECTIONS_FILENAME = 'injections.feather'

H_H_MIN = 70  # Threshold imposed on ⟨ℎ∣ℎ⟩
D_LUMINOSITY_MAX = 1.5e4  # Mpc

# Number of injections after applying the `H_H_MIN` cut:
N_INJECTIONS = 128

# Event data settings
EVENT_DATA_KWARGS = {'detector_names': 'HLV',
                     'duration': 120.,
                     'asd_funcs': ['asd_H_O3', 'asd_L_O3', 'asd_V_O3'],
                     'tgps': 0.,
                     'fmax': 1600.}

# PE settings
APPROXIMANT = 'IMRPhenomXPHM'
INJECTION_PRIOR_CLS = IASInjectionPrior
PE_PRIOR_CLS = 'IntrinsicIASPrior'
PLOTTING_PRIOR_CLS = 'AlignedSpinIASPrior'

PRIOR_KWARGS = {'mchirp_range': (1, 5),
                'q_min': .05,
                'f_ref': 50.}
LIKELIHOOD_KWARGS  = {}
REF_WF_FINDER_KWARGS = {}
SAMPLER_CLS = sampling.PyMultiNest
RUN_KWARGS = {'n_live_points': 512}
