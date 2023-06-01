from pathlib import Path

Q_MIN = 1 / 20
MCHIRP_RANGES = [(1, 5),
                 (5, 25),
                 (25, 125)]

INJECTION_SET_DIRS = [Path(__file__).parents[1]/'data'/f'set_{i}'
                      for i in range(len(MCHIRP_RANGES))]

INJECTIONS_FILENAME = 'injections.feather'

# Threshold imposed on ⟨ℎ∣ℎ⟩:
H_H_MIN = 60

# Number of injections after applying the `H_H_MIN` cut:
N_INJECTIONS_ABOVE_THRESHOLD = 1024

# Event data settings
EVENT_DATA_KWARGS = {'detector_names': 'HLV',
                     'duration': 120,
                     'asd_funcs': ['asd_H_O3', 'asd_L_O3', 'asd_V_O3'],
                     'tgps': 0,
                     'fmax': 1600}

# PE settings
APPROXIMANT = 'IMRPhenomXPHM'
PRIOR_NAME = 'IntrinsicIASPrior'
SAMPLER_CLS = sampling.PyMultiNest
RUN_KWARGS = {'n_live_points': 512}
