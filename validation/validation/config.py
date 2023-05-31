from pathlib import Path

MCHIRP_RANGES = [(1, 5),
                 (5, 25),
                 (25, 125)]

INJECTION_SET_DIRS = [Path(__file__).parents[1]/'data'/f'set_{i}'
                      for i in range(len(MCHIRP_RANGES))]

INJECTIONS_FILENAME = 'injections.feather'

Q_MIN = 1 / 20

# Threshold imposed on ⟨h|h⟩:
H_H_MIN = 60

# Number of injections after applying the h_h_min_cut:
MIN_N_INJECTIONS_ABOVE_THRESHOLD = 1024

DETECTOR_NAMES = 'HLV'
ASDS = ['asd_H_O3', 'asd_L_O3', 'asd_V_O3']
