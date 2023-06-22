import sys
path_to_cogwheel = '/home/zihanz/custom-waveform-model/cogwheel'
sys.path.append(path_to_cogwheel)
#sys.path.insert(0,"/home/hschia/data/love")
#sys.path.insert(-1, "/home/hschia/PE/gw_detection_ias")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from cogwheel import data
from cogwheel import gw_prior
from cogwheel import likelihood
from cogwheel import sampling
from cogwheel import waveform
from cogwheel import gw_plotting
from cogwheel import posterior


from cogwheel.gw_prior import IntrinsicAlignedSpinIASPrior
from cogwheel.likelihood import MarginalizedExtrinsicLikelihoodQAS
import cogwheel.waveform_models.quadrupole_dissipation_octupole_love
from cogwheel.waveform_models.quadrupole_dissipation_octupole_love import IntrinsicQDOLPrior,\
QuadrupoleDissipationOctupoleLoveMarginalizedExtrinsicLikelihoodQAS,\
QuadrupoleDissipationOctupoleLoveWaveformGenerator, DEFAULT_PARS

metadata = pd.read_csv(data.DATADIR/'events_metadata.csv', index_col=0)  # Chirp mass guesses

# Choose from the above options:
eventname = sys.argv[1]
mchirp_guess = metadata['mchirp'][eventname]
approximant = 'quadrupole_dissipation_octupole_love'
prior_name = 'IntrinsicQDOLPrior'
delta_tc = 0.1

post0 = posterior.Posterior.from_event(eventname, mchirp_guess, 'IMRPhenomD', 'AlignedSpinIASPrior', 
                            prior_kwargs={'symmetrize_lnq': True},
                            ref_wf_finder_kwargs={'time_range':{-delta_tc, delta_tc}})


import copy

new_like = QuadrupoleDissipationOctupoleLoveMarginalizedExtrinsicLikelihoodQAS(**copy.deepcopy(post0.likelihood).get_init_dict())
new_like.waveform_generator = QuadrupoleDissipationOctupoleLoveWaveformGenerator(**post0.likelihood.waveform_generator.get_init_dict() | {'approximant': approximant})
new_prior = IntrinsicQDOLPrior(**post0.prior.get_init_dict() | {'symmetrize_lnq': True, #'spin_quadrupole_rng':(-10000, 10000), 
                                          'schw_dissipation_rng': (0, 10000), 'spin1_dissipation_rng': (-10000,0)
                                          #'spin3_dissipation_rng':(-10000,0)
                                         #, 'spin_octupole_rng':(-10000, 10000), 
                                          #'max_tidal_deformability': 10000
                                         })
post = posterior.Posterior(new_prior, new_like)

post.likelihood.par_dic_0 = DEFAULT_PARS | post.likelihood.par_dic_0


# Run the sampler and postprocess:
pym = sampling.PyMultiNest(post, sample_prior=False)
pym.run_kwargs['n_live_points'] = 2048

parentdir = 'GW2023'  # Directory that will contain parameter estimation runs
rundir = pym.get_rundir(parentdir)
print('PE rundir:', rundir)
pym.run(rundir)