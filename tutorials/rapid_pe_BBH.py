import sys
import os, glob, time, corner
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append('/data/tislam/works/KITP/final_repo/cogwheel/')
import cogwheel
from cogwheel import prior, gw_prior, posterior, utils, data, sampling, gw_plotting
from cogwheel.gw_prior.intrinsic_prior import IntrinsicParametersPrior
from cogwheel.likelihood.marg_likelihood import MarginalizedRelativeBinningLikelihood

t1=time.time()

# [ Directory Setup ] ##
parentdir='/bigdata/tousif/KITPRuns_Tousif/FinalRunsForPaper/'
eventname = '2022GW002'

# [ Simulated Event Data ] ##
event_data = data.EventData.from_npz(filename='/data/tislam/works/KITP/2022/PhenomD/Cogwheel_Injections/%s.npz'%eventname)
print('Read event data')

t2=time.time()
print('*\ntime taken upto downloading strain data : %.7f seconds'%(t2-t1))

# [ Relaitve binning ] ##
post = posterior.Posterior.from_event(event=event_data, 
                                      mchirp_guess=28.8,
                                      approximant='IMRPhenomXAS',
                                      prior_class=IntrinsicParametersPrior, 
                                      likelihood_class=MarginalizedRelativeBinningLikelihood,
                                      prior_kwargs={'d_hat_max': 120, 'symmetrize_lnq': True},
                                      likelihood_kwargs={'nra': 1000, 'ndec': 1000, 'nsinc_interp': 1},
                                      ref_wf_finder_kwargs={'time_range':(-0.4,-0.1)})

eventdir = post.get_eventdir(parentdir)
post.to_json(eventdir, overwrite=True)

t2=time.time()
print('*\ntime taken upto finding out max likelihood estimation : %.7f seconds'%(t2-t1))

# [ Sampling ] ##
pym = sampling.PyMultiNest(post)
pym.run_kwargs = {'n_iter_before_update': 1000,'n_live_points': 2048,'evidence_tolerance': 0.25}
rundir = pym.get_rundir(parentdir)
print('\nPE rundir : %s'%rundir)
pym.run(rundir)

t2=time.time()
print('*\ntime taken upto sampling : %.7f seconds'%(t2-t1))

# [ Read samples ] ##
cs_samples = pd.read_feather('%s/samples.feather'%rundir)
cs_samples['q'] = np.exp(-np.abs(cs_samples['lnq']))
cs_samples['psi'] = cs_samples['psi'] % np.pi

# [ Plot ] ##
plt.figure(figsize=(10,14))
gw_plotting.MultiCornerPlot(
     [cs_samples],
     labels=['cogwheel \n(factorized PE)'],
     params=['mchirp', 'q', 'chieff', 'd_luminosity', 'iota',  'ra', 'dec', 'psi', 'phi_ref']
     ).plot(max_n_ticks=4)
plt.savefig('BBH_rapid_pe_PhenomXAS.pdf', dpi=300)

t2=time.time()
print('*\ntime taken upto plotting : %.7f seconds'%(t2-t1))
print('Done!')
