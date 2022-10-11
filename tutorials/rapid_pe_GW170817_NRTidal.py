import sys
import os, glob, time, corner
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append('/data/tislam/works/KITP/final_repo/cogwheel/')
import cogwheel
from cogwheel import prior, gw_prior, posterior, utils, data, sampling, gw_plotting
from cogwheel.gw_prior.intrinsic_prior import IntrinsicTidalPrior
from cogwheel.likelihood.marg_likelihood import MarginalizedRelativeBinningLikelihood
import gwosc
import subprocess

t1=time.time()

# [ Directory Setup ] ##
parentdir='/bigdata/tousif/KITPRuns_Tousif/FinalRunsForPaper/'
eventname = 'GW170817'

## [ Download Data ] ##
urls = gwosc.locate.get_event_urls('GW170817', version=2)  # Cleaned GW170817
outdir = data.GWOSC_FILES_DIR/eventname
subprocess.run(['wget', '-P', outdir, *urls])
filenames = [outdir/url.split('/')[-1] for url in urls] 
detector_names = ''.join(filename.name[0] for filename in filenames)
tgps = gwosc.datasets.event_gps(eventname)
event_data = data.EventData.from_timeseries(
    filenames, eventname.split('-')[0], detector_names, tgps, t_before=128., fmax=1600.)


t2=time.time()
print('*\ntime taken upto downloading strain data : %.7f seconds'%(t2-t1))

# [ Relaitve binning ] ##
post = posterior.Posterior.from_event(event=event_data, 
                                      mchirp_guess=1.198,
                                      approximant='IMRPhenomD_NRTidalv2',
                                      prior_class=IntrinsicParametersPrior, 
                                      likelihood_class=MarginalizedRelativeBinningLikelihood,
                                      prior_kwargs={'d_hat_max': 120, 'symmetrize_lnq': True},
                                      likelihood_kwargs={'nra': 5000, 'ndec': 5000, 'nsinc_interp': 8},
                                      ref_wf_finder_kwargs={'time_range':(-0.1,0.1)})

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
     params=['mchirp', 'q', 'chieff', 'l1', 'l2','d_luminosity', 'iota',  'ra', 'dec', 'psi', 'phi_ref']
     ).plot(max_n_ticks=4)
plt.savefig('GW170817_rapid_pe_PhenmDNRTidal.pdf', dpi=300)

t2=time.time()
print('*\ntime taken upto plotting : %.7f seconds'%(t2-t1))
print('Done!')
