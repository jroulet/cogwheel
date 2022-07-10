# cogwheel

`cogwheel` is a code for parameter estimation of gravitational wave sources.
It implements a convenient system of coordinates for sampling, a "folding" algorithm to reduce the multimodality of posteriors, and the relative binning algorithm for fast likelihood evaluation (generalized to waveforms with higher modes).
It interfaces with third-party routines for downloading public data (GWOSC, `GWpy`), generating waveforms (`lalsuite`) and sampling distributions (`PyMultiNest`, `dynesty`).

## Installation
```bash
git clone git@github.com:jroulet/cogwheel.git
conda create --name <environment_name> python=3.9 numpy scipy swig h5py pkg-config matplotlib multiprocess numba pandas ipykernel python-lal python-lalsimulation pymultinest dynesty ipywidgets notebook pyarrow astropy gwpy
```
(replace `<environment_name>` by a name of your choice).


## Crash course

Example: how to sample a gravitational wave source posterior using `PyMultiNest`:
```python
path_to_cogwheel = ''  # Edit as appropriate

import sys
sys.path.append(path_to_cogwheel)

from cogwheel.posterior import Posterior
from cogwheel import sampling

parentdir = 'example'  # Replace by a path to a directory that will contain parameter estimation runs

eventname, mchirp_guess = 'GW150914', 20
approximant = 'IMRPhenomXPHM'
prior_class = 'IASPrior'
post = Posterior.from_event(eventname, mchirp_guess, approximant, prior_class)

pym = sampling.PyMultiNest(post, n_live_points=512)

rundir = pym.get_rundir(parentdir)
pym.run(rundir)  # Will take a while
```
Load and plot the samples:
```python
import pandas as pd
from cogwheel import gw_plotting

samples = pd.read_feather(rundir/sampling.SAMPLES_FILENAME)
gw_plotting.CornerPlot(samples).plot()
```
Transform the samples to a standard system of coordinates:
```python
post.prior.transform_samples(samples)

gw_plotting.CornerPlot(samples[['ra', 'dec']]).plot()
```
