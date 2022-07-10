# Cogwheel

`cogwheel` is a code for parameter estimation of gravitational wave sources.
It implements a convenient system of coordinates for sampling, a "folding" algorithm to reduce the multimodality of posteriors, and the relative binning algorithm for waveforms with higher modes.
It interfaces with third-party routines for downloading public data (GWOSC, \texttt{GWpy}), generating waveforms (\texttt{lalsuite}) and sampling distributions (\texttt{PyMultiNest}, \texttt{dynesty}).

## Installation
```bash
git clone git@github.com:jroulet/cogwheel.git
cd cogwheel
conda create --name <environment_name> python=3.9 numpy scipy swig h5py pkg-config matplotlib multiprocess numba pandas ipykernel python-lal python-lalsimulation pymultinest dynesty ultranest ipywidgets notebook pyarrow astropy gwpy
```
(replace `<environment_name>` by a name of your choice).


## Crash course

```python
import sys
sys.path.append('..')

from cogwheel.posterior import Posterior
from cogwheel import sampling

parentdir = 'example'  # Replace by a path to a directory that will contain parameter estimation runs

eventname, mchirp_guess = 'GW150914', 20
approximant = 'IMRPhenomXPHM'
prior_class = 'IASPrior'
post = Posterior.from_event(eventname, mchirp_guess, approximant, prior_class)

pym = sampling.PyMultiNest(post)

pym.run(pym.get_rundir(parentdir))  # Will take a while
```
