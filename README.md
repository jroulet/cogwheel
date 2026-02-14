# `cogwheel`

`cogwheel` is a code for parameter estimation of gravitational wave sources.
It implements a convenient system of coordinates for sampling, a "folding" algorithm to reduce the multimodality of posteriors, and the relative binning algorithm for fast likelihood evaluation (generalized to waveforms with higher modes).
It supports likelihood marginalization over distance, as well as over all extrinsic parameters describing a merger.
It interfaces with third-party routines for downloading public data (GWOSC, `GWpy`), generating waveforms (`lalsuite`) and sampling distributions (`PyMultiNest`, `dynesty`, `zeus`, `nautilus`).

## References

* Coordinate system and folding algorithm: https://arxiv.org/abs/2207.03508 ([Poster (PDF)](https://raw.githubusercontent.com/jroulet/cogwheel/main/docs/source/_static/cogwheel_poster.pdf)
)

* Marginalization over extrinsic parameters for quadrupolar, aligned-spin signals: https://arxiv.org/abs/2210.16278

* Marginalization over extrinsic parameters for signals with precession and higher modes: https://arxiv.org/abs/2404.02435

## Installation
```bash
conda install -c conda-forge cogwheel-pe
```

## Crash course

Example: how to sample a gravitational wave source posterior using `Nautilus`:
```python
from cogwheel import data
from cogwheel import sampling
from cogwheel.posterior import Posterior

parentdir = 'example'  # Directory that will contain parameter estimation runs

eventname, mchirp_guess = 'GW150914', 30
approximant = 'IMRPhenomXPHM'
prior_class = 'CartesianIntrinsicIASPrior'

filenames, detector_names, tgps = data.download_timeseries(eventname)
event_data = data.EventData.from_timeseries(
    filenames, eventname, detector_names, tgps)

post = Posterior.from_event(event_data, mchirp_guess, approximant, prior_class)

sampler = sampling.Nautilus(post, run_kwargs=dict(n_live=1000))

rundir = sampler.get_rundir(parentdir)
sampler.run(rundir)  # Will take a while
```
Load and plot the samples:
```python
import matplotlib.pyplot as plt
import pandas as pd
from cogwheel import gw_plotting

samples = pd.read_feather(rundir/sampling.SAMPLES_FILENAME)
gw_plotting.CornerPlot(samples,
                       params=sampler.sampled_params,
                       tail_probability=1e-4).plot()
plt.savefig(rundir/f'{eventname}.pdf', bbox_inches='tight')
```

## Documentation

https://cogwheel.readthedocs.io/en/latest/index.html


## Acknowledgment

This package is based upon work supported by the National Science Foundation under PHY-2012086, and PHY-1748958.

Any opinions, findings, and conclusions or recommendations expressed in `cogwheel` are those of the authors and do not necessarily reflect the views of the National Science Foundation.

This research has made use of data or software obtained from the Gravitational Wave Open Science Center (gw-openscience.org), a service of LIGO Laboratory, the LIGO Scientific Collaboration, the Virgo Collaboration, and KAGRA. LIGO Laboratory and Advanced LIGO are funded by the United States National Science Foundation (NSF) as well as the Science and Technology Facilities Council (STFC) of the United Kingdom, the Max-Planck-Society (MPS), and the State of Niedersachsen/Germany for support of the construction of Advanced LIGO and construction and operation of the GEO600 detector. Additional support for Advanced LIGO was provided by the Australian Research Council. Virgo is funded, through the European Gravitational Observatory (EGO), by the French Centre National de Recherche Scientifique (CNRS), the Italian Istituto Nazionale di Fisica Nucleare (INFN) and the Dutch Nikhef, with contributions by institutions from Belgium, Germany, Greece, Hungary, Ireland, Japan, Monaco, Poland, Portugal, Spain. KAGRA is supported by Ministry of Education, Culture, Sports, Science and Technology (MEXT), Japan Society for the Promotion of Science (JSPS) in Japan; National Research Foundation (NRF) and Ministry of Science and ICT (MSIT) in Korea; Academia Sinica (AS) and National Science and Technology Council (NSTC) in Taiwan.
