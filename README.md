# `cogwheel`

`cogwheel` is a code for parameter estimation of gravitational wave sources.
It implements a convenient system of coordinates for sampling, a "folding" algorithm to reduce the multimodality of posteriors, and the relative binning algorithm for fast likelihood evaluation (generalized to waveforms with higher modes).
It supports likelihood marginalization over distance, as well as over all extrinsic parameters describing a merger.
It interfaces with third-party routines for downloading public data (GWOSC, `GWpy`), generating waveforms (`lalsuite`) and sampling distributions (`PyMultiNest`, `dynesty`, `zeus`, `nautilus`).

## References

* Coordinate system and folding algorithm: https://arxiv.org/abs/2207.03508

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


## Overview

`cogwheel` is a code for Bayesian parameter estimation of gravitational wave sources.
This is achieved by defining a posterior probability density in a suitable space of coordinates and then generating samples from that distribution.
These tasks are implemented across a hierarchy of classes, the top-level object being a `Sampler` instance, which contains other instances nested as follows:

    sampler
        .posterior
            .prior
            .likelihood
                .event_data
                .waveform_generator

Here we briefly list the responsibilities and features of each of these classes, and in the tutorial notebooks we expand on how to use them.

### Sampler

Instance of the abstract class `cogwheel.sampling.Sampler` (e.g. `cogwheel.sampling.PyMultiNest`, `cogwheel.sampling.Dynesty`, `cogwheel.sampling.Zeus`, `cogwheel.sampling.Nautilus`).

* Interfaces with third-party stochastic samplers.
* Constructs the [folded distribution](https://arxiv.org/pdf/2207.03508.pdf#section*.15) (to mitigate multimodality).
* Reconstructs the original distribution from samples in the folded space.
* Can run live or submit a job to a scheduler (SLURM, LSF, HTCondor).

### Posterior

Instance of `cogwheel.posterior.Posterior`.

* Defines a (log) posterior density `Posterior.lnposterior()`, combining a prior density and a likelihood function.
* Provides a constructor `Posterior.from_event()` that handles several choices for the prior and likelihood automatically (reference waveform for relative binning, choice of loudest and second-loudest detectors, time of arrival at leading detector, chirp-mass range, reference frequency). This is the recommended way of instantiating new posterior objects for simple cases.
* Implements likelihood maximization. `Posterior.from_event()` uses an efficient but simplified maximization to find a reference waveform over a restricted parameter space. This is enough for most purposes, but the user may further refine this waveform over the full parameter space using the more expensive `Posterior.refine_reference_waveform()`.
* Ensures that the "standard parameters" of the prior and likelihood are the same.

### Prior

Instance of the abstract class `cogwheel.prior.Prior` (pre-built options are in `cogwheel.gw_prior.prior_registry`, alternatively users can define their own).

* Defines the coordinate system to sample: the parameter ranges, and which parameters are periodic and/or folded.
* Defines direct and inverse transformations between the sampled-parameter space and the standard-parameter space (`Prior.tranform()`, `Prior.inverse_transform()`).
* Defines the (log) prior probability density (`Prior.lnprior()`).
* Priors for subsets of variables can be combined modularly.
* Standard parameters can be fixed (e.g. the reference frequency, tidal deformability, ...).

### Likelihood

Instance of `cogwheel.likelihood.RelativeBinningLikelihood`, `cogwheel.likelihood.MarginalizedDistanceLikelihood`, `cogwheel.likelihood.MarginalizedExtrinsicLikelihood` or `cogwheel.likelihood.MarginalizedExtrinsicLikelihoodQAS`.

* Defines a (log) likelihood function in terms of a "standard" system of coordinates.
* Measures, records and applies the [ASD drift-correction](https://arxiv.org/pdf/1908.05644.pdf#section*.9), defined as the local standard deviation of the matched-filtered score of a reference template in a particular detector.
* Implements relative binning for fast likelihood evaluation (`.lnlike()`).
* Stores the parameters of the reference waveform for relative binning (`.par_dic_0`).
* Implements likelihood without the relative binning approximation, for testing purposes (`.lnlike_fft()`).
* The `Marginalized*` classes implement marginalization over various extrinsic parameters (and un-marginalization for postprocessing), this increases robustness. Use with a compatible prior, since the marginalized parameters are no longer standard parameters.
* Can overplot a signal on the whitened data (`.plot_whitened_wf()`).

### Event data

Instance of `cogwheel.data.EventData`.

* Stores a chunk of frequency-domain data, the whitening filter, event name and GPS time.
* A set of examples from GWTC-3 is shipped with the repository (`cogwheel/cogwheel/data/*.npz`), load with `EventData.from_npz()`. Note these involved some data analysis choices.
* Alternatively you can make your own: use `cogwheel.data.download_timeseries()` to download `hdf5` files from GWOSC and then `EventData.from_timeseries()`.
* Can plot a spectrogram of the whitened data (`EventData.specgram()`).

### Waveform generator

Instance of `cogwheel.waveform.WaveformGenerator`.

* Can generate a waveform in terms of (+, ×) polarizations, or strain at detectors.
* Can toggle harmonic modes (edit the attribute `harmonic_modes`).
* Can change approximant (edit the attribute `approximant`). Implemented approximants and their allowed harmonic modes are in `waveform.APPROXIMANTS`.

## Acknowledgment

This package is based upon work supported by the National Science Foundation under PHY-2012086, and PHY-1748958.

Any opinions, findings, and conclusions or recommendations expressed in `cogwheel` are those of the authors and do not necessarily reflect the views of the National Science Foundation.

This research has made use of data or software obtained from the Gravitational Wave Open Science Center (gw-openscience.org), a service of LIGO Laboratory, the LIGO Scientific Collaboration, the Virgo Collaboration, and KAGRA. LIGO Laboratory and Advanced LIGO are funded by the United States National Science Foundation (NSF) as well as the Science and Technology Facilities Council (STFC) of the United Kingdom, the Max-Planck-Society (MPS), and the State of Niedersachsen/Germany for support of the construction of Advanced LIGO and construction and operation of the GEO600 detector. Additional support for Advanced LIGO was provided by the Australian Research Council. Virgo is funded, through the European Gravitational Observatory (EGO), by the French Centre National de Recherche Scientifique (CNRS), the Italian Istituto Nazionale di Fisica Nucleare (INFN) and the Dutch Nikhef, with contributions by institutions from Belgium, Germany, Greece, Hungary, Ireland, Japan, Monaco, Poland, Portugal, Spain. KAGRA is supported by Ministry of Education, Culture, Sports, Science and Technology (MEXT), Japan Society for the Promotion of Science (JSPS) in Japan; National Research Foundation (NRF) and Ministry of Science and ICT (MSIT) in Korea; Academia Sinica (AS) and National Science and Technology Council (NSTC) in Taiwan.
