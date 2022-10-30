# `cogwheel`

`cogwheel` is a code for parameter estimation of gravitational wave sources.
It implements a convenient system of coordinates for sampling, a "folding" algorithm to reduce the multimodality of posteriors, and the relative binning algorithm for fast likelihood evaluation (generalized to waveforms with higher modes).
It supports likelihood marginalization over distance, as  well as over all extrinsic parameters  describing a merger (the latter restricted to quadrupole-only waveform models with aligned spins).
It interfaces with third-party routines for downloading public data (GWOSC, `GWpy`), generating waveforms (`lalsuite`) and sampling distributions (`PyMultiNest`, `dynesty`).

The coordinate system and folding algorithm are described in an accompanying article https://arxiv.org/abs/2207.03508

## Installation
```bash
git clone git@github.com:jroulet/cogwheel.git
cd cogwheel
conda create --name <environment_name> --file requirements.txt --channel conda-forge
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

parentdir = 'example'  # Directory that will contain parameter estimation runs

eventname, mchirp_guess = 'GW150914', 30
approximant = 'IMRPhenomXPHM'
prior_class = 'IASPrior'
post = Posterior.from_event(eventname, mchirp_guess, approximant, prior_class)

pym = sampling.PyMultiNest(post, run_kwargs=dict(n_live_points=512))

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

Instance of the abstract class `cogwheel.sampling.Sampler` (e.g. `cogwheel.sampling.PyMultiNest`, `cogwheel.sampling.Dynesty`).

* Interfaces with third-party stochastic samplers.
* Constructs the [folded distribution](https://arxiv.org/pdf/2207.03508.pdf#section*.15) (to mitigate multimodality).
* The distribution to sample can be either the posterior or the prior.
* Reconstructs the original distribution from samples in the folded space.
* Can run live or submit a job to a scheduler (SLURM, LSF).

### Posterior

Instance of `cogwheel.posterior.Posterior`.

* Defines a (log) posterior density `Posterior.lnposterior()`, combining a prior density and a likelihood function.
* Provides a constructor `Posterior.from_event()` that handles several choices for the prior and likelihood
 automatically (reference waveform for relative binning, choice of loudest and second-loudest detectors, time of arrival at leading detector, chirp-mass range, reference frequency). This is the recommended way of instantiating new posterior objects for simple cases.
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

Instance of `cogwheel.likelihood.RelativeBinningLikelihood`.

* Defines a (log) likelihood function in terms of a "standard" system of coordinates.
* Measures, records and applies the [ASD drift-correction](https://arxiv.org/pdf/1908.05644.pdf#section*.9), defined as the local standard deviation of the matched-filtered score of a reference template in a particular detector.
* Implements relative binning for fast likelihood evaluation (`RelativeBinningLikelihood.lnlike()`).
* Stores the parameters of the reference waveform for relative binning (`RelativeBinningLikelihood.par_dic_0`).
* Implements likelihood without the relative binning approximation, for testing purposes (`cogwheel.likelihood.CBCLikelihood.lnlike_fft()`).
* A subclass implements distance marginalization (and un-marginalization for postprocessing), this is sometimes more robust (`cogwheel.likelihood.MarginalizedDistanceLikelihood`). Use with a compatible prior, since the distance to the source is no longer a standard parameter.
* Can overplot a signal on the whitened data (`CBCLikelihood.plot_whitened_wf()`).

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
