Overview
========

``cogwheel`` is a code for Bayesian parameter estimation of gravitational wave sources.
This is achieved by defining a posterior probability density in a suitable space of coordinates and then generating samples from that distribution.
These tasks are implemented across a hierarchy of classes, the top-level object being a ``Sampler`` instance, which contains other instances nested as follows::

    sampler
        .posterior
            .prior
            .likelihood
                .event_data
                .waveform_generator

Here we briefly list the responsibilities and features of each of these classes, and in the `tutorial notebooks <https://github.com/jroulet/cogwheel/tree/main/tutorials>`_ we expand on how to use them.

Sampler
-------

Instance of the abstract class :py:class:`cogwheel.sampling.Sampler` (e.g. :py:class:`~cogwheel.sampling.PyMultiNest`, :py:class:`~cogwheel.sampling.Dynesty`, :py:class:`~cogwheel.sampling.Zeus`, :py:class:`~cogwheel.sampling.Nautilus`).

* Interfaces with third-party stochastic samplers.
* Constructs the `folded distribution <https://arxiv.org/pdf/2207.03508.pdf#section*.15>`_ (to mitigate multimodality).
* Reconstructs the original distribution from samples in the folded space.
* Can run live or submit a job to a scheduler (SLURM, LSF, HTCondor).

**Tutorial:** `Sampling a posterior <https://github.com/jroulet/cogwheel/blob/main/tutorials/sampling_a_posterior.ipynb>`_

Posterior
---------

Instance of :py:class:`cogwheel.posterior.Posterior`.

* Defines a (log) posterior density :py:meth:`~cogwheel.posterior.Posterior.lnposterior`, combining a prior density and a likelihood function.
* Provides a constructor :py:meth:`~cogwheel.posterior.Posterior.from_event` that handles several choices for the prior and likelihood automatically (reference waveform for relative binning, choice of loudest and second-loudest detectors, time of arrival at leading detector, chirp-mass range, reference frequency). This is the recommended way of instantiating new posterior objects for simple cases.
* Implements likelihood maximization. :py:meth:`~cogwheel.posterior.Posterior.from_event` uses an efficient but simplified maximization to find a reference waveform over a restricted parameter space. This is enough for most purposes, but the user may further refine this waveform over the full parameter space using the more expensive :py:meth:`~cogwheel.posterior.Posterior.refine_reference_waveform`.
* Ensures that the "standard parameters" of the prior and likelihood are the same.

Prior
-----

Instance of a concrete implementation of the abstract class :py:class:`cogwheel.prior.Prior`. Pre-defined :any:`concrete implementations <cogwheel.gw_prior.combined>` are available in :py:data:`cogwheel.gw_prior.prior_registry`. Users can also define their own by subclassing :py:class:`cogwheel.prior.Prior`.

* Defines the coordinate system to sample: the parameter names, ranges, and which parameters are periodic, reflective and/or folded.
* Defines direct and inverse transformations between the sampled-parameter space and the standard-parameter space (:py:meth:`~cogwheel.prior.Prior.transform()`, :py:meth:`~cogwheel.prior.Prior.inverse_transform`).
* Defines the (log) prior probability density (:py:meth:`~cogwheel.prior.Prior.lnprior`).
* Priors for subsets of variables can be combined modularly.
* Standard parameters can be fixed (e.g. the reference frequency, tidal deformability, ...).

**Tutorial:** `Make your own prior <https://github.com/jroulet/cogwheel/blob/main/tutorials/make_your_own_prior.ipynb>`_

Likelihood
----------

Instance of a subclass of  :py:class:`cogwheel.likelihood.likelihood.CBCLikelihood`. For most cases, which subclass to use gets decided automatically based on the prior (:py:attr:`~cogwheel.gw_prior.combined.IASPrior.default_likelihood_class`).

* Defines a (log) likelihood function in terms of a "standard" system of coordinates.
* Measures, records and applies the `ASD drift correction <https://arxiv.org/pdf/1908.05644.pdf#section*.9>`_, defined as the local standard deviation of the matched-filtered score of a reference template in a particular detector.
* Implements `relative binning <https://arxiv.org/abs/1806.08792>`_ for fast likelihood evaluation (:py:meth:`~cogwheel.likelihood.relative_binning.RelativeBinningLikelihood.lnlike`).
* Stores the parameters of the reference waveform for relative binning (``.par_dic_0``).
* Implements likelihood without the relative binning approximation, for testing purposes (:py:meth:`~cogwheel.likelihood.likelihood.CBCLikelihood.lnlike_fft`).
* The :py:class:`~cogwheel.likelihood.marginalized_extrinsic.MarginalizedExtrinsicLikelihood` and :py:class:`~cogwheel.likelihood.marginalized_extrinsic_qas.MarginalizedExtrinsicLikelihoodQAS` classes implement marginalization over extrinsic parameters (and demarginalization for postprocessing), this increases robustness (`tutorial <https://github.com/jroulet/cogwheel/blob/main/tutorials/extrinsic_marginalization.ipynb>`_).
* Can overplot a signal on the whitened data (:py:meth:`~cogwheel.likelihood.likelihood.CBCLikelihood.plot_whitened_wf`).

Event data
----------

Instance of :py:class:`cogwheel.data.EventData`.

* Stores a chunk of frequency-domain data, the whitening filter, event name and GPS time.
* If you installed from source, a set of examples from GWTC-3 is shipped with the repository (``cogwheel/cogwheel/data/*.npz``), load with :py:meth:`cogwheel.data.EventData.from_npz`. Note these involved some data analysis choices.
* Alternatively you can make your own: use :py:func:`cogwheel.data.download_timeseries` to download ``hdf5`` files from GWOSC and then :py:meth:`cogwheel.data.EventData.from_timeseries`.
* Can plot a spectrogram of the whitened data (:py:meth:`~cogwheel.data.EventData.specgram`).

Tutorial: `Event data <https://github.com/jroulet/cogwheel/blob/main/tutorials/event_data.ipynb>`_

Waveform generator
------------------

Instance of :py:class:`cogwheel.waveform.WaveformGenerator`.

* Can generate a waveform in terms of (+, ×) polarizations, or strain at detectors.
* Can toggle harmonic modes (edit the attribute :py:attr:`~cogwheel.waveform.WaveformGenerator.harmonic_modes`).
* Can change approximant (edit the attribute :py:attr:`~cogwheel.waveform.WaveformGenerator.approximant`). Implemented approximants and their allowed harmonic modes are in :py:data:`waveform.APPROXIMANTS`.
