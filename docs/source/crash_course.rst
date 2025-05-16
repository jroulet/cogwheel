Crash course
------------

Example: how to sample a gravitational wave source posterior using ``Nautilus``:

.. code-block:: python

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

Load and plot the samples:

.. code-block:: python


    import matplotlib.pyplot as plt
    import pandas as pd
    from cogwheel import gw_plotting

    samples = pd.read_feather(rundir/sampling.SAMPLES_FILENAME)
    gw_plotting.CornerPlot(samples,
                           params=sampler.sampled_params,
                           tail_probability=1e-4).plot()
    plt.savefig(rundir/f'{eventname}.pdf', bbox_inches='tight')