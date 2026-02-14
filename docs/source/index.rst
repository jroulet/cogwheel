.. cogwheel documentation master file, created by
   sphinx-quickstart on Wed Nov 27 14:13:37 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to cogwheel's documentation!
====================================

``cogwheel`` is a code for parameter estimation of gravitational wave sources.
It implements a convenient system of coordinates for sampling, a "folding" algorithm to reduce the multimodality of posteriors, and the relative binning algorithm for fast likelihood evaluation (generalized to waveforms with higher modes).
It supports likelihood marginalization over distance, as well as over all extrinsic parameters describing a merger.
It interfaces with third-party routines for downloading public data (GWOSC, ``GWpy``), generating waveforms (``lalsuite``) and sampling distributions (``PyMultiNest``, ``dynesty``, ``zeus``, ``nautilus``).
The source code is available on `GitHub <https://github.com/jroulet/cogwheel>`_.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   crash_course
   overview
   api
   acknowledgment

References
----------

* Coordinate system and folding algorithm: https://arxiv.org/abs/2207.03508 (`poster (PDF) <_static/cogwheel_poster.pdf>`_)
* Marginalization over extrinsic parameters for quadrupolar, aligned-spin signals: https://arxiv.org/abs/2210.16278
* Marginalization over extrinsic parameters for signals with precession and higher modes: https://arxiv.org/abs/2404.02435
* ``IMRPhenomXODE`` waveform approximant: https://arxiv.org/abs/2306.08774


.. Indices and tables
   ==================

   * :ref:`genindex`
   * :ref:`modindex`
   * :ref:`search`
