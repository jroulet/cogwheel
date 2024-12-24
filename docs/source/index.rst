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

References
----------

* Coordinate system and folding algorithm: https://arxiv.org/abs/2207.03508
* Marginalization over extrinsic parameters for quadrupolar, aligned-spin signals: https://arxiv.org/abs/2210.16278
* Marginalization over extrinsic parameters for signals with precession and higher modes: https://arxiv.org/abs/2404.02435
* ``IMRPhenomXODE`` waveform approximant: https://arxiv.org/abs/2306.08774

Acknowledgment
--------------

This package is based upon work supported by the National Science Foundation under PHY-2012086, and PHY-1748958.

Any opinions, findings, and conclusions or recommendations expressed in ``cogwheel`` are those of the authors and do not necessarily reflect the views of the National Science Foundation.

This research has made use of data or software obtained from the Gravitational Wave Open Science Center (gw-openscience.org), a service of LIGO Laboratory, the LIGO Scientific Collaboration, the Virgo Collaboration, and KAGRA. LIGO Laboratory and Advanced LIGO are funded by the United States National Science Foundation (NSF) as well as the Science and Technology Facilities Council (STFC) of the United Kingdom, the Max-Planck-Society (MPS), and the State of Niedersachsen/Germany for support of the construction of Advanced LIGO and construction and operation of the GEO600 detector. Additional support for Advanced LIGO was provided by the Australian Research Council. Virgo is funded, through the European Gravitational Observatory (EGO), by the French Centre National de Recherche Scientifique (CNRS), the Italian Istituto Nazionale di Fisica Nucleare (INFN) and the Dutch Nikhef, with contributions by institutions from Belgium, Germany, Greece, Hungary, Ireland, Japan, Monaco, Poland, Portugal, Spain. KAGRA is supported by Ministry of Education, Culture, Sports, Science and Technology (MEXT), Japan Society for the Promotion of Science (JSPS) in Japan; National Research Foundation (NRF) and Ministry of Science and ICT (MSIT) in Korea; Academia Sinica (AS) and National Science and Technology Council (NSTC) in Taiwan.


.. Indices and tables
   ==================

   * :ref:`genindex`
   * :ref:`modindex`
   * :ref:`search`
