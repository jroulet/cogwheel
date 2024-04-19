"""
Provide class ``MarginalizedDistanceLikelihood`` to sample using a
likelihood marginalized over distance.
"""
import numpy as np

from .relative_binning import RelativeBinningLikelihood


class MarginalizedDistanceLikelihood(RelativeBinningLikelihood):
    """
    Modified `RelativeBinningLikelihood` that uses a likelihood
    function marginalized semi-analytically over distance.
    Thus, it removes one dimension from the parameter space.
    """
    def __init__(self, lookup_table, event_data, waveform_generator,
                 par_dic_0, fbin=None, pn_phase_tol=None,
                 spline_degree=3):
        """
        Parameters
        ----------
        event_data: Instance of `data.EventData`
        waveform_generator: Instance of `waveform.WaveformGenerator`.
        par_dic_0: dictionary with parameters of the reference waveform,
                   should be close to the maximum likelihood waveform.
        fbin: Array with edges of the frequency bins used for relative
              binning [Hz]. Alternatively, pass `pn_phase_tol`.
        pn_phase_tol: Tolerance in the post-Newtonian phase [rad] used
                      for defining frequency bins. Alternatively, pass
                      `fbin`.
        spline_degree: int, degree of the spline used to interpolate the
                       ratio between waveform and reference waveform for
                       relative binning.
        lookup_table: Instance of ``likelihood.LookupTable`` to compute
                      the marginalized likelihood.
        """
        if lookup_table.marginalized_params != {'d_luminosity'}:
            raise ValueError('Use ``LookupTable`` class.')

        super().__init__(event_data, waveform_generator, par_dic_0, fbin,
                         pn_phase_tol, spline_degree)

        self.lookup_table = lookup_table

    @property
    def params(self):
        """
        Parameters expected in `par_dic` for likelihood evaluations.
        """
        return sorted(set(self.waveform_generator.params) - {'d_luminosity'})

    def lnlike(self, par_dic):
        """
        Return log likelihood, marginalized over distance, using
        relative binning.
        """
        return self.lnlike_and_metadata(par_dic)[0]

    def lnlike_and_metadata(self, par_dic):
        """
        Parameters
        ----------
        par_dic: dict
            Keys must include ``.params``.

        Return
        ------
        lnl_marginalized: float
            Log likelihood, marginalized over distance, using relative
            binning.

        metadata: dict
            Contains the marginalized lnl, as well as a distance draw
            and its corresponding (non-marginalized) log-likelihood.
        """
        dh_hh = self._get_dh_hh_no_asd_drift(
            dict(par_dic)
            | {'d_luminosity': self.lookup_table.REFERENCE_DISTANCE})

        d_h, h_h = np.matmul(dh_hh, self.asd_drift**-2)

        lnl_marginalized = self.lookup_table.lnlike_marginalized(d_h, h_h)

        # Generate a sample of d_luminosity
        d_luminosity = self.lookup_table.sample_distance(d_h, h_h)
        lnl = d_h / d_luminosity - h_h / d_luminosity**2 / 2

        return lnl_marginalized, {'d_luminosity': d_luminosity,
                                  'lnl': lnl,
                                  'lnl_marginalized': lnl_marginalized}

    def lnlike_no_marginalization(self, par_dic):
        """
        Return log likelihood, not marginalized over distance, using
        relative binning.
        """
        return super().lnlike(par_dic)

    def postprocess_samples(self, samples):
        """
        Add a column 'd_luminosity' to a DataFrame of samples, with
        values taken randomly from the conditional posterior.
        `samples` needs to have columns for all `self.params`.

        Parameters
        ----------
        samples: Dataframe with sampled params
        """
        @np.vectorize
        def sample_distance(**par_dic):
            dh_hh = self._get_dh_hh_no_asd_drift(
                par_dic
                | {'d_luminosity': self.lookup_table.REFERENCE_DISTANCE})

            d_h, h_h = np.matmul(dh_hh, self.asd_drift**-2)
            return self.lookup_table.sample_distance(d_h, h_h)
        samples['d_luminosity'] = sample_distance(**samples[self.params])
