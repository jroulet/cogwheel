"""Make likelihood objects (with injections) and test them."""

from unittest import TestCase, main
import inspect

from cogwheel import data
from cogwheel import gw_prior
from cogwheel import likelihood
from cogwheel import waveform
from cogwheel.posterior import Posterior

from .test_waveform import get_random_par_dic


def get_subclasses(cls):
    """Return set of all subclasses of `cls`, recursive."""
    return set(cls.__subclasses__()) | {ssub
                                        for sub in cls.__subclasses__()
                                        for ssub in get_subclasses(sub)}


class PosteriorTestCase(TestCase):
    """Class to test priors, likelihoods and posteriors."""
    @classmethod
    def setUpClass(cls):
        """Instantiate likelihoods and priors."""
        cls.par_dic_0 = get_random_par_dic(aligned_spins=True)
        approximant = 'IMRPhenomXAS'

        event_data = data.EventData.gaussian_noise(
            eventname='test', duration=8, detector_names='HLV',
            asd_funcs=['asd_H_O3', 'asd_L_O3', 'asd_V_O3'], tgps=0.)
        event_data.inject_signal(cls.par_dic_0, approximant)

        waveform_generator = waveform.WaveformGenerator.from_event_data(
            event_data, approximant)

        lookup_table = likelihood.LookupTable()

        cls.likelihoods = []
        for likelihood_class in get_subclasses(likelihood.BaseRelativeBinning):
            kwargs = {}
            if 'lookup_table' in inspect.signature(
                    likelihood_class.__init__).parameters:
                kwargs['lookup_table'] = lookup_table

            cls.likelihoods.append(
                likelihood_class(event_data=event_data,
                                 waveform_generator=waveform_generator,
                                 par_dic_0=cls.par_dic_0,
                                 pn_phase_tol=.05,
                                 **kwargs))

        rwf = next(like for like in cls.likelihoods
                   if isinstance(like, likelihood.ReferenceWaveformFinder))
        cls.priors = [prior_class.from_reference_waveform_finder(rwf)
                      for prior_class in gw_prior.prior_registry.values()
                      if prior_class is not gw_prior.ExtrinsicParametersPrior]

    def test_prior(self):
        """
        Test that the ``.lnprior()`` method of all registered priors returns a
        float.
        """
        for prior in self.priors:
            with self.subTest(prior):
                sampled_dic = prior.inverse_transform(**self.par_dic_0)
                self.assertIsInstance(prior.lnprior(**sampled_dic), float)

    def test_likelihood(self):
        """
        Test that the ``.lnlike()`` method of all subclasses of
        ``BaseRelativeBinning`` returns a float.
        """
        for like in self.likelihoods:
            with self.subTest(like):
                self.assertIsInstance(like.lnlike(self.par_dic_0), float)


    def test_posterior(self):
        """
        Test that the ``.lnposterior()`` method of posteriors from all
        combinations of priors and likelihoods returns a float.
        """
        for prior in self.priors:
            sampled_dic = prior.inverse_transform(**self.par_dic_0)
            for like in self.likelihoods:
                if set(prior.standard_params) == set(like.params):
                    with self.subTest((prior, like)):
                        post = Posterior(prior, like)
                        self.assertIsInstance(post.lnposterior(**sampled_dic),
                                              float)

if __name__ == '__main__':
    main()
