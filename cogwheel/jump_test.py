import numpy as np
import sys

sys.path.append('..')
from cogwheel import gw_plotting


def perform_jump_test(posterior, samples, jump_function,
                      label='Original', weights_col='weights'):
    nsamples = len(samples)
    lnposterior = np.vectorize(posterior.lnposterior)
    params = posterior.prior.sampled_params
    jumped_samples = jump_function(samples)

    relevant = ~np.all(samples == jumped_samples, axis=1)
    samples = samples[relevant]
    jumped_samples = jumped_samples[relevant]

    lnprob = lnposterior(**samples[params])
    lnprob_jumped = lnposterior(**jumped_samples[params])
    # jump_probability = 1 / (1 + np.exp(lnprob - lnprob_jumped))
    jump_probability = np.exp(lnprob_jumped - lnprob)

    jump = np.random.uniform(0, 1, len(samples)) < jump_probability
    print(f'{100 * np.count_nonzero(jump) / nsamples:.1f}% of samples jumped.')

    test_samples = samples.copy()
    test_samples[jump] = jumped_samples[jump]

    gw_plotting.MultiCornerPlot.from_samples(
        [samples, test_samples],
        labels=[label, 'Jump test'],
        weights_col=weights_col,
        params=params
        ).plot(tightness=.999, max_n_ticks=4)
    return test_samples

lnq0 = -2.  # Monkey-patchable
@np.vectorize
def _lnq_jump_aux(lnq):
    """This function is its own inverse and its Jacobian is 1."""
    lnq1 = lnq0 / 2
    if lnq < lnq0:
        return lnq
    if lnq < lnq1:
        return lnq + lnq1 - lnq0
    return lnq - (lnq1 - lnq0)


def lnq_jump_function(samples):
    """
    Return a copy of a samples dataframe where the jump has been applied
    to column 'lnq'.
    """
    jumped_samples = samples.copy()
    jumped_samples['lnq'] = _lnq_jump_aux(samples['lnq'])
    return jumped_samples


def psi_jump_function(samples):
    """
    Return a copy of a samples dataframe where the jump has been applied
    to column 'lnq'.
    """
    jumped_samples = samples.copy()
    psi = samples['psi']
    jumped_samples['psi'] = (
        psi + np.pi/4 + np.pi/2*((psi%(np.pi/2)) > np.pi/4)
        ) % np.pi
    return jumped_samples


def cumchidiff_jump_function(samples):
    jumped_samples = samples.copy()
    jumped_samples['cumchidiff'] = (samples['cumchidiff'] + 1/2) % 1


def weighted_quantile(values, quantiles, sample_weight=None,
                      values_sorted=False, old_style=False):
    """ Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :param old_style: if True, will correct output to be consistent
        with numpy.percentile.
    :return: numpy.array with computed quantiles.
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), \
        'quantiles should be in [0, 1]'

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)

def gen_reflection_jump_function(params, samples):
    """
    Generate a jump function that is robust to linearly correlated variables.
    Should work best if ``params`` are unimodal parameters.
    """
    def jump_function(samples):
        jumped_samples = samples.copy()
        valid = np.ones(len(samples), dtype=bool)
        inds = np.arange(len(samples))
        min_dic = {}
        max_dic = {}

        # Remove outliers:
        for par in params:
            min_dic[par], max_dic[par] = weighted_quantile(
                samples[valid][par], (0.2, 0.8),
                samples[valid].get('weights'))
            valid[inds[valid][samples[par][valid] < min_dic[par]]] = False
            valid[inds[valid][samples[par][valid] > max_dic[par]]] = False

        for par in params:
            @np.vectorize
            def aux_jump(value):
                if value < min_dic[par] or value > max_dic[par]:
                    return value
                return min_dic[par] + max_dic[par] - value

            jumped_samples[par].iloc[inds[valid]] = aux_jump(samples[par][valid])

        return jumped_samples
    return jump_function

def gen_shift_jump_function(params, samples):
    """
    Generate a jump function that is robust to linearly correlated variables.
    Should work best if ``params`` are unimodal parameters.
    """
    def jump_function(samples):
        jumped_samples = samples.copy()
        valid = np.ones(len(samples), dtype=bool)
        inds = np.arange(len(samples))
        min_dic = {}
        max_dic = {}

        # Remove outliers:
        for par in params:
            min_dic[par], max_dic[par] = weighted_quantile(
                samples[valid][par], (0.2, 0.8),
                samples[valid].get('weights'))
            valid[inds[valid][samples[par][valid] < min_dic[par]]] = False
            valid[inds[valid][samples[par][valid] > max_dic[par]]] = False

        for par in params:
            @np.vectorize
            def aux_jump(value):
                if value < min_dic[par] or value > max_dic[par]:
                    return value
                return ((value - min_dic[par] + (max_dic[par]-min_dic[par]) / 2)
                        % (max_dic[par] - min_dic[par])) + min_dic[par]

            jumped_samples[par].iloc[inds[valid]] = aux_jump(samples[par][valid])

        return jumped_samples
    return jump_function