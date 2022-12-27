import numpy as np
import pandas as pd
# from cogwheel import likelihood
# from cogwheel.gw_utils import get_fplus_fcross_0, get_geocenter_delays
# from cogwheel.waveform import FORCE_NNLO_ANGLES, compute_hplus_hcross
# from cogwheel.gw_prior import linear_free
# import lalsimulation as lalsim
from cogwheel import likelihood
from lal import CreateDict, GreenwichMeanSiderealTime
from cogwheel.sampler_free import evidence_calculator
from cogwheel import data, posterior, skyloc_angles, gw_utils, waveform

##TODO: Convert this test module to Cogwheel tests standards.

def test_components(par_dic, lk=None):

    if lk is None:
        tgps = 0.0
        eventname = 'test'
        duration = 32
        detector_names = 'HLV'
        asd_funcs = ['asd_H_O3', 'asd_L_O3', 'asd_V_O3']
        event_data = data.EventData.gaussian_noise(eventname=eventname, duration=duration,
                                                   detector_names=detector_names,
                                                   asd_funcs=asd_funcs, tgps=tgps)

        event_data.inject_signal(par_dic, 'IMRPhenomXPHM')

        mchirp = np.prod(x := [par_dic.get(k) for k in ['m1', 'm2']]) ** 0.6 / np.sum(x) ** 0.2

        likelihood_kwargs = {'lookup_table': likelihood.marginalized_distance.LookupTable()}
        ref_wf_finder_kwargs = {'time_range': (-.1, .1)}

        post = posterior.Posterior.from_event(event=event_data, mchirp_guess=mchirp,
                                              likelihood_class=likelihood.marginalized_distance.MarginalizedDistanceLikelihood,
                                              likelihood_kwargs=likelihood_kwargs,
                                              approximant='IMRPhenomXPHM',
                                              ref_wf_finder_kwargs=ref_wf_finder_kwargs,
                                              prior_class='MarginalizedDistanceIASPrior')
        lk = post.likelihood

    else:
        event_data = lk.event_data
        tgps = event_data.tgps

    intrinsic_samples, extrinsic_samples = convert_par_dic_to_intrinsic_and_extrinsic_samples(par_dic, tgps)

    x = evidence_calculator.SampleProcessing(intrinsic_samples, extrinsic_samples, lk)

    # test detector response
    passed_detector_response_test = test_detector_responses(par_dic, x, tgps)

    # test direct likelihood computation
    passed_fft_lnlike_computation_test = test_fft_lnlike_computation(par_dic, x)

    passed_rel_bin_lnlike_test = test_rel_bin_lnlike_computation(par_dic, x)



def test_detector_responses(par_dic, samples_processor, tgps):
    """
    make sure the detector response calculated through (ra, dec, tgps) and (lat, lon) is the same
    """
    det_response_edp = samples_processor.compute_detector_responses(samples_processor.detector_names,
                                                                    *[samples_processor.extrinsic_samples[k]
                                                                     for k in ('lat', 'lon', 'psi')])
    det_response_pd = gw_utils.fplus_fcross('HLV', par_dic['ra'], par_dic['dec'], par_dic['psi'], tgps)

    passed_detector_response_test = np.allclose(det_response_edp[0,].T, det_response_pd)

    print(f'passed detector response test: {passed_detector_response_test}')
    return passed_detector_response_test

def test_fft_lnlike_computation(par_dic, samples_processor):
    """
    test lnlike cacluclaiton without using relative binning. Assume single intrinsic and extrinsic samples
    """
    frequencies = samples_processor.likelihood.event_data.frequencies
    fslice = samples_processor.likelihood.event_data.fslice
    df = samples_processor.likelihood.event_data.df

    h_mpf = samples_processor.get_hplus_hcross_0(par_dic, f=frequencies, force_fslice=True, fslice=fslice)
    timeshift_intrinsic_f = samples_processor.get_intrinsic_linfree_time_shift_exp(par_dic, frequencies)
    fpfc_dp = samples_processor.compute_detector_responses(samples_processor.detector_names,
                                                           samples_processor.extrinsic_samples['lat'],
                                                           samples_processor.extrinsic_samples['lon'],
                                                           samples_processor.extrinsic_samples['psi'])[0]
    h_phasor_m = np.exp(1j * samples_processor.m_arr * par_dic['phi_ref'])
    timeshift_extrinsic_df = samples_processor.compute_extrinsic_timeshift(samples_processor.extrinsic_samples,
                                                                           frequencies)[0]
    geocentric_delays = evidence_calculator.get_geocenter_delays(
        samples_processor.detector_names,
        samples_processor.extrinsic_samples['lat'].values,
        samples_processor.extrinsic_samples['lon'].values).T  # ed
    total_delays = (geocentric_delays + samples_processor.extrinsic_samples['t_geocenter'].values[:, np.newaxis])  # ed

    signal_strain_df = np.einsum('mpf, f, dp, m, df->df', h_mpf, np.ones_like(timeshift_intrinsic_f), fpfc_dp,
                                 h_phasor_m, timeshift_extrinsic_df)
    dh = 4 * df * np.einsum('df, df',
                            signal_strain_df.conj(),
                            samples_processor.likelihood.event_data.blued_strain).real
    hh = 4 * df * np.einsum('df, df',
                            signal_strain_df.conj() * samples_processor.likelihood.event_data.wht_filter,
                            signal_strain_df * samples_processor.likelihood.event_data.wht_filter).real
    lnlike_full_inner_products = samples_processor.likelihood.lookup_table(dh, hh) + dh ** 2 / 2 / hh
    lnlike_reference = samples_processor.likelihood.lnlike(par_dic)
    rel_diff = np.abs(lnlike_full_inner_products - lnlike_reference)/lnlike_reference
    passed_fft_lnlike_computation_test = rel_diff < 1e-1

    print(f'passed fft lnlike computation test: {passed_fft_lnlike_computation_test}.')
    print(f'lnlike_full_inner_products: {lnlike_full_inner_products:.3E}, ' +
          f'lnlike_reference: {lnlike_reference:.3E}, ' +
          f'rel_diff: {rel_diff:.3E}')

    return passed_fft_lnlike_computation_test

def test_rel_bin_lnlike_computation(par_dic, samples_processor):
    dh_weights_dmpb, hh_weights_dmppb = samples_processor.get_summary()
    h_mpb = samples_processor.get_hplus_hcross_0(par_dic)

    # intrinsic timeshift is explicitly ignored
    timeshift_intrinsic_b = samples_processor.get_intrinsic_linfree_time_shift_exp(par_dic)

    fpfc_dp = samples_processor.compute_detector_responses(samples_processor.detector_names,
                                                           samples_processor.extrinsic_samples['lat'],
                                                           samples_processor.extrinsic_samples['lon'],
                                                           samples_processor.extrinsic_samples['psi'])[0]
    h_phasor_m = np.exp(1j * samples_processor.m_arr * par_dic['phi_ref'])
    timeshift_extrinsic_db = samples_processor.compute_extrinsic_timeshift(samples_processor.extrinsic_samples)[0]

    dh_relbin = np.einsum('dmpb, mpb, b, dp, m, db', dh_weights_dmpb.conj(), h_mpb, np.ones_like(timeshift_intrinsic_b),
                          fpfc_dp, h_phasor_m,
                          timeshift_extrinsic_db).real

    hh_relbin = np.einsum('dmppb, mpb, mPb, dp, dP, m', hh_weights_dmppb, h_mpb[samples_processor.m_inds, ...],
                          h_mpb.conj()[samples_processor.mprime_inds, ...], fpfc_dp, fpfc_dp,
                          h_phasor_m[samples_processor.m_inds,] * h_phasor_m[samples_processor.mprime_inds,].conj()).real

    lnlike_rel_bin = samples_processor.likelihood.lookup_table(dh_relbin, hh_relbin) + dh_relbin**2/2/hh_relbin
    lnlike_direct_cogwheel = samples_processor.likelihood.lnlike(par_dic)
    rel_diff = np.abs(lnlike_direct_cogwheel - lnlike_rel_bin)/lnlike_direct_cogwheel

    passed_rel_bin_lnlike_computation_test = rel_diff < 1e-1

    print(f'passed rel bin lnlike computation test: {passed_rel_bin_lnlike_computation_test}')
    print(f'lnlike_rel_bin: {lnlike_rel_bin:.3E}, ' +
          f'lnlike_direct_cogwheel: {lnlike_direct_cogwheel:.3E}, ' +
          f'rel_diff: {rel_diff:.3E}')
    return passed_rel_bin_lnlike_computation_test

def test_evidence_object(par_dic, samples_processor):
    dh_weights_dmpb, hh_weights_dmppb = samples_processor.get_summary()
    response_dpe = np.moveaxis(samples_processor.compute_detector_responses(
        samples_processor.detector_names, samples_processor.extrinsic_samples['lat'],
        samples_processor.extrinsic_samples['lon'], samples_processor.extrinsic_samples['psi']), (0, 1, 2), (2, 0, 1))

    timeshift_dbe = np.moveaxis(samples_processor.compute_extrinsic_timeshift(samples_processor.extrinsic_samples),
                                (0, 1, 2), (2, 0, 1))

    h_impb, timeshift_intrinsic_ib = samples_processor.compute_intrinsic_arrays(samples_processor.intrinsic_samples)

    evidence_obj = evidence_calculator.Evidence(100, samples_processor.m_arr, samples_processor.likelihood.lookup_table)


    lnlike_o, *_= evidence_obj.calculate_lnlike_and_evidence(dh_weights_dmpb, h_impb, response_dpe, timeshift_dbe,
                                               hh_weights_dmppb, samples_processor.likelihood.asd_drift,
                                               np.ones(samples_processor.n_intrinsic),
                                               np.ones(samples_processor.n_extrinsic))

    res_1 = lnlike_o.max()
    res_2 = samples_processor.likelihood.lnlike(par_dic)
    passed_evidence_object_test = np.abs((res_1 - res_2)/res_2) < 1e-1
    return passed_evidence_object_test, locals()

def convert_par_dic_to_intrinsic_and_extrinsic_samples(par_dic, tgps):
    """
    convert parameter dictionary (as used for injections, likelihood.lnlike) to intrinsic and extrinsic parameters
    used by evidence calculator, each in 1-row pandas DataFrame
    """
    gmst = GreenwichMeanSiderealTime(tgps)
    fast_params = waveform.WaveformGenerator.fast_params
    slow_params = waveform.WaveformGenerator.slow_params
    intrinsic_samples = pd.DataFrame(data={k: np.ones(1) * v for k, v in par_dic.items()
                                           if k in slow_params})
    extrinsic_samples = pd.DataFrame(data={k: np.ones(1) * v for k, v in par_dic.items()
                                           if k in fast_params})
    extrinsic_samples.drop('d_luminosity', axis=1)
    extrinsic_samples.drop('phi_ref', axis=1)
    extrinsic_samples['lat'] = extrinsic_samples['dec']
    extrinsic_samples['lon'] = skyloc_angles.ra_to_lon(extrinsic_samples['ra'], gmst)
    extrinsic_samples.drop('ra', axis=1, inplace=True)
    extrinsic_samples.drop('dec', axis=1, inplace=True)

    return intrinsic_samples, extrinsic_samples

