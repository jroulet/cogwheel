"""
Here go things that talk to the pipeline.
This file is not intended to be public.
We should keep it as small as possible.
"""

import os
import sys
from copy import deepcopy
import numpy as np
from scipy.interpolate import interp1d

PIPELINE_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'gw_detection_ias'))

COGWHEEL_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..'))

sys.path.insert(0, COGWHEEL_PATH)
sys.path.insert(0, PIPELINE_PATH)

import triggers_single_detector as trig
from cogwheel import data

# ordering of detector names for passing fnames
DETECTOR_NAMES_ORDERED = 'HLV'
# default maximization range
DEFAULT_TC_RANGE = (-.1, .1)
# default minimum mass ratio m2 / m1
DEFAULT_Q_MIN = 1 / 20
# default total time interval for data
DEFAULT_T_INTERVAL = 128.

event_registry = {}  # EventMetadata instances register themselves here


def get_event_data(eventname):
    """Return `data.EventData` instance of the desired event."""
    return event_registry[eventname].get_event_data()


def guess_bank_id(mchirp, i_subbank=0):
    """
    Use mchirp to guess of bank_id=(source, i_multibank, i_subbank).
    `source` will be 'BNS' or 'BBH'.
    No attempt to guess i_subbank is done.
    """
    mchirp_multibank_edges = 1.1, 1.3, 3 * 2**-.2, 5, 10, 20, 40
    n_bns = 3
    n_bbh = 5
    bank_ids = [*[('BNS', i_mb, i_subbank) for i_mb in range(n_bns)],
                *[('BBH', i_mb, i_subbank) for i_mb in range(n_bbh)]]
    return bank_ids[np.searchsorted(mchirp_multibank_edges, mchirp)]

#@np.vectorize
# take off decorator because may be causing problems in cluster runs?
def _x_of_mchirp(mchirp):
    """
    Chirp-mass reparametrization in which the uncertainty
    is approximately homogeneous.
    """
    mchirp0 = 60
    if not hasattr(mchirp, '__len__') and mchirp > mchirp0:
        return mchirp
    a = -3/5 * mchirp0**(8/3)
    b = 8/5 * mchirp0
    # compute for all
    mchirpout = a * mchirp**(-5/3) + b
    if isinstance(mchirp, np.ndarray):
        # if input is array, revert to original mchirp where > mchirp0
        revertmask = mchirp > mchirp0
        mchirpout[revertmask] = mchirp[revertmask]
    return mchirpout

_mchirp_grid = np.geomspace(.1, 1000, 10000)
_mchirp_of_x = interp1d(_x_of_mchirp(_mchirp_grid), _mchirp_grid)

def estimate_mchirp_range(mchirp, sigmas=5, snr=8):
    """
    Return an array with the minimum and maximum estimated
    values of mchirp with posterior support.
    Intended for setting ranges for sampling.
    Keep in mind this is very approximate, check your results
    responsibly.

    Parameters
    ----------
    mchirp: Estimate of the center of the mchirp distribution.
    sigmas: How big (conservative) to make the range compared
            to the expected width of the distribution.
    snr: Signal-to-noise ratio, lower values give bigger ranges.
    """
    x = _x_of_mchirp(mchirp)
    dx = 200. * sigmas / snr
    mcmin = _mchirp_of_x(x - dx)
    mcmax = _mchirp_of_x(x + dx)
    return np.array([mcmin, mcmax])

def _get_linear_free_shift_from_bank(bank, calpha=None, **pars):
    if calpha is not None:
        return bank.get_linear_free_shift_from_calpha(calpha)
    return bank.get_linear_free_shift_from_pars(**pars)


def get_linear_free_time_shift(triggerlists, calpha=None, i_refdet=0,
                               max_tsep=0.07, **pars):
    """
    max_tsep: maximum time difference (seconds) between shifts
      from different triggerlists above which error is raised
    """
    if isinstance(triggerlists, trig.TriggerList):
        triggerlists = [triggerlists]
    shifts = [_get_linear_free_shift_from_bank(tgl.templatebank,
                                               calpha=calpha, **pars)
              for tgl in triggerlists]
    if np.abs(np.max(shifts) - np.min(shifts)) > max_tsep:
        raise ValueError(f'Disparate trigger times! Shifts = {shifts}')
    return shifts[i_refdet]


def _get_f_strain_psd_from_triggerlist(triggerlist, tgps, tcoarse,
                                       t_interval):
    """
    Extract frequency array, frequency domain strain and PSD from a
    triggerlist instance.
    """
    f_sampling = int(1 / triggerlist.dt)

    t_start = tgps - triggerlist.time[0] - tcoarse
    ind_start = int(t_start * f_sampling)
    ind_end = ind_start + int(t_interval * f_sampling)
    nfft = ind_end - ind_start

    # Compute whitening filter for the data segment
    wt_filter_fd = triggerlist.templatebank.wt_filter_fd
    wt_filter_fd_full = trig.utils.change_filter_times_fd(
        wt_filter_fd, triggerlist.fftsize, len(triggerlist.strain))
    wt_filter_td_full = np.fft.irfft(wt_filter_fd_full)
    wt_filter_td_nfft = trig.utils.change_filter_times_td(
        wt_filter_td_full, len(wt_filter_td_full),
        int(t_interval * f_sampling))
    wt_filter_fd_nfft = np.fft.rfft(wt_filter_td_nfft)

    wt_data_td_nfft = triggerlist.strain[ind_start : ind_end]
    wt_data_fd_nfft = np.fft.rfft(wt_data_td_nfft)

    frequencies = np.fft.rfftfreq(nfft, triggerlist.dt)
    strain = (wt_data_fd_nfft / wt_filter_fd_nfft
              * np.sqrt(t_interval / 2 / nfft))
    psd = np.abs(wt_filter_fd_nfft) ** -2

    return frequencies, strain, psd


def get_f_strain_psd_dic(triggerlists, tgps, tcoarse, t_interval):
    """
    Return a dictionary with keys `'strain', 'psd', 'frequencies'`.
    The values for 'strain' and 'psd' are arrays of shape
    `(n_triggerlists, n_frequencies)`. The value for 'frequencies'
    is an array of length 'n_frequencies'
    """
    dic = {}

    data_by_det = [_get_f_strain_psd_from_triggerlist(triggerlist, tgps,
                                                      tcoarse, t_interval)
                   for triggerlist in triggerlists]

    freq_arrs, dic['strain'], dic['psd'] = (
        np.array(arr) for arr in zip(*data_by_det))
    dic['frequencies'], *copies = freq_arrs
    for freq_arr in copies:
        np.testing.assert_array_equal(dic['frequencies'], freq_arr)

    return dic


class EventMetadata(data.utils.JSONMixin):
    """
    Class that can be used for data information about events, and
    provides a method `get_event_data()` to create EventData instances.
    """
    def __init__(self, eventname, tgps, mchirp_range=None,
                 q_min=DEFAULT_Q_MIN, t_interval=DEFAULT_T_INTERVAL,
                 tcoarse=None, bank_id=None, fnames=None,
                 load_data=False, triggerlist_kw=None,
                 tc_range=DEFAULT_TC_RANGE, ref_det_name=None,
                 calpha=None, compute_linear_free_shift=False,
                 max_tsep=0.07, compute_par_dic_0=False, **par_dic_0):
        """
        Instantiate `EventMetadata`, register the instance in
        `event_registry`.

        Parameters
        ----------
        eventname: string.
        tgps: float, GPS time.
        mchirp_range: 2-tuple of floats, chirp-mass range to probe.
        q_min: float between 0 and 1, minimum mass ratio to probe.
        t_interval: length of data to analyze, in seconds.
        tcoarse: where to "center" the event (at `tgps`) in seconds,
                 defaults to `t_interval / 2`
        bank_id: 3-tuple with (source, i_multibank, i_subbank) where
                 `source` is 'BNS', 'NSBH' or 'BBH', `i_multibank` and
                 `i_subbank` are integers. Can be left as `None` to
                 guess a good choice from `mchirp_range`.
        fnames: 3-list with strings or `None`s or `0`s specifying
                the Hanford, Livingston and Virgo triggerlists.
                Strings are interpreted as filenames, `None` will ignore
                that detector, and `0` means that the pipeline's file
                will be retrieved. Passing a single `None` (default) is
                equivalent to [0, 0, 0].
        load_data: 3-tuple with booleans, whether to reload the data in
                   the triggerlist for each detector H, L, V.
                   `False` means it's loaded as is from json.
                   A single boolean is ok, applied to all detectors.
        triggerlist_kw: dict of kwargs to pass TriggerList.from_json()
        tc_range: range of times around tcoarse that should be searched
        ref_det_name: 'H'/'L'/'V' indicating detector with highest SNR
        calpha: calpha coefficients for bank template that triggered
        **par_dic_0: physical parameters `m1`, `m2`, `s1z`, `s2z` (and
          others if the bank has a precessing and/or HM waveform generator)
          for best fit waveform -- note that if this is not passed then
          upon calling self.get_event_data() it will be set using
          bank.get_pdic_from_calpha(self.calpha), and if calpha is None
          then it will not be set
        """
        # These are set by `self._setup()`:
        self.fnames = NotImplemented
        self.detector_names = NotImplemented
        self.load_data = NotImplemented
        self.triggerlists = None
        self.triggerlist_kw = ({} if triggerlist_kw is None
                               else deepcopy(triggerlist_kw))
        # These are passed
        self._setup_pars = {'bank_id': bank_id,
                            'fnames': fnames,
                            'load_data': load_data}
        self.eventname = eventname
        self.tgps = tgps
        self.t_interval = t_interval
        self.tcoarse = t_interval / 2 if tcoarse is None else tcoarse
        # Now specify (and compute) physical waveform properties
        self.par_dic_0 = deepcopy(par_dic_0)
        self.update_guess(
            mchirp_range=mchirp_range, q_min=q_min, tc_range=tc_range,
            ref_det_name=ref_det_name, calpha=calpha)
        self.compute_from_guess(
            compute_par_dic_0=compute_par_dic_0,
            compute_linear_free_shift=compute_linear_free_shift,
            max_tsep=max_tsep)
        # update event registry and save previous instance if it exists
        self.old_metadata = event_registry.get(self.eventname, None)
        if self.old_metadata is not None:
            print(f"Updating metadata for {self.eventname}, see",
                  f"event_registry[`{self.eventname}`].old_metadata",
                  "for old version.")
        event_registry[self.eventname] = self

    @classmethod
    def from_cand(cls, cand_dict, **init_kwargs):
        eventname = cand_dict.get(
            'evname', trig.utils.get_evname_from_tgps(cand_dict['tgps']))
        init_dict = {'eventname': eventname}
        init_dict.update({k: v for k, v in cand_dict.items() if k in [
            'tgps', 'mchirp_range', 'tc_range', 'calpha', 'max_tsep',
            'compute_linear_free_shift', 'bank_id', 'fnames', 'q_min',
            'ref_det_name', 'compute_par_dic_0']})
        init_dict.update(init_kwargs)
        if 'mchirp_range' not in init_dict:
            init_dict['compute_par_dic_0'] = True
            sigmas = init_dict.pop('sigmas', init_dict.pop('mchirp_sigmas', 5))
            snr = np.sqrt(np.min([v for k, v in cand_dict.items()
                                  if 'snr2' in k]))
        instance = cls(**init_dict)
        instance.load_triggerlists(store=True)
        if 'mchirp_range' not in init_dict:
            m1, m2 = instance.par_dic_0['m1'], instance.par_dic_0['m2']
            mc0 = (m1*m2)**.6/(m1+m2)**.2
            instance.mchirp_range = estimate_mchirp_range(
                mchirp=mc0, sigmas=sigmas, snr=snr)
        return instance

    def get_init_dict(self):
        """Return dictionary with keyword arguments to `__init__`."""
        # vv these must be taken from self (always set on init)
        init_dict = {key: getattr(self, key) for key in
                     ['eventname', 'tgps', 'mchirp_range',
                      'q_min', 't_interval', 'tcoarse']}
        # vv bankid, fnames, load_data come from self._setup_pars,
        # representing a snapshot of the initialization, so changes
        # made to these parameters post-init are not in JSON file
        init_dict.update(self._setup_pars)
        # vv these also must be taken from self (always set on init)
        init_dict.update({key: getattr(self, key, None) for key in
                          ['triggerlist_kw', 'tc_range']})
        # vv these will be taken from self if set, else set to None
        init_dict.update({key: getattr(self, key, None) for key in [
            'triggerlist_kw', 'tc_range', 'ref_det_name', 'calpha']})
        # par_dic_0 should just contain physical params, but since it
        # eats kwargs in many functions let us check that we have not
        # accidentally put another init kwarg in it during usage
        for k, v in self.par_dic_0.items():
            if k not in init_dict:
                init_dict[k] = v
            else:
                print(f'Warning! self.par_dic_0[`{k}`] = {v} lost'
                      f' in init_dict, using {k} = {init_dict[k]}')
        return init_dict

    def update_guess(self, mchirp_range=None, q_min=None, tc_range=None,
                     ref_det_name=None, calpha=None, **par_dic_0):
        # set attributes with inputs that are not None
        if mchirp_range is not None:
            self.mchirp_range = mchirp_range
        if q_min is not None:
            self.q_min = q_min
        if tc_range is not None:
            self.tc_range = tc_range
        if ref_det_name is not None:
            self.ref_det_name = ref_det_name
            self.i_refdet = (DETECTOR_NAMES_ORDERED
                             if self.detector_names is NotImplemented
                             else self.detector_names).index(ref_det_name)
        if calpha is not None:
            self.calpha = calpha
        self.par_dic_0.update(par_dic_0)

    def compute_from_guess(self, triggerlists=None, calpha=None,
                           ref_det_name=None, compute_par_dic_0=None,
                           compute_linear_free_shift=True, max_tsep=0.07,
                           **par_dic_0):
        if compute_par_dic_0 is None:
            compute_par_dic_0 = (not par_dic_0) and (calpha is not None)

        if not (compute_linear_free_shift or compute_par_dic_0):
            return

        # update guesses with whatever new stuff was passed
        self.update_guess(calpha=calpha, ref_det_name=ref_det_name,
                          **par_dic_0)
        # if no triggerlists passed, get them from self (load if None)
        if triggerlists is None:
            triggerlists = self.triggerlists or self.load_triggerlists()
        # computing linear free time shift
        if compute_linear_free_shift:
            self.tgps_shift = get_linear_free_time_shift(
                triggerlists, calpha=self.calpha, i_refdet=self.i_refdet,
                max_tsep=max_tsep, **self.par_dic_0)
        # computing physical parameters from calpha
        if compute_par_dic_0:
            self.par_dic_0 = triggerlists[
                self.i_refdet].templatebank.get_pdic_from_calpha(
                    self.calpha)

    def get_event_data(self, shift_tgps=False, max_tsep=0.07,
                       calpha=None, ref_det_name=None,
                       store_triggerlists=False, **par_dic_0):
        """
        Return an instance of `data.EventData`.

        Load a triggerlist corresponding to the event, use it to extract
        frequency domain strain and psd, and combine with the rest of
        the metadata.

        :param store_triggerlists: flag to set self.triggerlists
        :param shift_tgps: flag to shift tgps in event data from self.tgps
          to self.tgps + self.tgps_shift; if this flag is set to True,
          upon loading the triggerlists the shift will be computed
          from the linear free time shift of the bank template with
          geometric coefficients given in :param calpha:, or if
          :param calpha: is None will use self.calpha, or if no calpha
          then will use template bank approximant (usually IMRPhenomD)
          with physical parameters given in
          self.par_dic_0.update(:param par_dic_0:)
        :param max_tsep: if shifting tgps, a shift larger than this will
          result in error
        :param calpha: if not None, replace self.calpha
        :param ref_det_name: set new self.ref_det_name and self.i_refdet
        :param **par_dic_0: update self.par_dic_0
        """
        # Load triggerlists (note: this calls self._setup())
        triggerlists = self.load_triggerlists(store=store_triggerlists)
        use_tgps = self.tgps
        if shift_tgps:
            self.compute_from_guess(
                triggerlists=triggerlists, calpha=calpha,
                ref_det_name=ref_det_name, compute_linear_free_shift=True,
                max_tsep=max_tsep, compute_par_dic_0=False, **par_dic_0)
            use_tgps += self.tgps_shift

        dic = {key: getattr(self, key)
               for key in ['eventname', 'detector_names', 'tcoarse']}
        dic['tgps'] = use_tgps
        dic |= get_f_strain_psd_dic(triggerlists, use_tgps,
                                    self.tcoarse, self.t_interval)

        return data.EventData(**dic)

    def load_triggerlists(self, store=False):
        """
        Run self._setup() and return triggerlists associated to the event.
        :param store: flag to set self.triggerlists with output
        """
        self._setup()
        triggerlists = [trig.TriggerList.from_json(fname, load_data=load,
                                                   do_signal_processing=False,
                                                   **self.triggerlist_kw)
                        for fname, load in zip(self.fnames, self.load_data)]
        if store:
            self.triggerlists = triggerlists
        return triggerlists

    def _get_fnames(self, bank_id, fnames=None):
        """
        Auxiliary function to get json filenames of triggerlists
        implementing defaults.

        bank_id: 3-tuple with (source, i_multibank, i_subbank) where
                 `source` is 'BNS', 'NSBH' or 'BBH', `i_multibank` and
                 `i_subbank` are integers. Can be left as `None` to
                 guess a good choice from `mchirp_range`.
        fnames: 3-list with strings or `None`s or `0`s specifying
                the Hanford, Livingston and Virgo triggerlists.
                Strings are interpreted as filenames, `None` will ignore
                that detector, and `0` means that the pipeline's file
                will be retrieved. Passing a single `None` (default) is
                equivalent to [0, 0, 0].
        """
        if fnames is None:
            fnames = [0, 0, 0]

        i_get_default = np.where([fnm == 0 for fnm in fnames])[0]
        if len(i_get_default) > 0:
            if bank_id is None:
                bank_id = guess_bank_id(np.mean(self.mchirp_range))
            source, i_multibank, i_subbank = bank_id
            json_fnames = trig.utils.get_detector_fnames(
                self.tgps, i_multibank, i_subbank, source=source)
            for ind in i_get_default:
                fnames[ind] = json_fnames[ind]

        return fnames

    def _setup(self):
        """
        Set attributes `self.fnames`, `self.detector_names`,
        `self.load_data`. This will crash if the user does not have
        access to the IAS filesystem. This method is there so that the
        module can be imported without crashing.
        """
        if self.fnames is not NotImplemented:
            return

        fnames = self._get_fnames(self._setup_pars['bank_id'],
                                  self._setup_pars['fnames'])

        self.detector_names = ''.join(
            det for det, fname in zip(DETECTOR_NAMES_ORDERED, fnames)
            if fname is not None)
        # Reference detector information, if it exists
        refdet_name = getattr(self, 'ref_det_name', None)
        if refdet_name is not None:
            assert refdet_name in self.detector_names, \
                (f'Detector mismatch! self.ref_det_name={refdet_name}' +
                 f' is not in self.detector_names={self.detector_names}')
            self.i_refdet = self.detector_names.index(refdet_name)

        self.fnames = [fname for fname in fnames if fname is not None]

        load_data = self._setup_pars['load_data']
        if load_data in (True, False):
            load_data = [load_data] * len(self.fnames)
        self.load_data = load_data

    def __repr__(self):
        return f'{self.__class__.__name__}({self.eventname})'


# ----------------------------------------------------------------------
# Bookkeeping
# Note, fname=None means "use pipeline's triggerlist if available".

EventMetadata('GW151216', 1134293073.165, (10, 40), bank_id=('BBH', 3, 2))
EventMetadata('GW170121', 1169069154.565, (10, 50), bank_id=('BBH', 3, 0))
EventMetadata('GW170202', 1170079035.715, (10, 40), bank_id=('BBH', 3, 0))
EventMetadata('GW170304', 1172680691.356, (20, 80), bank_id=('BBH', 4, 0))
EventMetadata('GW170425', 1177134832.178, (20, 110), bank_id=('BBH', 4, 0))
EventMetadata('GW170727', 1185152688.019, (20, 80), bank_id=('BBH', 4, 0))
EventMetadata('GW170403', 1175295989.221, (20, 80), bank_id=('BBH', 4, 1))
EventMetadata('GWC170402', 1175205128.567, (10, 100), bank_id=('BBH', 3, 0))
EventMetadata('GW170817A', 1186974184.716, (25, 100), bank_id=('BBH', 4, 2),
              fnames=[
    '/data/bzackay/GW/H-H1_GWOSC_O2_4KHZ_R1-1186971648-4096_notched_config.json',
    0,
    0],
              load_data=(False, False, True), triggerlist_kw={'fmax': 512.})

EventMetadata('GW150914', 1126259462.4, (10, 50), bank_id=('BBH', 3, 0))
EventMetadata('GW151012', 1128678900.4, (15, 21), bank_id=('BBH', 2, 0))
EventMetadata('GW151226', 1135136350.6, (9.5, 10.5), bank_id=('BBH', 1, 0),
              fnames=(
    "/data/srolsen/GW/gw_pe/GW151226_reweighting_triggerlists/trigH1notch1024.json",
    "/data/srolsen/GW/gw_pe/GW151226_reweighting_triggerlists/trigL1notch1024.json"))
EventMetadata('GW170104', 1167559936.6, (10, 50), bank_id=('BBH', 3, 0))
EventMetadata('GW170608', 1180922494.5, (8.2, 9), bank_id=('BBH', 1, 0),
              fnames=[
    '/data/bzackay/GW/H-H1_GWOSC_O2_4KHZ_R1-1180920447-4096_notched_config.json',
    '/data/bzackay/GW/L-L1_GWOSC_O2_4KHZ_R1-1180920447-4096_config.json'])
EventMetadata('GW170729', 1185389807.3, (20, 80), bank_id=('BBH', 4, 0),
              fnames=[
    0,
    0,
    '/data/bzackay/GW/V-V1_GWOSC_4KHZ_R1-1185387760-4096_holefilled.json'])
EventMetadata('GW170809', 1186302519.8, (10, 50), bank_id=('BBH', 3, 0),
              load_data=(False, False, True), triggerlist_kw={'fmax': 512.})
EventMetadata('GW170814', 1186741861.5, (10, 50), bank_id=('BBH', 3, 0),
              load_data=(False, False, True), triggerlist_kw={'fmax': 512.})
EventMetadata('GW170818', 1187058327.1, (10, 50), bank_id=('BBH', 3, 0))
EventMetadata('GW170823', 1187529256.5, (10, 50),
              fnames=[
    '/data/bzackay/GW/OutputDir/O2_Fri_Mar_8_12_29O2_BBH_3_multibank_bank_0/H-H1_GWOSC_O2_4KHZ_R1-1187528704-4096_config.json',
    '/data/bzackay/GW/OutputDir/O2_Fri_Mar_8_12_29O2_BBH_3_multibank_bank_0/L-L1_GWOSC_O2_4KHZ_R1-1187528704-4096_config.json']
             )  # There's a Virgo file but no Virgo data at tgps

EventMetadata('GW190412', 1239082262.2, (13.9, 16.8))
EventMetadata('GW190814', 1249852257.0, (6.25, 6.57),
              fnames=[
    '/data/bzackay/GW/O3Events/GW190814/H-H1_GWOSC_4KHZ_R1-1249850209-4096.json',
    '/data/bzackay/GW/O3Events/GW190814/H-L1_GWOSC_4KHZ_R1-1249850209-4096.json',
    '/data/bzackay/GW/O3Events/GW190814/H-V1_GWOSC_4KHZ_R1-1249850209-4096.json'])
EventMetadata('GW190521', 1242442967.4, (25, 155))
EventMetadata('GW190408_181802', 1238782700.3, (14.1, 32.9))
EventMetadata('GW190413_052954', 1239168612.5, (7.91, 84.6))
EventMetadata('GW190413_134308', 1239198206.7, (9.57, 109))
EventMetadata('GW190421_213856', 1239917954.3, (11.2, 86),
              fnames=[
    '/data/bzackay/GW/O3Events/GW190421_213856/H-H1_GWOSC_4KHZ_R1-1239915907-4096.json',
    '/data/bzackay/GW/O3Events/GW190421_213856/L-L1_GWOSC_4KHZ_R1-1239915907-4096.json'])
EventMetadata('GW190424_180648', 1240164426.1, (12.6, 77.9),
              fnames=[
    None,  # H, V were off
    '/data/bzackay/GW/O3Events/GW190424_180648/L-L1_GWOSC_4KHZ_R1-1240162379-4096.json'])
# EventMetadata('GW190426_152155', 1240327333.3, (2.53, 2.67))  # BNS
EventMetadata('GW190503_185404', 1240944862.3, (9.36, 72.2))
EventMetadata('GW190512_180714', 1241719652.4, (13.7, 23.9))
EventMetadata('GW190513_205428', 1241816086.8, (10.8, 56.1))
EventMetadata('GW190514_065416', 1241852074.8, (8.11, 101))
EventMetadata('GW190517_055101', 1242107479.8, (15, 58))
EventMetadata('GW190519_153544', 1242315362.4, (13.8, 114))
EventMetadata('GW190521_074359', 1242459857.5, (24.9, 53.5))
EventMetadata('GW190527_092055', 1242984073.8, (4.25, 156),
              fnames=[
    '/data/bzackay/GW/O3Events/GW190527_092055/H-H1_GWOSC_4KHZ_R1-1242982026-4096.json',
    '/data/bzackay/GW/O3Events/GW190527_092055/L-L1_GWOSC_4KHZ_R1-1242982026-4096.json'])
EventMetadata('GW190602_175927', 1243533585.1, (8.43, 142))
EventMetadata('GW190620_030421', 1245035079.3, (8.51, 113))
EventMetadata('GW190630_185205', 1245955943.2, (20.5, 38.6))
EventMetadata('GW190701_203306', 1246048404.6, (11.4, 101))
EventMetadata('GW190706_222641', 1246487219.3, (7.45, 149))
EventMetadata('GW190707_093326', 1246527224.2, (9.38, 10.5))
EventMetadata('GW190708_232457', 1246663515.4, (13.9, 17.2))
EventMetadata('GW190719_215514', 1247608532.9, (7.85, 84.9))
EventMetadata('GW190720_000836', 1247616534.7, (9.58, 11.4))
EventMetadata('GW190727_060333', 1248242632.0, (11.8, 78.4))
EventMetadata('GW190728_064510', 1248331528.5, (9.66, 10.7))
EventMetadata('GW190731_140936', 1248617394.6, (9.16, 89.5))
EventMetadata('GW190803_022701', 1248834439.9, (9.86, 80.6))
EventMetadata('GW190828_063405', 1251009263.8, (18.8, 50.4))
EventMetadata('GW190828_065509', 1251010527.9, (13.7, 21.1))
EventMetadata('GW190909_114149', 1252064527.7, (3.8, 199))
EventMetadata('GW190910_112807', 1252150105.3, (20.8, 68.8),
              fnames=[
    None,
    '/data/bzackay/GW/O3Events/GW190910_112807/L-L1_GWOSC_4KHZ_R1-1252148058-4096.json',
    '/data/bzackay/GW/O3Events/GW190910_112807/V-V1_GWOSC_4KHZ_R1-1252148058-4096.json'])
EventMetadata('GW190915_235702', 1252627040.7, (12.8, 53))
EventMetadata('GW190924_021846', 1253326744.8, (6.22, 6.72))
EventMetadata('GW190929_012149', 1253755327.5, (5.24, 143))
EventMetadata('GW190930_133541', 1253885759.2, (8.8, 11.2))

# IAS O1 / O2 subthreshold
# EventMetadata('1172487817', 1172487817.477, (20, 100), bank_id=('BBH', 4, 1))
# EventMetadata('1170914187', 1170914187.455, (10, 50), bank_id=('BBH', 3, 0))
# EventMetadata('1172449151', 1172449151.468, (10, 80), bank_id=('BBH', 3, 1))
# EventMetadata('1174138338', 1174138338.385, (20, 80), bank_id=('BBH', 4, 0))
# EventMetadata('1171863216', 1171863216.108, (10, 50), bank_id=('BBH', 3, 0))
# EventMetadata('1182674889', 1182674889.044, (10, 170), bank_id=('BBH', 3, 0))
# EventMetadata('1171410777', 1171410777.2, (10, 170), bank_id=('BBH', 3, 1))
# EventMetadata('1187176593', 1187176593.222, (10, 50), bank_id=('BBH', 3, 1))
# EventMetadata('1175991666', 1175991666.075, (4.5, 4.7), bank_id=('BBH', 0, 0))
