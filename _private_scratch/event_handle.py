"""
Get samples, likelihood, and prior objects from parameter estimation run
after post-processing for data analysis and visualization.
"""
import numpy as np
import os
import pathlib
import sys
import json
import pandas as pd
import datetime
from copy import deepcopy as dcopy
from gwpy.time import tconvert

from . import parameter_label_formatting as label_formatting
from . import standard_intrinsic_transformations as pxform
from . import pe_plotting as peplot
from . import analysis_handle as ahand
DEFAULT_PRIOR, DEFAULT_PARENTDIR = ahand.DEFAULT_PRIOR, ahand.DEFAULT_PARENTDIR
key_rngs_mask, AnalysisHandle = ahand.key_rngs_mask, ahand.AnalysisHandle

COGWHEEL_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'cogwheel'))
sys.path.append(COGWHEEL_PATH)
from cogwheel import _private_ias
trig = _private_ias.trig

# back when evnames were 8 chars
SPECIAL_EVNAMES = {'GW150914': {'tgps': 1126259462.4},
                   'GW151012': {'tgps': 1128678900.4},
                   'GW151216': {'tgps': 1134293073.165},
                   'GW151226': {'tgps': 1135136350.6},
                   'GW170104': {'tgps': 1167559936.6},
                   'GW170121': {'tgps': 1169069154.565},
                   'GW170202': {'tgps': 1170079035.715},
                   'GW170304': {'tgps': 1172680691.356},
                   'GW170403': {'tgps': 1175295989.221},
                   'GW170425': {'tgps': 1177134832.178},
                   'GW170608': {'tgps': 1180922494.5},
                   'GW170727': {'tgps': 1185152688.019},
                   'GW170729': {'tgps': 1185389807.3},
                   'GW170809': {'tgps': 1186302519.8},
                   'GW170814': {'tgps': 1186741861.5},
                   'GW170818': {'tgps': 1187058327.1},
                   'GW170823': {'tgps': 1187529256.5},
                   'GW190412': {'tgps': 1239082262.2},
                   'GW190521': {'tgps': 1242442967.4},
                   'GW190814': {'tgps': 1249852257.0}}

def get_run(tgps_or_evname):
    return peplot.run_from_evname(str(tgps_or_evname))

def evname_from_tgps(tgps_sec, prefix='GW'):
    if hasattr(tgps_sec, '__len__'):
        return [evname_from_tgps(tt, prefix) for tt in tgps_sec]
    return trig.utils.get_evname_from_tgps(tgps_sec, prefix)

def tgps_from_evname(evn, events_dict=SPECIAL_EVNAMES):
    tgpsout = events_dict.get(evn, {}).get('tgps', None)
    if tgpsout is None:
        evn = evn.replace('GWC', 'GW')
        assert (len(evn) == 15) and (evn[:2]+evn[8:9] == 'GW_'), \
            'evn must have form GWyymmdd_hhmmss'
        tgpsout = tconvert(datetime.datetime(2000+int(evn[2:4]),
                    int(evn[4:6]), int(evn[6:8]), int(evn[9:11]),
                    int(evn[11:13]), int(evn[13:]))).gpsSeconds
    return tgpsout


class EventName(str):
    """Class for GW event names of form GWyymmdd<_hhmmss>"""
    def __init__(self, evname, events_dict=SPECIAL_EVNAMES):
        super().__init__()
        # this sets self._tgps, self.left, self.right, and self.run
        self.tgps = tgps_from_evname(evname, events_dict=events_dict)

    @property
    def tgps(self):
        """GPS time in seconds"""
        return self._tgps

    @tgps.setter
    def tgps(self, value):
        """Set GPS time, lef and right evnames, and data run"""
        self._tgps = float(value)
        self.left = evname_from_tgps(self._tgps - 1)
        self.right = evname_from_tgps(self._tgps + 1)
        self.run = get_run(self._tgps)

    @classmethod
    def from_tgps(cls, tgps, recompute_tgps=False):
        if (not isinstance(tgps, str)) and hasattr(tgps, '__len__'):
            return [cls.from_tgps(t0, recompute_tgps) for t0 in tgps]
        instance = cls(evname_from_tgps(float(tgps)))
        if not recompute_tgps:
            instance.tgps = tgps
        return instance

    def same(self, compare, pad=True):
        if pad:
            return (compare in [self, self.left, self.right])
        return (self == compare)

    def near(self, compare, tpad=1.5, events_dict=SPECIAL_EVNAMES):
        if isinstance(compare, str):
            if compare.isnumeric():
                tgps = float(compare)
            else:
                tgps = tgps_from_evname(compare, events_dict=events_dict)
        elif isinstance(compare, EventName):
            tgps = compare.tgps
        else:
            tgps = float(compare)
        return (np.abs(self.tgps - tgps) < tpad)


class EventHandle(ahand.AnalysisHandle):
    """Adding triggerlist stuff to AnalysisHandle"""

    @classmethod
    def from_cand(cls, cand_dict, i_run=0, parentdir=ahand.DEFAULT_PARENTDIR,
                  prior_name=ahand.DEFAULT_PRIOR, **init_kwargs):
        instance = cls.from_evname(EventName.from_tgps(cand_dict['tgps']),
                                   i_run=i_run, parentdir=parentdir,
                                   prior_name=prior_name, **init_kwargs)
        instance.set_candidate(cand_dict, det_names=instance.evdata.detector_names)
        return instance

    def eventname(self):
        evn = EventName(self.evname)
        evn.tgps = self.evdata.tgps
        return evn

    def set_candidate(self, cand_dict, set_triggerlists=True,
                      det_names='HL'):
        # trigger information
        self.cdic = dcopy(cand_dict)
        self.trigger = self.cdic.get('trigger')
        self.calpha = self.cdic.get('calpha')
        self.bank_id = (self.cdic.get('i_multibank'),
                        self.cdic.get('i_subbank'))
        # config files for triggerlists
        self.fnames = self.cdic.get('fnames')
        if self.fnames is None:
            self.fnames = trig.utils.get_detector_fnames(
                self.cdic.get('tgps', self.evdata.tgps),
                *self.bank_id, n_multibanks=6)
        # loading triggerlists
        if set_triggerlists:
            self.triggerlists = self.load_triggerlists()
        # detector-specific trigger information
        self.cand_det_names = ''.join([d for d, fnm in
                                       zip(det_names, self.fnames)
                                       if fnm is not None])
        self.snr2 = np.array([self.cdic.get(f'snr2_{d}')
                              for d in self.cand_det_names])

    def load_triggerlists(self, fnames=None):
        if fnames is None:
            fnames = self.fnames
        return [trig.TriggerList.from_json(fnm) for fnm in fnames
                if fnm is not None]

    def match(self, pdic1_or_wf1_wtd_td, pdic2_or_wf2_wtd_td=None,
              allow_shift=True, allow_phase=True, return_cov=False,
              det_inds=None):
        """
        Get match (full time series if return_cov=True) between waveforms
        specified by numpy arrays with the whitened time domain waveforms
        OR by parameter dicts from which the waveforms will be generated
        --> arguments allow_shift, allow_phase are passed to utils.match()
        """
        # get whitened TD waveforms to match
        if (isinstance(pdic1_or_wf1_wtd_td, np.ndarray)
                or (isinstance(pdic1_or_wf1_wtd_td, (list, tuple))
                    and (len(pdic1_or_wf1_wtd_td) < 4))):
            wf1 = pdic1_or_wf1_wtd_td
        else:
            wf1 = self.get_h_t(pdic1_or_wf1_wtd_td, whiten=True)

        if (isinstance(pdic2_or_wf2_wtd_td, np.ndarray)
                or (isinstance(pdic2_or_wf2_wtd_td, (list, tuple))
                    and (len(pdic2_or_wf2_wtd_td) < 4))):
            wf2 = pdic2_or_wf2_wtd_td
        else:
            wf2 = self.get_h_t(pdic2_or_wf2_wtd_td, whiten=True)
        # broadcast so that number of detectors is the same
        if np.ndim(wf1) == 1:
            # if only one detector, just return match there
            if np.ndim(wf2) == 1:
                return trig.utils.match(wf1, wf2, allow_shift=allow_shift,
                                        allow_phase=allow_phase, return_cov=return_cov)
            wf1 = [wf1] * len(wf2)
        elif np.ndim(wf2) == 1:
            wf2 = [wf2] * len(wf1)
        # check that number of detectors is the same
        ndet = len(wf1)
        assert len(wf2) == ndet, 'Mismatch between number of detectors for match input!'
        if ndet == 1:
            det_inds = 0
        elif det_inds is None:
            det_inds = range(ndet)
        if not hasattr(det_inds, '__len__'):
            # if only one detector, just return match there
            return trig.utils.match(wf1[det_inds], wf2[det_inds], allow_shift=allow_shift,
                                    allow_phase=allow_phase, return_cov=return_cov)
        return np.array([trig.utils.match(wf1[j], wf2[j],
                                          allow_shift=allow_shift, allow_phase=allow_phase,
                                          return_cov=return_cov) for j in det_inds])

    def calpha_wf_wtd_td(self, triggerlists=None, calpha=None, bank_grid=False, det_inds=None):
        calpha = calpha or self.calpha
        triggerlists = triggerlists or self.triggerlists
        if bank_grid:
            return [tgl.templatebank.gen_whitened_wfs_td(calpha=calpha)
                    for tgl in triggerlists]
        trig_wfs = np.zeros((len(triggerlists), len(self.evdata.frequencies)),
                            dtype=np.complex128)
        for jj, tgl in enumerate(triggerlists):
            trig_wfs[jj][self.evdata.fslice] = \
                tgl.templatebank.gen_wfs_fd_from_calpha(calpha=calpha,
                                                        fs_out=self.evdata.frequencies[self.evdata.fslice])
        if det_inds is None:
            det_inds = range(min(len(trig_wfs), len(self.evdata.wht_filter)))
        elif not hasattr(det_inds, '__len__'):
            det_inds = [det_inds]
        return np.sqrt(2 * self.evdata.nfft * self.evdata.df) * np.fft.irfft(
            np.array([wffd * self.evdata.wht_filter[jj] for jj, wffd in zip(det_inds, trig_wfs)]))

    def IMRPhenomD_wf_wtd_td(self, triggerlists=None, calpha=None, bank_grid=False, det_inds=None,
                             linear_free=True, **pars):
        triggerlists = triggerlists or self.triggerlists
        if not np.all([k in pars for k in ['m1', 'm2', 's1z', 's2z']]):
            calpha = calpha or self.calpha
            pars = triggerlists[0].templatebank.get_pdic_from_calpha(calpha, m1rng=(1, 250), m2rng=(1, 200))

        if bank_grid:
            return [tgl.templatebank.gen_whitened_wf_td_from_pars(
                approximant='IMRPhenomD', gen_domain='fd', **pars) for tgl in triggerlists]

        trig_wfs = np.zeros((len(triggerlists), len(self.evdata.frequencies)),
                            dtype=np.complex128)
        for jj, tgl in enumerate(triggerlists):
            trig_wfs[jj][self.evdata.fslice] = tgl.templatebank.gen_wf_fd_from_pars(
                fs_out=self.evdata.frequencies[self.evdata.fslice], gen_domain='fd',
                linear_free=linear_free, approximant='IMRPhenomD', **pars)
        if det_inds is None:
            det_inds = range(min(len(trig_wfs), len(self.evdata.wht_filter)))
        elif not hasattr(det_inds, '__len__'):
            det_inds = [det_inds]
        return np.sqrt(2 * self.evdata.nfft * self.evdata.df) * np.fft.irfft(
            np.array([wffd * self.evdata.wht_filter[jj] for jj, wffd in zip(det_inds, trig_wfs)]))

    def trigger_cov(self, pdic1=None, calpha=None, det_inds=None,
                    bank_grid=False, use_approximant=False, allow_shift=True,
                    allow_phase=True, return_cov=False, linear_free=True):
        """MUST CALL self.set_candidate() BEFORE using this function."""
        if det_inds is None:
            tgls = self.triggerlists
        elif hasattr(det_inds, '__len__'):
            tgls = [self.triggerlists[j] for j in det_inds]
        else:
            tgls = [self.triggerlists[int(det_inds)]]
        # make sure we don't have more triggerlists than detectors
        if len(tgls) > len(self.evdata.wht_filter):
            tgls = [tgls[jj] for jj in [self.evdata.detector_names.index(d)
                                        for d in self.cand_det_names]]

        # get trigger parameters
        if calpha is None:
            calpha = self.calpha
        # should we include shift? _private_ias._get_linear_free_shift_from_bank(bank, calpha=None, **pars)
        #### GETTING NON-TRIGGER WAVEFORM (or trigger IMRPhenomD waveform if pdic1=`trig`)
        match1 = pdic1
        trigVtrig = (isinstance(pdic1, str) and ('trig' in pdic1))
        if trigVtrig:
            match1 = self.IMRPhenomD_wf_wtd_td(triggerlists=tgls, calpha=calpha,
                                               bank_grid=bank_grid, det_inds=det_inds, linear_free=linear_free)
        elif bank_grid:
            match1 = [tgl.templatebank.gen_whitened_wfs_td(
                wfs_fd=tgl.templatebank.gen_strain_fd_from_pars(
                    fs_out=tgl.templatebank.fs_fft, linear_free=linear_free,
                    det_name=(d if d[-1] == '1' else d + '1'),
                    tgps=self.cdic.get('tgps', self.evdata.tgps),
                    approximant=self.wfgen.approximant, f_ref=self.wfgen.f_ref,
                    f_min=self.evdata.fmin, xphm_modes=self.wfgen.harmonic_modes,
                    whiten=False, **self.get_par_dic(pdic1)))
                for d, tgl in zip(self.cand_det_names, tgls)]
        # note if match1 is still pdic1 after this, it means we want an XPHM waveform on the PE grid
        # and this will be generated by passing the parameter dict to match

        #### GETTING TRIGGER WAVEFORM
        if use_approximant and (not trigVtrig):
            trig_wfs = self.IMRPhenomD_wf_wtd_td(triggerlists=tgls, calpha=calpha,
                                                 bank_grid=bank_grid, det_inds=det_inds, linear_free=linear_free)
        else:
            trig_wfs = self.calpha_wf_wtd_td(triggerlists=tgls, calpha=calpha,
                                             bank_grid=bank_grid, det_inds=det_inds)

        return self.match(match1, pdic2_or_wf2_wtd_td=trig_wfs,
                          allow_shift=allow_shift, allow_phase=allow_phase,
                          return_cov=return_cov, det_inds=det_inds)

    def plot_cov(self, pdic1='max', pdic2='trig', allow_shift=True, allow_phase=True,
                 fig=None, ax=None, xlab=None, ylab=None, figsize=None, bank_grid=False,
                 det_inds=None, label_max=True, take_abs=False, cov_kwargs_trig={},
                 **plot_kwargs):
        """
        if pdic2 = 'trig' or contains this substring then will get cov from self.trigger_cov()
        use cov_kwargs={`use_approximant`: True} to generate trigger with IMRPhenomD instead of self.calpha
        NOTE: MUST call self.set_candidate(cand_dict) before using trigger option
        """
        if xlab is None:
            xlab = 'Time (s)'
        if ylab is None:
            ylab = r'$\langle h_1 | h_2 \rangle / \sqrt{\langle h_1 | h_1 \rangle \langle h_2 | h_2 \rangle}$'
        if isinstance(pdic2, str) and ('trig' in pdic2):
            dt = self.triggerlists[0].templatebank.dt
            covplot = self.trigger_cov(pdic1=pdic1, calpha=cov_kwargs_trig.pop('calpha', None),
                det_inds=det_inds, bank_grid=bank_grid, allow_shift=allow_shift,
                allow_phase=allow_phase, return_cov=True, **cov_kwargs_trig)
        else:
            dt = self.evdata.dt
            covplot = self.match(pdic1, pdic2, allow_shift=allow_shift,
                                 allow_phase=allow_phase, return_cov=True,
                                 det_inds=det_inds)
        if det_inds is None:
            det_inds = range(len(covplot))
        elif not hasattr(det_inds, '__len__'):
            det_inds = [det_inds]
        if ax is None:
            fig, ax = ahand.peplot.get_dets_figure(xlabel=xlab, ylabel=ylab, figsize=figsize,
                                                   detector_names=[self.evdata.detector_names[dind] for dind in
                                                                   det_inds])
        lab0 = plot_kwargs.get('label', '')
        zfunc = (np.abs if take_abs else np.real)
        for aa, yy in zip(ax, covplot):
            xx = np.arange(len(yy)) * dt
            yy = zfunc(yy)
            if label_max:
                imax = np.argmax(yy)
                plot_kwargs['label'] = ((lab0 + ', ' if lab0 else '') +
                                        f"max = {yy[imax]:.2f} at t = {xx[imax]:.4f}")
            aa.plot(np.arange(len(yy)) * dt, yy, **plot_kwargs)
        return fig, ax

