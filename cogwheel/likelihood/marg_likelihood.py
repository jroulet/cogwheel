"""Compute likelihood of GW events."""
import sys
import inspect
import itertools
from functools import wraps
import numpy as np
from scipy import special, stats
from scipy.optimize import differential_evolution, minimize_scalar
import scipy.interpolate
import matplotlib.pyplot as plt

#from cogwheel import utils
#from cogwheel import gw_utils
from cogwheel import waveform
#from cogwheel import skyloc_angles
#import gw_prior.extrinsic.UniformTimePrior
#from skyloc_angles import SkyLocAngles
#import waveform.out_of_bounds as out_of_bounds
#import waveform.APPROXIMANTS as APPROXIMANTS
from . import RelativeBinningLikelihood

sys.path.append('/data/tislam/works/KITP/repos/fast_pe_using_cs/')
#DEFAULT_CS_FNAME = '/data/tislam/works/KITP/final_repo/fast_pe_using_cs/example_ra_dec_time_grid/dense_RA_dec_grid_H1_L1_32768_1136574828.npz' 
DEFAULT_CS_FNAME = '/data/tislam/works/KITP/final_repo/fast_pe_using_cs/example_ra_dec_time_grid/RA_dec_grid_H1_L1_4096_1136574828.npz'
import coherent_score_mz_fast_tousif as cs
import lal

class MarginalizedRelativeBinningLikelihood(RelativeBinningLikelihood):
    """
    Generalization of 'RelativeBinningLikelihood' that implements computation of 
    marginalized likelihood (over distance and phase) with the relative binning method.
    
    It integrates cogwheel and coherent_score code.
    Intrinsic parameters are sampled using cogwheel.
    Extrinsic parameters (ra,dec,mu,psi) are sampled using coherent score.
    Distance and Phase is sampled using analytical distribution funcitons.
    
    dist_ref is set to 1Mpc.
    """
    def __init__(self, event_data, waveform_generator, par_dic_0,
                 fbin=None, pn_phase_tol=None,# tolerance_params=None,
                 spline_degree=3, cs_fname=DEFAULT_CS_FNAME, t_rng=(-.06, .06), dist_ref=1): #reference distance dist_ref
        
        self.t_rng = t_rng
        self.dist_ref = dist_ref
        
        self.cs_fname = cs_fname
        self.cs_obj = cs.CoherentScoreMZ(samples_fname=cs_fname, run='03a')
        self.dt = self.cs_obj.dt_sinc/1000   # from milisecond to seconds
        self.timeshifts = np.arange(*t_rng, self.dt)
        
        self.ref_pardict = {'d_luminosity': self.dist_ref, 'iota': 0.0, 'phi_ref': 0.0}
        
        super().__init__(event_data, waveform_generator, par_dic_0, fbin,
                         pn_phase_tol, spline_degree)
        
    @property
    def params(self):
        
        return sorted(set(self.waveform_generator._waveform_params)-self.ref_pardict.keys())
        
    def _set_summary(self):
        """
        Compute summary data for the fiducial waveform at all detectors.
        `asd_drift` is not applied to the summary data, to not have to
        keep track of it.
        Update `asd_drift` using the reference waveform.
        The summary data `self._d_h_weights` and `self._d_h_weights` are
        such that:
            (d|h) ~= sum(_d_h_weights * conj(h_fbin)) / asd_drift^2
            (h|h) ~= sum(_h_h_weights * abs(h_fbin)^2) / asd_drift^2

        Assuming no higher modes.
        """
        h0_f = self._get_h_f(self.par_dic_0, by_m=False)
        h0_fbin = self.waveform_generator.get_strain_at_detectors(
            self.fbin, self.par_dic_0, by_m=False)  # ndet x len(fbin)
 

        d_h0 = (self.event_data.blued_strain
                * np.conj(h0_f) * np.exp(2j * np.pi * self.event_data.frequencies
                                 * self.timeshifts[:, np.newaxis, np.newaxis]
                                 ))  #  (ntimes, ndet, nfft)
        self._d_h_weights = self._get_summary_weights(d_h0) / np.conj(h0_fbin)
        # add a time shift to have the peak at t=0
        self._d_h_weights *= np.exp(1j*2*np.pi*(self.waveform_generator.tcoarse + self._par_dic_0['t_geocenter']) * self.fbin) #use t_gps

        h0_h0 = h0_f * h0_f.conj() * self.event_data.wht_filter**2
        self._h_h_weights = (self._get_summary_weights(h0_h0)
                             / (h0_fbin * h0_fbin.conj()))  # (ndet, nbin)

        self.asd_drift = self.compute_asd_drift(self.par_dic_0)
        self._lnl_0 = self.lnlike(self.par_dic_0, bypass_tests=True)

        self.timestamps = NotImplemented  # TODO

    def get_z_timeseries(self, par_dic):
        """
        Return (d|h)/sqrt(h|h) timeseries, with asd_drift correction
        applied.
        """
        h_fbin, _ = self.waveform_generator.get_hplus_hcross(self.fbin, par_dic|self.ref_pardict)
        
        # Sum over f axis, leave det and time axes unsummed.
        d_h = (self._d_h_weights * h_fbin.conj()).sum(axis=-1)
        h_h = (self._h_h_weights * h_fbin * h_fbin.conj()).real.sum(axis=-1)
        norm_h = np.sqrt(h_h)
        
        return d_h / norm_h / self.asd_drift, norm_h

    def lnlike(self, par_dic, **_kwargs):
        """
        Return likelihood marginalized over distance and phase
        i.e. return coherent score
        """
        # 1) get z timeseries: (time x det)
        z_timeseries, norm_h = self.get_z_timeseries(par_dic)

        # slice data segement that contains the event
#         fixed_pars = _kwargs.get("fixed_pars", None)
#         fixed_vals = _kwargs.get("fixed_vals", None)
#         fixing = fixed_pars is not None

#         t_indices = np.zeros(len(z_timeseries[0]), dtype=np.int32)
#         for ind_det in range(len(t_indices)):
#             tnstr = 't' + str(ind_det)
#             if fixing and (tnstr in fixed_pars):
#                 tnind = fixed_pars.index(tnstr)
#                 tn = fixed_vals[tnind]
#                 t_indices[ind_det] = np.searchsorted(self.timeshifts, tn)
#             else:
#                 t_indices[ind_det] = np.argmax(
#                     np.real(z_timeseries[:, ind_det])**2 +
#                     np.imag(z_timeseries[:, ind_det])**2)
        t_indices = np.argmax( np.real(z_timeseries)**2 + np.imag(z_timeseries)**2, axis=0 )
        
        # create a pc list
        event_phys = np.zeros((len(self.event_data.detector_names), 7))
        event_phys[:,0] = self.timeshifts[t_indices] # time
        event_phys[:,1] = [np.abs(z_timeseries[tind, i])**2 for i,tind in enumerate(t_indices)]
        event_phys[:,2] = norm_h #norm_fac
        event_phys[:,3] = 1 #hole_correction
        event_phys[:,4] = self.asd_drift 
        event_phys[:,5] = [np.real(z_timeseries[tind, i]) for i,tind in enumerate(t_indices)]
        event_phys[:,6] = [np.imag(z_timeseries[tind, i]) for i,tind in enumerate(t_indices)]
        
        z_timeseries_cs = np.zeros((len(self.event_data.detector_names),len(self.timeshifts),3))
        z_timeseries_cs[:,:,0] = self.timeshifts[:]
        z_timeseries_cs[:,:,1] = np.transpose(z_timeseries.real)
        z_timeseries_cs[:,:,2] = np.transpose(z_timeseries.imag)
              
        # 2) call the coherent score(z(t))
#         prior_terms, *_ = self.cs_obj.get_all_prior_terms_with_samp(
#                 event_phys, timeseries=z_timeseries_cs, nsamples=10000,
#                 fixed_pars=fixed_pars, fixed_vals=fixed_vals)
        prior_terms, *_ = self.cs_obj.get_all_prior_terms_with_samp(
                event_phys, timeseries=z_timeseries_cs, nsamples=10000)
        
        # coherent score is the marginalized likelihood
        coherent_score = prior_terms[0][0]
    
        return coherent_score/2 # devide by two
        #, mu, psi, ra, dec, U, T2

    def obtain_sky_loc_params_from_cs(self, par_dic, **_kwargs):
        """
        given a set of intrinsic params (Mc,q,chieff, cumchidif)
        this function calls to coherent score code to compute coherent score
        and picks a particular value for [mu,psi,ra,dec,U,T2]
        TODO: Use fixed values inside
        """
        # 1) get z timeseries: (time x det)
        z_timeseries, norm_h = self.get_z_timeseries(par_dic)

        # slice data segement that contains the event
        fixed_pars = _kwargs.get("fixed_pars", None)
        fixed_vals = _kwargs.get("fixed_vals", None)
        fixing = fixed_pars is not None

        t_indices = np.zeros(len(z_timeseries[0]), dtype=np.int32)
        for ind_det in range(len(t_indices)):
            tnstr = 't' + str(ind_det)
            if fixing and (tnstr in fixed_pars):
                tnind = fixed_pars.index(tnstr)
                tn = fixed_vals[tnind]
                t_indices[ind_det] = np.searchsorted(self.timeshifts, tn)
            else:
                t_indices[ind_det] = np.argmax(
                    np.real(z_timeseries[:, ind_det])**2 +
                    np.imag(z_timeseries[:, ind_det])**2)
        # t_indices = np.argmax(np.real(z_timeseries)**2 + np.imag(z_timeseries)**2, axis=0)
        
        # create a pc list
        event_phys = np.zeros((len(self.event_data.detector_names), 7))
        event_phys[:,0] = self.timeshifts[t_indices] # time
        event_phys[:,1] = [np.abs(z_timeseries[tind, i])**2 for i,tind in enumerate(t_indices)]
        event_phys[:,2] = norm_h #norm_fac
        event_phys[:,3] = 1 #hole_correction
        event_phys[:,4] = self.asd_drift 
        event_phys[:,5] = [np.real(z_timeseries[tind, i]) for i,tind in enumerate(t_indices)]
        event_phys[:,6] = [np.imag(z_timeseries[tind, i]) for i,tind in enumerate(t_indices)]
        
        z_timeseries_cs = np.zeros((len(self.event_data.detector_names),len(self.timeshifts),3))
        z_timeseries_cs[:,:,0] = self.timeshifts[:]
        z_timeseries_cs[:,:,1] = np.transpose(z_timeseries.real)
        z_timeseries_cs[:,:,2] = np.transpose(z_timeseries.imag)
              
        # 2) call the coherent score(z(t))
        prior_terms, _, _, samples_all, UT2samples = \
            self.cs_obj.get_all_prior_terms_with_samp(
                event_phys, timeseries=z_timeseries_cs,
                nra=100, ndec=100, nsamples=10000,
                fixed_pars=fixed_pars, fixed_vals=fixed_vals)
        
        # choose one point for mu,psi,ra,dec from the distributions
        cweights = np.cumsum(samples_all[0][:, 4])
        cweights /= cweights[-1]
        indx_rand = cs.rand_choice_nb(np.arange(len(samples_all[0][:, 4])), cweights, 1)
        mu = samples_all[0][:, 0][indx_rand]
        psi = samples_all[0][:, 1][indx_rand]
        dec_all = self.cs_obj.dec_grid[samples_all[0][:,3].astype(int)]
        dec = dec_all[indx_rand]
        dra = (lal.GreenwichMeanSiderealTime(self.event_data.tgps) - lal.GreenwichMeanSiderealTime(1136574828.0))
        ra_all = (self.cs_obj.ra_grid[samples_all[0][:, 2].astype(int)] + dra) % (2*np.pi)
        ra = ra_all[indx_rand]
        
        H_time = samples_all[0][:, 5][indx_rand]
        
        globals()["dic"] = locals()
        
        # get corresponing value for U and T2
        U, T2 = UT2samples[:, indx_rand]
        T2 = T2.real
        coherent_score = prior_terms[0][0]
        
        return coherent_score/2, mu[0], psi[0], ra[0], dec[0], U[0], T2[0], UT2samples, H_time
   