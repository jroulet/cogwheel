"""
Compute marginalized likelihood of GW events, over extrinsic parameters
(RA, DEC, time, inclination, phase, distance) for non-precessing quadrupolar
waveforms.

A class ``CoherentScore`` is defined, which contains a dictionary with
detector responses and sky locations indexed by lags, and this class has
functionality to accept timeseries of matched-filtering scores and compute
the marginalized likelihoods, as well as information that is useful to
sample from the full (unmarginalized) posterior.
"""
import numpy as np
from numba.typed import List
from numba import float64, complex128
from numba import njit, vectorize
from scipy.special import i0e, i1e
from cogwheel import utils, gw_utils, skyloc_angles
import scipy.signal

import lal
# create Detector dictionary
DETMAP = {'H1': lal.LALDetectorIndexLHODIFF,
          'H2': lal.LALDetectorIndexLHODIFF,
          'LHO': lal.LALDetectorIndexLHODIFF,
          'H': lal.LALDetectorIndexLHODIFF,
          'L1': lal.LALDetectorIndexLLODIFF,
          'LLO': lal.LALDetectorIndexLLODIFF,
          'L': lal.LALDetectorIndexLLODIFF,
          'G1': lal.LALDetectorIndexGEO600DIFF,
          'GEO': lal.LALDetectorIndexGEO600DIFF,
          'GEO600': lal.LALDetectorIndexGEO600DIFF,
          'V1': lal.LALDetectorIndexVIRGODIFF,
          'V': lal.LALDetectorIndexVIRGODIFF,
          'VIRGO': lal.LALDetectorIndexVIRGODIFF,
          'T1': lal.LALDetectorIndexTAMA300DIFF,
          'TAMA': lal.LALDetectorIndexTAMA300DIFF,
          'TAMA300': lal.LALDetectorIndexTAMA300DIFF,
          'K1': lal.LALDetectorIndexKAGRADIFF,
          'KAGRA': lal.LALDetectorIndexKAGRADIFF,
          'LCGT': lal.LALDetectorIndexKAGRADIFF,
          'I1': lal.LALDetectorIndexLIODIFF,
          'LIO': lal.LALDetectorIndexLIODIFF,
          'E1': lal.LALDetectorIndexE1DIFF,
          'E2': lal.LALDetectorIndexE2DIFF,
          'E3': lal.LALDetectorIndexE3DIFF}


# Useful default variables
# Loose upper bound to travel time between any detectors in ms
DEFAULT_DT_MAX = 62.5
# Spacing of samples in ms
DEFAULT_DT = (1000/4096)/8
# Least count of timeslides in ms
DEFAULT_TIMESLIDE_JUMP = 100

# hard-wired f1=1 and f2=1
# can generate other values with the lG function
LG_FAST_XP = np.asarray(
    [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 1.1, 1.2,
     1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2., 2.1, 2.2, 2.3, 2.4, 2.5,
     2.6, 2.7, 2.8, 2.9, 3., 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8,
     3.9, 4., 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5., 5.1,
     5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6., 6.1, 6.2, 6.3, 6.4,
     6.5, 6.6, 6.7, 6.8, 6.9, 7., 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7,
     7.8, 7.9, 8., 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 9.,
     9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9])
f1 = 1
f2 = 1
# LG_FAST_YP = lG(LG_FAST_XP, f1, f2)
LG_FAST_YP = np.asarray(
    [-0.1135959, -0.12088098, -0.1427278, -0.1791095,
     -0.22997564, -0.2952433, -0.37478329, -0.46840065,
     -0.57580762, -0.69658744, -0.83014698, -0.97565681,
     -1.13197891, -1.29758569, -1.4704808, -1.64814261,
     -1.82752466, -2.00515831, -2.1774003, -2.34083626,
     -2.49278514, -2.63177248, -2.75780503, -2.87233483,
     -2.97792671, -3.07776058, -3.17513488, -3.27308869,
     -3.37418153, -3.48041019, -3.59321753, -3.71354945,
     -3.84192885, -3.97852807, -4.12323147, -4.27568566,
     -4.43533834, -4.60146807, -4.77320804, -4.94956693,
     -5.12944979, -5.31168161, -5.49503558, -5.67826708,
     -5.86015327, -6.0395366, -6.2153689, -6.38675167,
     -6.55296781, -6.71350061, -6.86803769, -7.01645979,
     -7.15881685, -7.29529541, -7.42618219, -7.55182832,
     -7.67261759, -7.78894088, -7.9011773, -8.0096818,
     -8.11477823, -8.21675658, -8.3158732, -8.41235286,
     -8.5063919, -8.59816165, -8.68781196, -8.77547441,
     -8.86126519, -8.94528754, -9.0276338, -9.10838716,
     -9.18762299, -9.26541003, -9.34181128, -9.41688478,
     -9.4906842, -9.5632594, -9.63465681, -9.70491982,
     -9.77408908, -9.84220278, -9.90929682, -9.9754051,
     -10.04055961, -10.10479061, -10.1681268, -10.23059541,
     -10.29222228, -10.35303204, -10.41304811, -10.47229284,
     -10.53078756, -10.58855264, -10.64560757, -10.70197099,
     -10.75766078, -10.81269406, -10.86708726, -10.92085616])


# ############## Functions to create library of saved samples #################
def get_geocenter_delays(detector_names, lat, lon):
    """
    Return array of shape (n_detectors, ...) time delays from geocenter [s].
    Vectorized over lat, lon.
    """
    locations = np.array([gw_utils.DETECTORS[detector_name].location
                          for detector_name in detector_names])  # (ndet, 3)

    direction = skyloc_angles.latlon_to_cart3d(lat, lon)
    return -np.einsum('di,i...->d...', locations, direction) / lal.C_SI


def get_fplus_fcross_0(detector_names, lat, lon):
    """
    Return array with antenna response functions fplus, fcross with
    polarization psi=0.
    Vectorized over lat, lon. Return shape is (..., n_det, 2)
    where `...` is the shape of broadcasting (lat, lon).
    """
    responses = np.array([gw_utils.DETECTORS[detector_name].response
                          for detector_name in detector_names]
                        )  # (n_det, 3, 3)

    lat, lon = np.broadcast_arrays(lat, lon)
    coslon = np.cos(lon)
    sinlon = np.sin(lon)
    coslat = np.cos(lat)
    sinlat = np.sin(lat)

    x = np.array([sinlon, -coslon, np.zeros_like(sinlon)])  # (3, ...)
    dx = np.einsum('dij,j...->di...', responses, x)  # (n_det, 3, ...)

    y = np.array([-coslon * sinlat,
                  -sinlon * sinlat,
                  coslat])  # (3, ...)
    dy = np.einsum('dij,j...->di...', responses, y)
    
    fplus0 = (np.einsum('i...,di...->d...', x, dx)
              - np.einsum('i...,di...->d...', y, dy))
    fcross0 = (np.einsum('i...,di...->d...', x, dy)
               + np.einsum('i...,di...->d...', y, dx))

    return np.moveaxis([fplus0, fcross0], (0, 1), (-1, -2))

def delays(ra, dec, detectors, gps_time):
    """
    Computes delays in arrival times between detectors, referred to the first
    (beyond 2 delays for most, and 3 delays for all, they're overspecified,
    but that's OK)
    :param ra: Right ascension (radians)
    :param dec: Right ascension (radians)
    :param detectors:
        List of dectectors, specified as instances of lal.CachedDetectors
    :param gps_time: GPS time at which the delays are computed
    :return: 1D array of delays referred to the first detector (in s)
    """
    times = np.asarray([
        lal.TimeDelayFromEarthCenter(det.location, ra, dec, gps_time)
        for det in detectors])
    dt = times[1:] - times[0]
    return dt


def phase_lags(ra, dec, detectors, gps_time):
    """
    Computes phase lags between detectors, referred to the first
    (beyond 2 delays for most, and 3 delays for all, they're overspecified,
    but that's OK)
    :param ra: Right ascension (radians)
    :param dec: Right ascension (radians)
    :param detectors:
        List of dectectors, specified as instances of lal.CachedDetectors
    :param gps_time: GPS time at which the antenna responses are computed
    :return: 1D array of phases referred to the first detector
    """
    # Greenwich mean sidereal time corresponding to the given GPS time
    gmst = lal.GreenwichMeanSiderealTime(gps_time)

    # Detector responses for phi = zero
    fs = [lal.ComputeDetAMResponse(det.response, ra, dec, 0, gmst)
          for det in detectors]

    # Phases
    phis = np.asarray([np.arctan2(f[1], f[0]) for f in fs])
    # Phase differences w.r.t the first detector (H1 for H1, L1)
    dphis = (phis[1:] - phis[0]) % (2 * np.pi)
    return dphis


@njit
def dt2key(dt, dt_sinc=DEFAULT_DT, dt_max=DEFAULT_DT_MAX):
    """
    Jitable key for the dictionary from the time delays
    This is clunky but faster than the nice expression by O(1) when vectorized
    :param dt:
        (ndelays = ndetectors-1) x nsamp array with time delays in milliseconds
        (should be 2d)
    :param dt_sinc: Time resolution in milliseconds
    :param dt_max: Maximum time delay in milliseconds
    :return: n_samp array of keys (always 1D)
    """
    dt = np.asarray(dt)
    nbase = int(np.floor(dt_max / dt_sinc)) * 2
    exparr = nbase ** np.arange(dt.shape[0])
    ind_arr = np.zeros(dt.shape[1], dtype=np.int32)
    for i_det_pair in range(dt.shape[0]):
        for i_sample in range(dt.shape[1]):
            ind_arr[i_sample] += (
                int(np.rint(dt[i_det_pair, i_sample] / dt_sinc)) % nbase
                * exparr[i_det_pair])
    return ind_arr


def create_time_dict(
        nlon, nlat, detector_names, dt_sinc=DEFAULT_DT,
        dt_max=DEFAULT_DT_MAX):
    """
    Creates dictionary indexed by time, giving RA-Dec pairs for montecarlo
    integration
    :param nlon: number of longitude points in grid
    :param nlat: number of lattitude in grid
    :param detector_names: detector names
    :param dt_sinc:
        Size of the time binning used in ms; it must coincide with the time
        separation of samples in the overlap if this dictionary will be used
        to do the marginalization
    :param dt_max: Rough upper bound on the individual delays
    :return:
        0. Dictionary indexed by the dts key, returning n_sky x 2 array with
           indices into ras and decs for each allowed dt tuple
        1. List of ras
        2. List of decs
        3. n_ra x n_dec x n_detector x 2 array with responses
        4. n_ra x n_dec x (n_detector - 1) array with delta ts
        5. n_ra x n_dec x (n_detector - 1) array with delta phis
        6. n_ra x n_dec array with network sensitivity (sum_{det, pol} resp^2)
    """
    lon_grid = np.linspace(0, 2.0 * np.pi, nlon, endpoint=False)
    # Declination minus the poles
    sin_dec_grid = np.linspace(-1.0, 1.0, nlat + 1, endpoint=False)
    sin_dec_grid = sin_dec_grid[1:]
    lat_grid = np.arcsin(sin_dec_grid)

    # Greenwich mean sidereal time corresponding to the given GPS time
    gmst = 0.0
    
    lon_grid_2d, lat_grid_2d = np.meshgrid(lon_grid, lat_grid, indexing='ij') # n_ra x n_dec
    # returns n_det x n_ra x n_dec
    # later change ra_grid to lon_grid
    delays_from_geocen = get_geocenter_delays(detector_names, lon_grid_2d, lat_grid_2d)
    # Time delays in milliseconds :  n_ra x n_dec x (n_det-1)
    deltats = np.moveaxis(1000 * (delays_from_geocen[1:,...] - delays_from_geocen[0,...]), 0, -1)
    # Detector responses for psi = zero : n_ra x n_dec x n_det x 2
    responses = get_fplus_fcross_0(detector_names, lon_grid_2d, lat_grid_2d)
    # Phases : n_ra x n_dec x n_det
    phis = np.arctan2(responses[...,1], responses[...,0])
    # Phase differences w.r.t the first detector (e.g., H1 for H1, L1) : n_ra x n_dec x (n_det-1)
    dphases = (phis[...,1:] - phis[...,(0,)]) % (2 * np.pi)
    # Network responses
    rtot2s = responses[...,1]**2 + responses[...,0]**2
    
    # Make `contour maps' of delta t, with entries = index of ra, index of dec
    dt_dict = {}
    for i in range(len(lon_grid)):
        for j in range(len(lat_grid)):
            key = dt2key(deltats[i, j][:, None], dt_sinc=dt_sinc, dt_max=dt_max)[0]
            if key in dt_dict:
                dt_dict[key].append((i, j))
            else:
                dt_dict[key] = [(i, j)]

    # Make the elements numpy arrays to pick and save easily
    for key in dt_dict.keys():
        dt_dict[key] = np.asarray(dt_dict[key], dtype=np.int32)

    return dt_dict, lon_grid, lat_grid, responses, deltats, dphases, rtot2s

# ############################ Useful functions ###############################
@vectorize(nopython=True)
def offset_background(dt, time_slide_jump, dt_shift):
    """
    Finds the amount to shift the detectors' data streams by
    :param dt: Time delays (s)
    :param time_slide_jump: Least count of time slides (s)
    :param dt_shift: Time resolution (s)
    :return:
        Amount to add to detector to bring the timeseries to zero lag
        wrt the reference detector (opposite convention to coherent_score_mz.py)
    """
    # dts = t_det - t_h1, where each is evaluated at the peak of the respective
    # SNR^2 timeseries
    dt0 = dt % time_slide_jump
    dt1 = dt % time_slide_jump - time_slide_jump
    if abs(dt0) < abs(dt1):
        shift = dt0 - dt
    else:
        shift = dt1 - dt
    return round(shift / dt_shift) * dt_shift


def incoherent_score(triggers):
    """
    :param triggers:
        n_cand x n_detector x row of processedclists
        (can be a 2d array if n_cand = 1)
    :return: Vector of incoherent scores (scalar if n_cand = 1)
    """
    return np.sum(triggers[..., 1], axis=-1)


@vectorize([complex128(float64, float64, float64, float64, float64)], nopython=True)
def gen_sample_amps_from_fplus_fcross(fplus, fcross, mu, c2psi, s2psi):
    """
    :param fplus: Response to the plus polarization for psi = 0
    :param fcross: Response to the cross polarization for psi = 0
    :param mu: Inclination
    :param psi: Polarization angle
    :returns A_p + 1j * A_c
    ## Note that this seems to have the wrong convention for mu
    """
    fp = c2psi * fplus + s2psi * fcross
    fc = -s2psi * fplus + c2psi * fcross
    # Strain amplitude at the detector
    ap = fp * (1. + mu ** 2) / 2.
    ac = -fc * mu
    return ap + 1j * ac


@njit
def lgg(zthatthatz, gtype):
    """
    param: zthatthatz
    """
    ## Formula obtained by bringing the prior to the exponent, expanding it to
    # quadratic order and integrating, used for the marginal likelihood
    ## The power law terms come from the prior and are not very important in
    # practice. zthatthatz has size of SNR^2
    if gtype == 0:
        ## Turn off distance prior
        logg = np.zeros_like(zthatthatz)
    else:
        logg = lg_fast(zthatthatz ** 0.5)
    return logg


def lG(x, f1, f2):
    """
    Exact form of lG
    :param x:
    :param f1:
    :param f2:
    :return:
    """
    anorm = (2. / np.pi) ** 0.5 / (1 + f1 + f2)
    term1 = f1 * np.exp(-0.5 * x ** 2)
    term2 = f2 * np.exp(-0.5 * x ** 2) * x ** 2
    term3 = np.pi / 8 * \
        ((6 - 6 * x ** 2 + x ** 4) * i0e(x ** 2 * 0.25) -
         x ** 2 * (-4 + x ** 2) * i1e(x ** 2 * 0.25))
    return np.log(anorm * (term1 + term2 + term3))


@vectorize(nopython=True)
def lg_approx(x, f1, f2):
    """
    Approximate form of lG valid for high SNR
    :param x:
    :param f1:
    :param f2:
    :return:
    """
    anorm = 9. / 2. / (1 + f1 + f2)
    term1 = 1 / x ** 5 * np.exp(12 / x ** 2 + 136 / x ** 4)
    return np.log(anorm * term1)


@njit
def lg_fast(x):
    """Fast evaluation of the marginalized piece via an interpolation table for
    a vector x, much faster via vectors than many scalars due to vectorized
    interp"""
    x = np.atleast_1d(np.asarray(x))
    out = np.zeros_like(x, dtype=np.float64)
    imask = x < 10
    if np.any(imask):
        out[imask] = np.interp(x[imask], LG_FAST_XP, LG_FAST_YP)
    imask = np.logical_not(imask)
    if np.any(imask):
        out[imask] = lg_approx(x[imask], f1, f2)
    return out


@njit
def marg_lk(zz, tt, gtype=1, nsamp=None):
    """
    Computes likelihood marginalized over distance and orbital phase
    :param zz:
        nsamp x n_detector array with rows having complex overlaps for each
        detector
    :param tt: nsamp x n_detector array with predicted overlaps in each
        detector for a fiducial orbital phase and distance
        (upto an arbitrary scaling factor)
    :param gtype: Flag passed to function to compute marginalization over
        distance and phase, determines which one to run
    :param nsamp: Pass to only sum over part of the arrays
    :returns:
        1. nsamp array of likelihoods marginalized over the orbital phase, and
           distance if needed (always 1D), with extra factor of e^{-|z^2|/2}
           arising from importance sampling over arrival times
        2. 2 x nsamp complex array of Z \bar{T}, and \bar{T} T
           (implicit summation over detectors), useful for reconstructing the
           distance and phase samples
    """
    if nsamp is None:
        nsamp = len(zz)

    # Sums over detectors, do it this way to save on some allocations
    z2 = np.zeros(nsamp)
    t2_pow = np.zeros(nsamp)
    zthatthatz = np.zeros(nsamp)
    UT2samples = np.zeros((2, nsamp), dtype=np.complex128)
    for i in range(nsamp):
        ztbar = 0. + 0.j
        t2 = 0
        for j in range(zz.shape[1]):
            z2[i] += zz[i, j].real**2 + zz[i, j].imag**2
            t2 += tt[i, j].real**2 + tt[i, j].imag**2
            ztbar += zz[i, j] * np.conj(tt[i, j])

        zttz = ztbar.real**2 + ztbar.imag**2
        zthatthatz[i] = zttz / t2
        t2_pow[i] = t2 ** 1.5

        UT2samples[0, i] = ztbar
        UT2samples[1, i] = t2

    logg = lgg(zthatthatz, gtype)

    lk = np.zeros(nsamp)
    for i in range(nsamp):
        dis_phase_marg = t2_pow[i] * np.exp(logg[i])
        lk[i] = np.exp(-0.5 * (z2[i] - zthatthatz[i])) * dis_phase_marg

    return lk, UT2samples


@njit
def coherent_score_montecarlo_sky(
        timeseries, offsets, nfacs, dt_dict_keys, dt_dict_items,
        responses, t3norm, musamps=None, psisamps=None, c2psisamps=None,
        s2psisamps=None, gtype=1, dt_sinc=DEFAULT_DT,
        dt_max=DEFAULT_DT_MAX, nsamples=10000, fixed_pars=None,
        fixed_vals=None):
    """
    Evaluates the coherent score integral by montecarlo sampling all
    relevant variables
    :param timeseries:
        Tuple with n_samp x 3 arrays with times, Re(z), Im(z) in each detector
    :param offsets:
        (n_det - 1) array with offsets to add to the detectors > first one
        (e.g., H1 for H1, L1) to bring them to zero lag
    :param nfacs:
        n_det array of instantaneous sensitivities in each detector
        (normfac/psd_drift x hole_correction)
    :param dt_dict_keys:
        Keys to dictionary computed using the delays in each dt_tuple, sorted
    :param dt_dict_items:
        Values in dictionary, tuple of n_sky x 2 arrays with indices into
        ras, decs for each allowed dt tuple
    :param responses:
        n_ra x n_dec x n_detector x 2 array with detector responses
    :param t3norm:
        normalization constant such that prior integrated over the sky = 1
    :param musamps: If available, array with samples of mu (cos inclination)
    :param psisamps: If available, array with samples of psi (pol angle)
    :param gtype: 0/1 to not marginalize/marginalize over distance
    :param dt_sinc: Time resolution for the dictionary (ms)
    :param dt_max: Rough upper bound on the individual delays (ms)
    :param nsamples: Number of samples for montecarlo evaluation
    :param fixed_pars: Tuple with the names of the parameters we fix
        Can be 'tn' (time of the n^th detector), 'mu', 'psi', 'radec_ind'
    :param fixed_vals: Tuple with the values of the parameters we fix
    :returns:
        1. Montecarlo evaluation of 2 * log(complete coherent score)
           (including the incoherent part)
        2. nsamples x 6 array with each row having
           mu,
           psi,
           ra_index,
           dec_index,
           (unnormalized) relative contribution of sample to coherent score
           time corresponding to the sample in the first detector
        3. 2 x nsamples_phys complex array of Z \bar{T}, and \bar{T} T
           (implicit summation over detectors), useful for reconstructing the
           distance and phase samples
        Note that the number of physical samples < nsamples
    """
    # Set the seed from the first H1 time (hardcoded)
    # np.random.seed(int(timeseries[0][0, 0]))

    # Define some required variables
    ndet = len(timeseries)
    n_sky_pos = responses.shape[0] * responses.shape[1]

    # Check if fixing parameters
    fixing = fixed_pars is not None

    if fixing and ('mu' in fixed_pars):
        muind = fixed_pars.index('mu')
        musamps = fixed_vals[muind] * np.ones(nsamples)
    elif musamps is None:
        musamps = np.random.uniform(-1, 1, size=nsamples)

    if fixing and ('psi' in fixed_pars):
        psiind = fixed_pars.index('psi')
        psisamps = fixed_vals[psiind] * np.ones(nsamples)
    elif psisamps is None:
        psisamps = np.random.uniform(0, 2 * np.pi, size=nsamples)
        c2psisamps = np.cos(2*psisamps)
        s2psisamps = np.sin(2*psisamps)

    # Pick samples of data points in each detector
    # -----------------------------------------------------------------------
    # Normalization factor for monte-carlo over times in the detectors
    twt = 1.

    # Pick samples ahead of time, since np.interp is faster when vectorized
    # Samples picked according to e^{\rho^2}/2
    tz_samples = np.zeros((ndet, 3, nsamples))
    z2max_samples = np.zeros(ndet)

    # Go over each detector
    for ind_d in range(ndet):
        pclist_d = timeseries[ind_d]

        # Define the cumsum of weights for the random sampling over times
        tnstr = 't' + str(ind_d)
        if fixing and (tnstr in fixed_pars):
            tnind = fixed_pars.index(tnstr)
            tn = fixed_vals[tnind]
            tnarr = np.searchsorted(pclist_d[:, 0], np.array([tn]))
        else:
            tnarr = np.arange(len(pclist_d))

        z2 = pclist_d[tnarr, 1]**2 + pclist_d[tnarr, 2]**2
        z2max_samples[ind_d] = np.max(z2)
        # Removing the maximum to avoid large exponentials
        z2 -= z2max_samples[ind_d]

        cwts_d = np.cumsum(np.exp(0.5 * z2))
        twt_d = cwts_d[-1]
        cwts_d /= twt_d

        # Record in the total weight factor
        twt *= twt_d

        # Pick samples according to the correct probabilities
        tz_samples[ind_d] = \
            utils.rand_choice_nb(pclist_d[tnarr], cwts_d, nsamples).T

        # Add offsets to detectors > the first detector
        if ind_d > 0:
            tz_samples[ind_d, 0] += offsets[ind_d - 1]

    # Generate keys into the RA-Dec dictionary from the delays, do them
    # at once to save some time
    # Delays in ms ((ndet - 1) x nsamples)
    dts = tz_samples[1:, 0, :] - tz_samples[0, 0, :]
    dts *= 1000
    keys = dt2key(dt=dts, dt_sinc=dt_sinc, dt_max=dt_max)

    # Populate the structures to evaluate the marginalized likelihood with
    # samples that have allowed delays
    nsamp_phys = 0
    zzs = np.zeros((nsamples, ndet), dtype=np.complex128)
    tts = np.zeros((nsamples, ndet), dtype=np.complex128)
    fskys = np.zeros(nsamples, dtype=np.int32)

    samples = np.zeros((nsamples, 6))

    dt_dict_key_inds = np.searchsorted(dt_dict_keys, keys)

    indx_offset_mu = np.random.randint(len(musamps))
    indx_offset_psi = np.random.randint(len(psisamps))
    indx_offset_ra = np.random.randint(10**7)

    for ind_s in range(nsamples):
        dt_dict_key_ind = dt_dict_key_inds[ind_s]
        key = keys[ind_s]
        if (dt_dict_key_ind < len(dt_dict_keys)) and \
                (dt_dict_keys[dt_dict_key_ind] == key):
            # Add to list of samples of z
            zzs[nsamp_phys] = \
                tz_samples[:, 1, ind_s] + 1j * tz_samples[:, 2, ind_s]

            # Pick RA, Dec to get f_+, f_x
            radec_indlist = dt_dict_items[dt_dict_key_ind]

            if fixing and ('radec_ind' in fixed_pars):
                radec_ind_ind = fixed_pars.index('radec_ind')
                radec_ind = int(fixed_vals[radec_ind_ind])
                fskys[nsamp_phys] = 1
                if radec_ind >= len(radec_indlist):
                    return float(-10 ** 5), samples, \
                        np.zeros((2, 0), dtype=np.complex128)
            else:
                #radec_ind = np.random.choice(len(radec_indlist))
                radec_ind = (ind_s + indx_offset_ra)%len(radec_indlist)
                # Record fsky
                # (normalization factor for monte-carlo over ra and dec)
                fskys[nsamp_phys] = len(radec_indlist)

            ra_ind, dec_ind = radec_indlist[radec_ind]

            # Pick mu and psi
            #mu = np.random.choice(musamps)
            #psi = np.random.choice(psisamps)
            mu = musamps[(ind_s + indx_offset_mu)%len(musamps)]
            psi_indx = (ind_s + indx_offset_psi)%len(psisamps)
            psi = psisamps[psi_indx]
            c2psi = c2psisamps[psi_indx]
            s2psi = s2psisamps[psi_indx]

            samples[nsamp_phys, 0] = mu
            samples[nsamp_phys, 1] = psi
            samples[nsamp_phys, 2] = ra_ind
            samples[nsamp_phys, 3] = dec_ind

            # Time entry
            # Zero for the first detector, 0 for time
            samples[nsamp_phys, 5] = tz_samples[0, 0, ind_s]

            # Add to list of predicted z
            for ind_d in range(ndet):
#                 tts[nsamp_phys, ind_d] = \
#                     nfacs[ind_d] * gen_sample_amps_from_fplus_fcross(
#                         responses[ra_ind, dec_ind, ind_d, 0],
#                         responses[ra_ind, dec_ind, ind_d, 1],
#                         mu, psi)
                tts[nsamp_phys, ind_d] = \
                nfacs[ind_d] * gen_sample_amps_from_fplus_fcross(
                    responses[ra_ind, dec_ind, ind_d, 0],
                    responses[ra_ind, dec_ind, ind_d, 1],
                    mu, c2psi, s2psi)

            nsamp_phys += 1

    if nsamp_phys > 0:
        # Generate list of unnormalized marginalized likelihoods
        marg_lk_list, UT2samples = marg_lk(zzs, tts, gtype=gtype, nsamp=nsamp_phys)

        # Sum with the right weights to get the net marginalized likelihood
        # The nsamples is not a bug, it needs to be to use weight = twt
        wfac = twt / nsamples / t3norm / n_sky_pos
        s = 0
        for i in range(nsamp_phys):
            s += marg_lk_list[i] * fskys[i]
            samples[i, 4] = marg_lk_list[i] * fskys[i]
        s *= wfac
        samples[:, 4] *= wfac
        score = 2. * np.log(s) + np.sum(z2max_samples)
        return score, samples, UT2samples
    return -1e5, samples, np.zeros((2, 0), dtype=np.complex128)


# ###############################################################################
class CoherentScore:
    def __init__(
            self, samples_fname=None, detnames=None, norm_angles=False, run='O2',
            empty_init=False):
        """
        :param samples_fname:
            Path to file with samples, created by create_samples
        :param detnames:
            If known, pass the list of detector names for plot labels.
            If not given, we will infer it from the filename, assuming it was
            generated by create_samples
        :param norm_angles: Flag to recompute normalization w.r.t angles
        :param run: Which run we're computing the coherent score for
        :param empty_init:
            Flag to return an empty instance, useful for alternate init
        """
        if empty_init:
            return

        if samples_fname is None:
            raise ValueError("Need to know where to source samples from.")

        # Read the contents of the sample file
        npzfile = np.load(samples_fname, allow_pickle=True)

        # Set some scalar parameters
        # Time resolution of dictionary in ms
        self.dt_sinc = float(npzfile['dt_sinc'])
        # Upper bound on the delays in ms
        self.dt_max = float(npzfile['dt_max'])

        # Arrays of RA and dec
        self.lon_grid = npzfile['lon_grid']
        self.lat_grid = npzfile['lat_grid']

        # Dictionary indexed by keys for delta_ts, containing
        # n_skypos x 2 arrays with indices into ra_grid and dec_grid
        dt_dict = npzfile['dt_dict'].item()
        self.dt_dict_keys = np.fromiter(dt_dict.keys(), dtype=np.int32)
        self.dt_dict_keys.sort()
        #self.dt_dict_items = tuple([dt_dict[key] for key in self.dt_dict_keys])
        self.dt_dict_items = List()
        for key in self.dt_dict_keys:
            self.dt_dict_items.append(dt_dict[key])

        # n_ra x n_dec x n_detectors x 2 array with f_+/x for phi = 0
        self.responses = npzfile['responses']
        # Convenience for later
        self.ndet = self.responses.shape[2]
        if detnames is not None:
            self.detnames = detnames
        else:
            self.detnames = \
                samples_fname.split("RA_dec_grid_")[1].split(
                    f"_{int(1000/self.dt_sinc)}")[0].split("_")

        # Samples of mu (cos inclination) and psis
        self.mus = npzfile['mus']
        self.psis = npzfile['psis']
        self.c2psis = np.cos(2*self.psis)
        self.s2psis = np.sin(2*self.psis)

        # Arrays for debugging purpose
        # n_ra x n_dec x (n_detectors - 1) array with delays w.r.t
        # the first detector
        self.deltats = npzfile['deltats']
        # n_ra x n_dec x (n_detectors - 1) array with phases w.r.t
        # the first detector
        self.dphases = npzfile['dphases']
        # n_ra x n_dec array with total network response
        self.rtot2s = npzfile['rtot2s']

        # choose a normalization constant
        # To recompute normalization use norm_angles=True
        self.T3norm = 0.2372446769308674
        if norm_angles:
            self.gen_t3_norm()

        # If self.Gtype=0 then distance-phase is not prior
        self.Gtype = 1

        # Choose a reference normfac for the run
        # accounts for the relative sensitivity of the detectors
        # chosen so that spinning guy has normalization=1
        # chosen so that Liang's trigger has normalization=1
        self.run = run
        if run.lower() == 'o1':
            self.norm_h1_normalization = 7.74908546328925
            self.norm_l1_normalization = 7.377041393512488
            self.normfac_pos = 2
            self.hole_correction_pos = None
            self.psd_drift_pos = 3
            self.rezpos = 4
            self.imzpos = 5
            self.c0_pos = 6
        else:
            # Set default to new runs
            self.norm_h1_normalization = 0.03694683692493602
            self.norm_l1_normalization = 0.042587464435623064
            self.normfac_pos = 2
            self.hole_correction_pos = 3
            self.psd_drift_pos = 4
            self.rezpos = 5
            self.imzpos = 6
            self.c0_pos = 7

    @classmethod
    def from_new_samples(
            cls, nlon, nlat, detnames,
            dt_sinc=DEFAULT_DT, dt_max=DEFAULT_DT_MAX, nsamples_mupsi=100000,
            run="O2"):
        """
        Function to create a new class instance with a dictionary from scratch
        :param nra: Number of points in the RA direction
        :param ndec: Number of points in the Dec direction
        :param detnames: iterable to detector names
        :param gps_time:
        :param dt_sinc: Time resolution of samples in ms for the dictionary
        :param dt_max: Maximum delay between detectors in ms
        :param nsamples_mupsi:
            Number of random samples of the inclination and the polarization
        :param run:
        :return: Instance of CoherentScoreMZ
        """
        
        
        # Create structures to deal with the mapping of the sphere to delays
        dt_dict, lon_grid, lat_grid, responses, deltats, dphases, rtot2s = \
            create_time_dict(
                nlon, nlat, detnames, dt_sinc=dt_sinc,
                dt_max=dt_max)

        # Create random samples of the cosine of the inclination, and the
        # polarization
        psis = np.random.uniform(0, 2 * np.pi, size=nsamples_mupsi)
        mus = np.random.uniform(-1, 1, size=nsamples_mupsi)  # cos(inclination)
        
        # Create an empty instance and read in the parameters
        instance = cls(empty_init=True)

        # Set parameters
        instance.dt_sinc = dt_sinc  # Time resolution of dictionary in ms
        instance.dt_max = dt_max  # Upper bound on the delays in ms

        # Arrays of RA and DEC
        instance.lon_grid = lon_grid
        instance.lat_grid = lat_grid

        # Dictionary indexed by keys for delta_ts, containing
        # n_skypos x 2 arrays with indices into ra_grid and dec_grid
#         instance.dt_dict = dt_dict
        instance.dt_dict_keys = np.fromiter(dt_dict.keys(), dtype=np.int32)
        instance.dt_dict_keys.sort()
        #instance.dt_dict_items = tuple([dt_dict[key] for key in instance.dt_dict_keys])
        instance.dt_dict_items = List()
        for key in instance.dt_dict_keys:
            instance.dt_dict_items.append(dt_dict[key])

        # n_ra x n_dec x n_detectors x 2 array with f_+/x for phi = 0
        instance.responses = responses
        # Convenience for later
        instance.ndet = responses.shape[2]
        instance.detnames = detnames

        # Samples of mu (cos inclination) and psis
        instance.mus = mus
        instance.psis = psis
        instance.c2psis = np.cos(2*instance.psis)
        instance.s2psis = np.sin(2*instance.psis)

        # Arrays for debugging purpose
        # n_ra x n_dec x (n_detectors - 1) array with delays w.r.t
        # the first detector
        instance.deltats = deltats
        # n_ra x n_dec x (n_detectors - 1) array with phases w.r.t
        # the first detector
        instance.dphases = dphases
        # n_ra x n_dec array with total network response
        instance.rtot2s = rtot2s

        # Generate norm to be safe
        instance.T3norm = 0.2372446769308674
        instance.gen_t3_norm()

        ## if self.Gtype=0 then distance-phase is not prior
        instance.Gtype = 1

        ## Choose value to normalize T
        ## accounts for the reltive sensitivity of the detectors
        ## chosen so that spinning guy has normalization=1
        ## chosen so that Liang's trigger has normalization=1
        instance.run = run
        if run == 'O1':
            instance.norm_h1_normalization = 7.74908546328925
            instance.norm_l1_normalization = 7.377041393512488
            instance.normfac_pos = 2
            instance.hole_correction_pos = None
            instance.psd_drift_pos = 3
            instance.rezpos = 4
            instance.imzpos = 5
            instance.c0_pos = 6
        else:
            instance.norm_h1_normalization = 0.03694683692493602
            instance.norm_l1_normalization = 0.042587464435623064
            instance.normfac_pos = 2
            instance.hole_correction_pos = 3
            instance.psd_drift_pos = 4
            instance.rezpos = 5
            instance.imzpos = 6
            instance.c0_pos = 7

        return instance

    def save_samples(self, fname):
        filename = utils.rm_suffix(fname, suffix='.npz', new_suffix="_") + \
            f"{int(1000 / self.dt_sinc)}" + f"_{int(self.gps)}.npz"
        dt_dict = dict(zip(self.dt_dict_keys, self.dt_dict_items))
        np.savez(filename,
                 dt_dict=dt_dict, ra_grid=self.ra_grid,
                 dec_grid=self.dec_grid, responses=self.responses,
                 deltats=self.deltats, dphases=self.dphases,
                 rtot2s=self.rtot2s, psis=self.psis, mus=self.mus,
                 gps_time=self.gps, dt_sinc=self.dt_sinc, dt_max=self.dt_max)

    def gen_t3_norm(self, nsamples=10 ** 6, verbose=False):
        if verbose:
            print('Old T3norm', self.T3norm)
        mus = np.random.choice(self.mus, size=nsamples, replace=True)
        psis = np.random.choice(self.psis, size=nsamples, replace=True)
        c2psis = np.cos(2*psis)
        s2psis = np.sin(2*psis)
        lon_inds = np.random.randint(0, len(self.lon_grid), size=nsamples)
        lat_inds = np.random.randint(0, len(self.lat_grid), size=nsamples)
        t_list = np.zeros(nsamples)
        for det_ind in range(self.ndet):
            responses = self.responses[lon_inds, lat_inds, det_ind, :]
            fplus = responses[:, 0]
            fcross = responses[:, 1]
            #fplus, fcross = self.responses[ra_inds, dec_inds, det_ind, :]
            avec = gen_sample_amps_from_fplus_fcross(fplus, fcross, mus, c2psis, s2psis)
            t_list += utils.abs_sq(avec)
        t3mean = np.mean(t_list ** 1.5)
        self.T3norm = t3mean
        if verbose:
            print('New T3norm', self.T3norm)

    def dt_dict(self, keyval):
        """Convenience function to simulate a dictionary"""
        ind = np.searchsorted(self.dt_dict_keys, keyval)

        if not ((ind < len(self.dt_dict_keys)) and
                (self.dt_dict_keys[ind] == keyval)):
            raise KeyError(f"Invalid key {keyval}")

        return self.dt_dict_items[ind]

    def get_all_prior_terms_with_samp(
            self, events, timeshifts, timeseries, ref_normfac=1,
            time_slide_jump=DEFAULT_TIMESLIDE_JUMP/1000, **score_kwargs):
        """
        :param events:
            n_detector x processedclist
            array with coincidence/background candidates (peaks of the
            timeseries)
        :param timeshifts: Array of times for timeseries
        :param timeseries: 2D array of shape n_len(timeshifts) x n_detector 
                    array with z
        :param ref_normfac: Reference normfac to scale the values relative to
        :param time_slide_jump: The least count of input time slides (s)
        :param score_kwargs: Extra arguments to comblist2cs
        :return:
            1. 2xlog(coherent score)
            2. Array n_samples x 6 array for each event
            3. Array
               2 x nsamples complex array of Z \bar{T}, and \bar{T} T
               (implicit summation over detectors), useful for reconstructing the
               distance and phase samples
        """
        if utils.checkempty(timeseries):
            raise ValueError(
                "Need input timeseries to compute the coherent score")

        # n_detector x 4
        params = self.get_params(events, time_slide_jump=time_slide_jump)[0] #TODO

        # Some useful parameters
        rhosq = incoherent_score(events)
        offsets = params[1:, 0] - events[1:, 0]
        nfacs = params[:, 3] / ref_normfac

        f_sampling = 1000/(self.dt_sinc) # ms to s
        if (fs_ratio := f_sampling * (timeshifts[1]-timeshifts[0])) != 1:
            timeseries, timeshifts = scipy.signal.resample(
                timeseries, int(len(timeshifts) * fs_ratio), timeshifts, axis=0)
        if not np.isclose(1/f_sampling, timeshifts[1] - timeshifts[0]):
            raise ValueError('`timeshifts` is incommensurate with `f_sampling`.') 

        timeseries_ev = []
        for ind_det in range(timeseries.shape[1]):
            timeseries_ev.append(np.c_[timeshifts, timeseries[:,ind_det].real, timeseries[:,ind_det].imag ])
        timeseries_ev = tuple(timeseries_ev)
        
        prior_terms, samples, UT2samples = self.comblist2cs(
            timeseries_ev, offsets, nfacs, **score_kwargs)
        
        return prior_terms, samples,  UT2samples

    def comblist2cs(
            self, timeseries, offsets, nfacs, gtype=1, nsamples=10000,
            **kwargs):
        """
        Takes the return value of trigger2comblist and returns the coherent
        score by calling the jitted function
        :param timeseries:
            Tuple of length n_detectors with n_samp x 3 array with Re(z), Im(z)
        :param offsets:
            Array of length n_detector -1 with shifts to apply to the
            timeseries > H1 to bring them to zero lag
        :param nfacs: Array of length n_detector with detector sensitivities
        :param gtype: If 0, turns off the integration over distance
        :param nsamples: Number of samples to use in the monte carlo
        :param kwargs:
            Generic variable to capture any extra arguments, in this case,
            we can pass
                fixed_pars = Tuple with the names of the parameters to
                    hold fixed, can be 'tn' (time of the n^th detector), 'mu',
                    'psi', 'radec_ind'
                fixed_vals = Tuple with the values of the parameters we fix
        :return:
            1. Montecarlo evaluation of 2 * log(complete coherent score)
               (including the incoherent part)
            2. nsamples x 6 array with each row having
               mu,
               psi,
               ra_index,
               dec_index,
               (unnormalized) relative contribution of sample to coherent score
               time corresponding to the sample in the first detector
            3. 2 x nsamples_phys complex array of Z \bar{T}, and \bar{T} T
               (implicit summation over detectors), useful for reconstructing
               the distance and phase samples
            Note that the number of physical samples < nsamples
        """
        # if timeseries is empty, return -10^5
        # (see coherent_score_montecarlo_sky()) and continue
        if any([len(x) == 0 for x in timeseries]):
            return -100000, np.zeros((nsamples, 6)), \
                np.zeros((2, 0), dtype=np.complex128)


        return coherent_score_montecarlo_sky(
            timeseries, offsets, nfacs, self.dt_dict_keys, self.dt_dict_items,
            self.responses, self.T3norm, musamps=self.mus, psisamps=self.psis,
            c2psisamps=self.c2psis, s2psisamps=self.s2psis,
            gtype=gtype, dt_sinc=self.dt_sinc, dt_max=self.dt_max,
            nsamples=nsamples, fixed_pars=kwargs.get("fixed_pars", None),
            fixed_vals=kwargs.get("fixed_vals", None))

    def get_params(self, events, time_slide_jump=DEFAULT_TIMESLIDE_JUMP/1000):
        """
        :param events:
            (n_cand x (n_det = 2) x processedclist)/
            ((n_det = 2) x processedclist) array with coincidence/background
            candidates
        :param time_slide_jump: Units of jumps (s) for timeslides
        :return: n_cand x (n_det = 2) x 4 array (always 3D) with
            shifted_ts, re(z), im(z), effective_sensitivity in each detector
        """
        if events.ndim == 2:
            # We're dealing with a single event
            events = events[None, :]

        dt_shift = self.dt_sinc / 1000  # in s

        # Add shifts to each detector to get to zero lag
        # n_cand x n_det
        ts_out = events[:, :, 0].copy()
        shifts = offset_background(
            ts_out[:, 1:] - ts_out[:, 0][:, None], time_slide_jump, dt_shift)
        ts_out[:, 1:] += shifts

        # Overlaps
        # n_cand x n_det
        rezs = events[:, :, self.rezpos]
        imzs = events[:, :, self.imzpos]

        # Sensitivity
        # The hole correction is a number between 0 and 1 reflecting the
        # sensitivity after some parts of the waveform fell into a hole
        # asd drift is the effective std that the score was divided with
        # => the bigger it is, the less the sensitivity
        asd_corrs = events[:, :, self.psd_drift_pos]
        ns = events[:, :, self.normfac_pos]
        if self.hole_correction_pos is not None:
            # New format
            # Hole corrections
            hs = events[:, :, self.hole_correction_pos]
            n_effs = ns / asd_corrs * hs
        else:
            # Old format in O1
            # Normfacs are actually normfac x psd_drift
            n_effs = ns / asd_corrs ** 2

        return np.stack((ts_out, rezs, imzs, n_effs), axis=2)
