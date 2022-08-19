import sys
sys.path.append('/data/tislam/works/KITP/repos/cogwheel/')
sys.path.append('/data/tislam/works/KITP/repos/fast_pe_using_cs/')
import coherent_score_mz_fast_tousif as cs
import numpy as np
import scipy.special

"""
This code samples distance and phase using coherent score code
returns full set of posteriors including both intrinsic and extrinsic ones
"""

def posterior_func_for_dist(u_pr, U, T2): 
    """
    Analytical posterior funtion for distance
    Eq. (3.39) in Teja's note
    u_pr is 1/dist
    """
    U_abs = abs(U)
    exponent = - 0.5 * T2 * ( u_pr - U_abs/T2)**2 
    return (1./u_pr**4)*np.exp( exponent  ) * scipy.special.i0e(u_pr*U_abs)


def posterior_fuc_for_phase(v_pr, U, y):  
    """
    Analytical posterior funtion for phase
    Eq. (3.38) in Teja's note
    v_pr is d_phase
    """
    U_abs = abs(U)
    return np.exp( U_abs* y * np.cos(2*v_pr) )


def get_distance_phase_point_for_given_U_T2(UT2samples, U, T2):
    """
    Function to sample distance and phase from their analytical distributions
    given U, T2 and UT2samples obatined for each values of intrinsic params
    returns single value for distance and phase
    """
    mean = abs(U)/T2 # expected mean for 1/distance
    sigma = 1/np.sqrt(T2)
    # demand that the minimum of u_vals can't be zero or negative because u_vals~1/dist
    u_vals = np.linspace( max(1e-5,mean-7*sigma), mean+7*sigma, 1000)
    like_vals = posterior_func_for_dist(u_vals, U, T2)
    # compute normalized cdf
    cdf = np.cumsum(like_vals)
    cdf = cdf/cdf[-1]
    u_pick = cs.rand_choice_nb(u_vals, cdf, 1)[0] # cs.rand_choise_nb is faster than numpy.random.choice()
    # note: fiducial value for dist_ref is 1Mpc; the following equation is valid for that
    dist_pick = 1/u_pick

    mean = np.angle(U)/2 # expected mean for phase
    sigma = 1/np.sqrt(abs(U)*u_pick) # expected sigma for the d_phase
    v_vals = np.linspace( max(-7*sigma,-np.pi/2), min(7*sigma, np.pi/2), 1000)
    like_vals = posterior_fuc_for_phase(v_vals, U, u_pick)
    # compute normalized cdf
    cdf = np.cumsum(like_vals)
    cdf = cdf/cdf[-1]
    # obtain a d_phase value
    v_pick = cs.rand_choice_nb(v_vals, cdf, 1)[0]
    # add it to mean expected phase
    # (np.random.random()>0.5)*np.pi to take care of the pi degeneracy of the 22 mode
    phi_pick = ( (v_pick + mean + (np.random.random()>0.5)*np.pi) )% (2*np.pi)

    return dist_pick, phi_pick


def sample_extrinsic_params(intrinsic_samples, post, **kwargs):
    """
    Wrapper to sample all extrinsic params
    returns a dictionary of all intrinsic and extrinsic samples, and
    the log of the unmarginalized likelihood for the set of samples
    """

    print('finding the corresponding extrinsic params now')

    samples ={}
    
    # mc, q, chieff values
    samples['mchirp'] = intrinsic_samples['mchirp']
    samples['q'] = np.exp(intrinsic_samples['lnq'])
    samples['chieff'] = intrinsic_samples['chieff']
    samples['cumchidiff'] = intrinsic_samples['cumchidiff']

    # initialize arrays to store sampling values
    samples['mu'] = np.zeros(len(samples['mchirp']))
    samples['psi'] = np.zeros(len(samples['mchirp']))
    samples['ra'] = np.zeros(len(samples['mchirp']))
    samples['dec'] = np.zeros(len(samples['mchirp']))
    samples['dist'] = np.zeros(len(samples['mchirp']))
    samples['phase'] = np.zeros(len(samples['mchirp']))
    samples['lnlike'] = np.zeros(len(samples['mchirp']))

    # compute mu, psi, ra, dec, dist, phase in loops
    count=0

    for iteration in range(len(samples['mchirp'])):
        # get mu, psi, ra, dec
        _, samples['mu'][count], samples['psi'][count], samples['ra'][count], \
            samples['dec'][count], U, T2, UT2samples, _ = \
            post.likelihood.obtain_sky_loc_params_from_cs(
                post.prior.transform(
                    mchirp=intrinsic_samples['mchirp'][iteration],
                    lnq=intrinsic_samples['lnq'][iteration],
                    chieff=intrinsic_samples['chieff'][iteration],
                    cumchidiff=intrinsic_samples['cumchidiff'][iteration]), **kwargs)

        # get distance and phase
        samples['dist'][count], samples['phase'][count] = \
            get_distance_phase_point_for_given_U_T2(UT2samples, U, T2)
        # note: fiducial value for dist_ref is 1Mpc; the following equation is valid for that
        Y_pick = (1/samples['dist'][count]) * \
            np.exp(2 * 1j * samples['phase'][count])
        samples['lnlike'][count] = 0.5 * (
                np.abs(U)**2/T2 - T2*np.abs(Y_pick - U/T2)**2)

        # increase count
        count = count + 1

    print('Sampling finished.')

    return samples