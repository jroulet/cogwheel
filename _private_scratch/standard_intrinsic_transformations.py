import numpy as np
# until moved into lib
import sys
sys.path.append("..")
from cogwheel import cosmology as cosmo

#     Mass Variable Conversions
# ---------------------------------
# q and eta
def eta2q(eta):
    return (1 - np.sqrt(1 - 4 * eta) - 2 * eta) / (2 * eta)

def q2eta(q):
    return q / (1 + q) ** 2

# component masses from chirp mass and ratio
def m1_of_mchirp_q(mchirp, q):
    return mchirp * (1 + q) ** .2 / q ** .6

def m2_of_mchirp_q(mchirp, q):
    return mchirp * (1 + 1 / q) ** .2 * q ** .6

# component masses from total mass and ratio
def m1_of_mtot_q(mtot, q):
    return mtot / (1 + q)

def m2_of_mtot_q(mtot, q):
    return mtot * q / (1 + q)

# chirp mass
def mchirp_of_m1_m2(m1, m2):
    return (m1 * m2) ** .6 / (m1 + m2) ** .2

def m1_of_mchirp_m2(mchirp, m2):
    """Also works for 1 <=> 2"""
    return (mchirp ** (5 / 3) * (2 * 3 ** (1 / 3) * mchirp ** (5 / 3) + 2 ** (1 / 3)
                                 * (9 * m2 ** 2.5 + np.sqrt(81 * m2 ** 5 - 12 * mchirp ** 5)) ** (2 / 3))
            / (6 ** (2 / 3) * m2 ** 1.5
               * (9 * m2 ** 2.5 + np.sqrt(81 * m2 ** 5 - 12 * mchirp ** 5)) ** (1 / 3)))

# component masses from chirp mass and eta
def m1_of_mchirp_eta(mchirp, eta):
    return m1_of_mchirp_q(mchirp, eta2q(eta))

def m2_of_mchirp_eta(mchirp, eta):
    return m2_of_mchirp_q(mchirp, eta2q(eta))

def q_and_eta(q=None, eta=None):
    if q is not None:
        eta = q / (1. + q)**2
    elif eta is not None:
        q = (1. - 2. * eta - np.sqrt(1. - 4. * eta)) / (2. * eta)
    else:
        raise RuntimeError("I need parameters to convert")
    return {'q': q, 'eta': eta}

def m1_m2_mt(m1=None, m2=None, mt=None):
    """complete m1, m2, mt or check consistency if all given --> return m1, m2, mt"""
    if mt is None:
        mt = m1 + m2
    elif m2 is None:
        m2 = mt - m1
    elif m1 is None:
        m1 = mt - m2
    else:
        assert np.allclose(m1 + m2, mt), 'parameter error: mt = m1 + m2 must hold!'
    assert np.any(m1 < m2) == False, 'cannot have m1 < m2'
    return m1, m2, mt

def standard_mass_conversion(**dic):
    """
    Computes all conversions given a subset of parameters describing the masses
    :param dic:
        Dictionary with known subset of mc, mt, m1, m2, q, eta
        (all can be scalars or numpy arrays)
    :return: dic with all values solved for
    """
    mc = dic.get('mchirp', dic.get('mc'))
    mt = dic.get('mtot', dic.get('mt'))
    m1 = dic.get('m1')
    m2 = dic.get('m2')
    q = dic.get('q')
    eta = dic.get('eta')
    if (q is None) and (dic.get('q1') is not None):
        q = 1 / dic['q1']
        dic['q'] = q

    # Check if we have to do any work
    if not np.any(map(lambda x: x is None, [mc, mt, m1, m2, q, eta])):
        return dic

    # First ensure that q is on the right domain
    if (q is not None) and np.any(q > 1):
        if hasattr(q, "__len__"):
            q[q > 1] = 1/q[q > 1]
        else:
            q = 1/q
        dic['q'] = q
        return standard_mass_conversion(**dic)

    # Second ensure that q and eta are always defined together
    if (q is not None) != (eta is not None):
        dic_q = q_and_eta(q=q, eta=eta)
        dic.update(dic_q)
        return standard_mass_conversion(**dic)

    # If q is not defined, do what is needed to get it
    if q is None:
        if mt is not None:
            if mc is not None:
                dic['eta'] = (mc / mt)**(5. / 3.)
                return standard_mass_conversion(**dic)
            elif m1 is not None:
                dic['q'] = mt / m1 - 1
                return standard_mass_conversion(**dic)
            elif m2 is not None:
                dic['q'] = m2 / (mt - m2)
                return standard_mass_conversion(**dic)
            else:
                raise RuntimeError("Enough parameters weren't defined")
        elif m1 is not None:
            if mc is not None:
                r = (mc / m1)**5
                if np.any(np.logical_or(0 > r, r > 1/2)):
                    raise RuntimeError("I couldn't find a physical solution")
                if hasattr(r, "__len__"):
                    qdic = []
                    for rval in r:
                        qvals = np.roots([1, 0, -rval, -rval])
                        qdic.append(qvals[np.isreal(qvals)][0].real)
                    dic['q'] = np.asarray(qdic)
                else:
                    qvals = np.roots([1, 0, -r, -r])
                    dic['q'] = qvals[np.isreal(qvals)][0].real
                return standard_mass_conversion(**dic)
            elif m2 is not None:
                dic['q'] = m2 / m1
                return standard_mass_conversion(**dic)
            else:
                raise RuntimeError("Enough parameters weren't defined")
        elif m2 is not None:
            if mc is not None:
                r = (m2 / mc)**5
                if np.any(np.logical_or(0 > r, r > 2)):
                    raise RuntimeError("I couldn't find a physical solution")
                if hasattr(r, "__len__"):
                    qdic = []
                    for rval in r:
                        qvals = np.roots([1, 1, 0, -rval])
                        qdic.append(
                            qvals[np.logical_and(
                                np.isreal(qvals), qvals > 0)][0].real)
                    dic['q'] = np.asarray(qdic)
                else:
                    qvals = np.roots([1, 1, 0, -r])
                    dic['q'] = qvals[
                        np.logical_and(np.isreal(qvals), qvals > 0)][0].real
                return standard_mass_conversion(**dic)
            else:
                raise RuntimeError("Enough parameters weren't defined")
        else:
            raise RuntimeError("Enough parameters weren't defined")
    else:
        if m1 is not None:
            dic['m2'] = m1 * q
            dic['mtot'] = m1 * (1 + q)
            dic['mchirp'] = m1 * q**0.6 / (1 + q)**0.2
            return dic
        if mt is not None:
            dic['m1'] = mt / (1 + q)
            return standard_mass_conversion(**dic)
        if mc is not None:
            dic['mtot'] = mc / dic['eta']**0.6
            return standard_mass_conversion(**dic)
        raise RuntimeError("Enough parameters weren't defined")


#     Spin (and Mass) Variable Conversions
# --------------------------------------------

def chieff_chia_of_s1z_s2z_m1_m2(s1z, s2z, m1, m2):
    mt = m1 + m2
    return (m1 * s1z + m2 * s2z) / mt, (m1 * s1z - m2 * s2z) / mt

def chip(s1x, s1y, s2x, s2y, m1, m2):
    m1, m2 = np.maximum(m1, m2), np.minimum(m1, m2)
    q = m2 / m1
    # A1 = 2 + 1.5/q
    # A2 = 2 + 1.5*q
    A1 = 2 + 1.5 * q
    A2 = 2 + 1.5 / q
    s1P = np.sqrt(s1x ** 2 + s1y ** 2)
    s2P = np.sqrt(s2x ** 2 + s2y ** 2)
    return np.maximum(A1 * s1P * m1 ** 2,
                      A2 * s2P * m2 ** 2) / A1 / m1 ** 2

def sx_sy_of_s_theta_phi(s, theta, phi):
    return s * np.sin(theta) * np.cos(phi), s * np.sin(theta) * np.sin(phi)

def s1z_of_chieff_s2z_m1_m2(chieff, s2z, m1, m2):
    mt = m1 + m2
    return mt * (chieff - m2 * s2z / mt) / m1

def s2z_of_chieff_s1z_m1_m2(chieff, s1z, m1, m2):
    mt = m1 + m2
    return mt * (chieff - m1 * s1z / mt) / m2

def s1z_s2z_of_chieff_chia_m1_m2(chieff, chia, m1, m2):
    mt = m1 + m2
    return mt * (chieff + chia) / (2 * m1), mt * (chieff - chia) / (2 * m2)

def s1z_of_chia_s2z_m1_m2(chia, s2z, m1, m2):
    mt = m1 + m2
    return mt * (chia + m2 * s2z / mt) / m1

def s2z_of_chia_s1z_m1_m2(chia, s1z, m1, m2):
    mt = m1 + m2
    return mt * (m1 * s1z / mt - chia)


######################################################
#### PARAMETER COMPLETION for SAMPLE DATAFRAMES

def compute_samples_aux_vars(samples):
    """
    Takes a dict-like object with some set of masses, spins
    etc. and adds entries for derived quantities like chieff
    and source frame masses.
    Samples MUST have all the standard parameters
    """
    # mass ratio variables
    samples['q'] = np.asarray(samples['m2']) / np.asarray(samples['m1'])
    samples['q1'] = 1 / np.asarray(samples['q'])
    samples['lnq'] = np.log(np.asarray(samples['q']))
    samples['eta'] = q2eta(np.asarray(samples['q']))
    # mass variables
    samples['mtot'] = np.asarray(samples['m1']) + np.asarray(samples['m2'])
    samples['mchirp'] = mchirp_of_m1_m2(np.asarray(samples['m1']), np.asarray(samples['m2']))
    # source frame
    samples['z'] = cosmo.z_of_DL_Mpc(np.asarray(samples['d_luminosity']))
    samples['d_comoving'] = np.asarray(samples['d_luminosity']) / (1 + np.asarray(samples['z']))
    for k in ['m1', 'm2', 'mtot', 'mchirp']:
        samples[f'{k}_source'] = np.asarray(samples[k]) / (1 + np.asarray(samples['z']))
    # effective spin
    samples['chieff'] = ((np.asarray(samples['s1z']) + np.asarray(samples['q']) * np.asarray(samples['s2z']))
                             / (1 + np.asarray(samples['q'])))
    samples['chia'] = .5 * (np.asarray(samples['s1z']) - np.asarray(samples['s2z']))
    # spin polar variables
    for j in [1, 2]:
        sx, sy, sz = [np.asarray(samples[f's{j}{coord}']) for coord in ['x', 'y', 'z']]
        samples[f's{j}'] = np.sqrt(sx ** 2 + sy ** 2 + sz ** 2)
        samples[f's{j}costheta'] = sz / np.asarray(samples[f's{j}'])
        samples[f's{j}theta'] = np.arccos(np.asarray(samples[f's{j}costheta']))
        samples[f's{j}phi'] = np.arctan2(np.asarray(samples[f's{j}y']), np.asarray(samples[f's{j}x'])) % (2 * np.pi)
    # chi_p for precession
    samples['chip'] = chip(np.asarray(samples['s1x']), np.asarray(samples['s1y']),
                           np.asarray(samples['s2x']), np.asarray(samples['s2y']),
                           np.asarray(samples['m1']), np.asarray(samples['m2']))
    
    
#### COMPLETION

def complete_samples_intrinsic_params(samples):
    """
    Takes a dict-like object with some set of masses, spins
    etc. and adds entries for derived quantities like chieff,
    etc.
    Note: 'm1', 'm2', 'mchirp' are detector frame.
          'm1_source', etc. are source frame.
    """
    # checking consistency
    old_m1, old_m2, old_q, old_lnq, old_q1, old_eta = [None] * 6
    if 'm1' in samples:
        old_m1 = np.asarray(samples['m1']).copy()
    if 'm2' in samples:
        old_m2 = np.asarray(samples['m2']).copy()
    if 'lnq' in samples:
        old_lnq = np.asarray(samples['lnq']).copy()
    if 'q1' in samples:
        old_q1 = np.asarray(samples['q1']).copy()
    if 'eta' in samples:
        old_eta = np.asarray(samples['eta']).copy()

    # maybe need to get q before attempting masses
    if 'q' in samples:
        old_q = np.asarray(samples['q']).copy()
    else:
        if 'm1' in samples and 'm2' in samples:
            samples['q'] = np.asarray(samples['m2']) / np.asarray(samples['m1'])
        elif 'q1' in samples:
            samples['q'] = 1 / np.asarray(samples['q1'])
        elif 'lnq' in samples:
            # correct in case lnq was symmetrized
            samples['q'] = np.exp(-np.abs(np.asarray(samples['lnq'])))
        elif 'eta' in samples:
            samples['q'] = eta2q(np.asarray(samples['eta']))

    if 'mchirp' in samples and 'eta' in samples:
        samples['m1'] = m1_of_mchirp_eta(np.asarray(samples['mchirp']), np.asarray(samples['eta']))
        samples['m2'] = m2_of_mchirp_eta(np.asarray(samples['mchirp']), np.asarray(samples['eta']))
    elif 'mchirp' in samples and 'q' in samples:
        samples['m1'] = m1_of_mchirp_q(np.asarray(samples['mchirp']), np.asarray(samples['q']))
        samples['m2'] = m2_of_mchirp_q(np.asarray(samples['mchirp']), np.asarray(samples['q']))
    elif 'mtot' in samples and 'q' in samples:
        samples['m1'] = m1_of_mtot_q(np.asarray(samples['mtot']), np.asarray(samples['q']))
        samples['m2'] = m2_of_mtot_q(np.asarray(samples['mtot']), np.asarray(samples['q']))
    elif 'mchirp_source' in samples and 'eta' in samples:
        samples['m1_source'] = m1_of_mchirp_eta(np.asarray(samples['mchirp_source']), np.asarray(samples['eta']))
        samples['m2_source'] = m2_of_mchirp_eta(np.asarray(samples['mchirp_source']), np.asarray(samples['eta']))
    elif 'mchirp_source' in samples and 'q' in samples:
        samples['m1_source'] = m1_of_mchirp_q(np.asarray(samples['mchirp_source']), np.asarray(samples['q']))
        samples['m2_source'] = m2_of_mchirp_q(np.asarray(samples['mchirp_source']), np.asarray(samples['q']))
    elif 'mtot_source' in samples and 'q' in samples:
        samples['m1_source'] = m1_of_mtot_q(np.asarray(samples['mtot_source']), np.asarray(samples['q']))
        samples['m2_source'] = m2_of_mtot_q(np.asarray(samples['mtot_source']), np.asarray(samples['q']))

    # distance options...priority:  distance_Mpc > volume, distance_Gpc > redshift  [and luminosity > comoving]
    if ('d_luminosity' not in samples) and ('d_comoving' not in samples):
        if 'z' in samples:
            samples['d_luminosity'] = cosmo.DL_Mpc_of_z(np.asarray(samples['z']))
    # having made best effort to get d_luminosity or d_comoving, now get z and convert detector frame to source frame
    if ('d_luminosity' in samples) or ('d_comoving' in samples):
        if 'd_luminosity' in samples:
            samples['z'] = cosmo.z_of_DL_Mpc(np.asarray(samples['d_luminosity']))  # z_of_d_luminosity(np.asarray(samples['d_luminosity']))
            samples['d_comoving'] = np.asarray(samples['d_luminosity']) / (1 + np.asarray(samples['z']))
        elif 'd_comoving' in samples:
            samples['z'] = cosmo.z_of_Dcomov_Mpc(np.asarray(samples['d_comoving']))
            samples['d_luminosity'] = np.asarray(samples['d_comoving']) * (1 + np.asarray(samples['z']))
        
        if 'm1' in samples and (('m2' in samples) or ('q' in samples)):
            if 'm2' not in samples:
                samples['m2'] = np.asarray(samples['q']) * np.asarray(samples['m1'])
            samples['m1_source'] = np.asarray(samples['m1']) / (1 + np.asarray(samples['z']))
            samples['m2_source'] = np.asarray(samples['m2']) / (1 + np.asarray(samples['z']))
        elif 'm1_source' in samples and (('m2_source' in samples) or ('q' in samples)):
            if 'm2_source' not in samples:
                samples['m2_source'] = np.asarray(samples['q']) * np.asarray(samples['m1_source'])
            samples['m1'] = np.asarray(samples['m1_source']) * (1 + np.asarray(samples['z']))
            samples['m2'] = np.asarray(samples['m2_source']) * (1 + np.asarray(samples['z']))

        samples['mchirp_source'] = mchirp_of_m1_m2(np.asarray(samples['m1_source']), np.asarray(samples['m2_source']))
        samples['mtot_source'] = np.asarray(samples['m1_source']) + np.asarray(samples['m2_source'])

    samples['mchirp'] = mchirp_of_m1_m2(np.asarray(samples['m1']), np.asarray(samples['m2']))
    samples['mtot'] = np.asarray(samples['m1']) + np.asarray(samples['m2'])
    samples['q'] = np.asarray(samples['m2']) / np.asarray(samples['m1'])
    samples['lnq'] = np.log(samples['q'])
    samples['q1'] = 1 / np.asarray(samples['q'])
    samples['eta'] = q2eta(np.asarray(samples['q']))
    # check consistency
    if old_m1 is not None:
        if not np.allclose(old_m1, np.asarray(samples['m1'])):
            print('WARNING: MASS 1 INCONSISTENCY before and after CONVERSION')
    if old_m2 is not None:
        if not np.allclose(old_m2, np.asarray(samples['m2'])):
            print('WARNING: MASS 2 INCONSISTENCY before and after CONVERSION')
    if old_q is not None:
        if not np.allclose(old_q, np.asarray(samples['q'])):
            print('WARNING: MASS RATIO INCONSISTENCY before and after CONVERSION')
    if old_lnq is not None:
        if not np.allclose(old_lnq, np.asarray(samples['lnq'])):
            print('WARNING: LOG MASS RATIO INCONSISTENCY before and after CONVERSION')
            print('--> expected if samples are from PE with symmetrize_lnq=True')
    if old_q1 is not None:
        if not np.allclose(old_q1, np.asarray(samples['q1'])):
            print('WARNING: INVERSE MASS RATIO INCONSISTENCY before and after CONVERSION')
    if old_eta is not None:
        if not np.allclose(old_eta, np.asarray(samples['eta'])):
            print('WARNING: SYMMETRIC MASS RATIO INCONSISTENCY before and after CONVERSION')

    for j in [1, 2]:
        if f'spin{j}' in samples and f'costilt{j}' in samples:
            samples[f's{j}z'] = np.asarray(samples[f'spin{j}']) * np.asarray(samples[f'costilt{j}'])
        elif f's{j}' in samples:
            if f's{j}t' in samples:
                samples[f's{j}z'] = np.asarray(samples[f's{j}']) * np.cos(np.asarray(samples[f's{j}t']))
            elif f's{j}theta' in samples:
                samples[f's{j}z'] = np.asarray(samples[f's{j}']) * np.cos(np.asarray(samples[f's{j}theta']))
            elif f's{j}costheta' in samples:
                samples[f's{j}z'] = np.asarray(samples[f's{j}']) * np.asarray(samples[f's{j}costheta'])

    if 's1z' in samples and 's2z' in samples:
        samples['chieff'] = ((np.asarray(samples['s1z']) + np.asarray(samples['q']) * np.asarray(samples['s2z']))
                             / (1 + np.asarray(samples['q'])))
        samples['chia'] = .5 * (np.asarray(samples['s1z']) - np.asarray(samples['s2z']))
    elif 'chieff' in samples and 'q' in samples and 'chia' in samples:
        delta = (1 - np.asarray(samples['q'])) / (1 + np.asarray(samples['q']))
        chis = np.asarray(samples['chieff']) - delta * np.asarray(samples['chia'])
        samples['s1z'] = chis + np.asarray(samples['chia'])
        samples['s2z'] = chis - np.asarray(samples['chia'])
    for j in [1, 2]:
        sx, sy, sz = [samples.get(f's{j}{coord}', None) for coord in ['x', 'y', 'z']]
        if (sx is None) or (sy is None) or (sz is None):
            sj, thetaj, phij = [samples.get(k, None) for k in [f's{j}', f's{j}theta', f's{j}phi']]
            if thetaj is None:
                costhetakey = samples.get(f's{j}costheta', None)
                if costhetakey is not None:
                    thetaj = np.arccos(np.asarray(samples[costhetakey]))
                    samples[f's{j}theta'] = thetaj
            else:
                samples[f's{j}costheta'] = np.cos(thetaj)
            if (sj is not None) and (thetaj is not None) and (phij is not None):
                samples[f's{j}x'] = sj * np.cos(phij) * np.sin(thetaj)
                samples[f's{j}y'] = sj * np.sin(phij) * np.sin(thetaj)
                samples[f's{j}z'] = sj * np.cos(thetaj)
            else:
                print('WARNING: could not get all spin components from variables in samples')
        else:
            samples[f's{j}'] = np.sqrt(sx ** 2 + sy ** 2 + sz ** 2)
            samples[f's{j}costheta'] = sz / np.asarray(samples[f's{j}'])
            samples[f's{j}theta'] = np.arccos(np.asarray(samples[f's{j}costheta']))
            samples[f's{j}phi'] = np.arctan2(np.asarray(samples[f's{j}y']), np.asarray(samples[f's{j}x'])) % (2 * np.pi)

    if ('s1z' in samples) and ('s2z' in samples) and ('chieff' not in samples):
        samples['chieff'] = ((np.asarray(samples['s1z']) + np.asarray(samples['q']) * np.asarray(samples['s2z']))
                             / (1 + np.asarray(samples['q'])))
    if all(p in samples for p in ['s1x', 's1y', 's2x', 's2y', 'm1', 'm2']):
        samples['chip'] = chip(np.asarray(samples['s1x']), np.asarray(samples['s1y']),
                               np.asarray(samples['s2x']), np.asarray(samples['s2y']),
                               np.asarray(samples['m1']), np.asarray(samples['m2']))

