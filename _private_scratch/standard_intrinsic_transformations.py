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