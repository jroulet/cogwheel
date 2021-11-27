import numpy as np

try:
    from gwtools.harmonics import sYlm
    # LAL version of sYlm, set to s = -2
    def lal_Y_lm(l, m, inclination, azimuth=0):
        return sYlm(-2, l, m, inclination, azimuth)
except:
    print('Unable to import module gwtools')
    lal_Y_lm = NotImplemented

# TESTED AGAINST LAL's ylm
def A_lm_inclin(l, m, iota):
    if l == 2:
        if m == 2:
            # (2, 2)
            return 0.5*np.sqrt(5/np.pi)*(np.cos(iota*0.5)**4)
        elif m == -2:
            # (2, -2)
            return 0.5*np.sqrt(5/np.pi)*(np.sin(iota*0.5)**4)
        elif m == 1:
            # (2, 1)
            return np.sqrt(5/np.pi)*(np.cos(iota*0.5)**3)*np.sin(iota*0.5)
        elif m == -1:
            # (2, -1)
            return 0.5*np.sqrt(5/np.pi)*(np.sin(iota*0.5)**2)*np.sin(iota)
        elif m == 0:
            # (2, 0)
            return np.sqrt(15/(32*np.pi))*(np.sin(iota)**2)
        else:
            raise RuntimeError(f'({l}, {m}) is NOT a supported mode!')
    elif l == 3:
        if m == 3:
            # (3, 3)
            return -np.sqrt(21/(2*np.pi))*(np.cos(iota*0.5)**5)*np.sin(iota*0.5)
        elif m == -3:
            # (3, -3)
            return np.sqrt(21/(8*np.pi))*(np.sin(iota*0.5)**4)*np.sin(iota)
        elif m == 2:
            # (3, 2)
            return np.sqrt(7/np.pi)*(np.cos(iota*0.5)**4)*(-1 + 1.5*np.cos(iota))
        elif m == -2:
            # (3, -2)
            return np.sqrt(7/np.pi)*(1 + 1.5*np.cos(iota))*(np.sin(iota*0.5)**4)
        elif m == 1:
            # (3, 1)
            return np.sqrt(35/(8*np.pi))*(np.cos(iota*0.5)**3)*(-1 + 3*np.cos(iota))*np.sin(iota*0.5)
        elif m == -1:
            # (3, -1)
            return np.sqrt(35/(8*np.pi))*np.cos(iota*0.5)*(1 + 3*np.cos(iota))*(np.sin(iota*0.5)**3)
        elif m == 0:
            # (3, 0)
            return np.sqrt(105/(32*np.pi))*np.cos(iota)*(np.sin(iota)**2)
        else:
            raise RuntimeError(f'({l}, {m}) is NOT a supported mode!')
    elif l == 4:
        if m == 4:
            # (4, 4)
            return 3*np.sqrt(7/np.pi)*(np.cos(iota*0.5)**6)*(np.sin(iota*0.5)**2)
        elif m == -4:
            # (4, -4)
            return 0.75*np.sqrt(7/np.pi)*(np.sin(iota*0.5)**4)*(np.sin(iota)**2)
        elif m == 3:
            # (4, 3)
            return -3*np.sqrt(7/(2.*np.pi))*(np.cos(iota*0.5)**5)*(-1 + 2*np.cos(iota))*np.sin(iota*0.5)
        elif m == -3:
            # (4, -3)
            return 3*np.sqrt(7/(2.*np.pi))*np.cos(iota*0.5)*(1 + 2*np.cos(iota))*(np.sin(iota*0.5)**5)
        elif m == 2:
            # (4, 2)
            return 0.75*(np.cos(iota*0.5)**4)*(9 - 14*np.cos(iota) + 7*np.cos(2*iota))/np.sqrt(np.pi)
        elif m == -2:
            # (4, -2)
            return 0.75*(9 + 14*np.cos(iota) + 7*np.cos(2*iota))*(np.sin(iota*0.5)**4)/np.sqrt(np.pi)
        elif m == 1:
            # (4, 1)
            return 3*(np.cos(iota*0.5)**3)*(6 - 7*np.cos(iota) + 7*np.cos(2*iota))*np.sin(iota*0.5)/np.sqrt(8*np.pi)
        elif m == -1:
            # (4, -1)
            return 3*np.cos(iota*0.5)*(6 + 7*np.cos(iota) + 7*np.cos(2*iota))*(np.sin(iota*0.5)**3)/np.sqrt(8*np.pi)
        elif m == 0:
            # (4, 0)
            return np.sqrt(45/(512*np.pi))*(5 + 7*np.cos(2*iota))*(np.sin(iota)**2)
        else:
            raise RuntimeError(f'({l}, {m}) is NOT a supported mode!')
    elif l == 5:
        if m == 5:
            # (5, 5)
            return -np.sqrt(330/np.pi)*(np.cos(iota*0.5)**7)*(np.sin(iota*0.5)**3)
        elif m == -5:
            # (5, -5)
            return np.sqrt(330/np.pi)*(np.cos(iota*0.5)**3)*(np.sin(iota*0.5)**7)
        else:
            raise RuntimeError(f'({l}, {m}) is NOT a supported mode!')
    else:
        raise RuntimeError(f'({l}, {m}) is NOT a supported mode!')

# TESTED AGAINST LAL's ylm
def Y_lm(l, m, iota, azim):
    """NOTE: azim = pi/2 - vphi"""
    return np.exp(1j * m * azim) * A_lm_inclin(l, m, iota)


def fplus_fcross_analytic(det_polar, det_azimuth, psi):
    """
    Get time-indep. geometric F+, Fx following https://arxiv.org/pdf/1102.5421.pdf
    where det_polar, det_azimuth are the polar and azimuthal angles of the
    line-of-sight in the frame whose x-y plane is defined by the detector arms
    with the z-axis pointing toward the sky.
    """
    costheta = np.cos(det_polar)
    part1 = 0.5 * (1 + (costheta ** 2)) * np.cos(2 * det_azimuth)
    part2 = costheta * np.sin(2 * det_azimuth)
    c2ps = np.cos(2 * psi)
    s2ps = np.sin(2 * psi)
    return part1 * c2ps - part2 * s2ps, part1 * s2ps + part2 * c2ps
