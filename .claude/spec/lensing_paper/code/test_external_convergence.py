from __future__ import annotations
import numpy as np
from chang_refsdal_operator import chang_refsdal_amplification
from chang_refsdal_geometry import macro_matrix, find_images, lens_residual
from chang_refsdal_global_tracking import GlobalChannelTracker


def test_mass_sheet_amplification_identity():
    w=13.0; y=np.array([0.17,-0.08]); gamma=0.12; beta=0.31; kappa=0.27
    lam=1-kappa
    lhs=chang_refsdal_amplification(w,y,gamma,beta=beta,kappa=kappa,max_order=44,dps=80)
    rhs=(np.exp(0.5j*w*np.log(lam)-0.5j*w*kappa*(y@y)/lam)/lam
         *chang_refsdal_amplification(w,y/np.sqrt(lam),gamma/lam,beta=beta,kappa=0,max_order=44,dps=80))
    assert abs(lhs-rhs) < 2e-12*max(1,abs(rhs))


def test_mass_sheet_image_scaling():
    y=np.array([0.12,0.06]); gamma=0.1; beta=0.22; kappa=0.25; lam=1-kappa
    A=macro_matrix(gamma,beta,kappa)
    imgs=find_images(y,A)
    A0=macro_matrix(gamma/lam,beta,0.0)
    scaled=find_images(y/np.sqrt(lam),A0)
    expected=[z/np.sqrt(lam) for z in scaled]
    assert len(imgs)==len(expected)
    for x in imgs:
        assert min(np.linalg.norm(x-z) for z in expected) < 2e-9
        assert np.linalg.norm(lens_residual(x,y,A)) < 1e-9


def test_global_partition_with_convergence():
    w=np.linspace(5,30,9)
    tr=GlobalChannelTracker(w,operator_max_order=44,operator_dps=75)
    p=tr.evaluate(gamma=.12,kappa=.25,beta=.2,y=[.11,.04])
    assert p.reconstruction_error < 2e-12
    assert np.all(p.operator_converged)
