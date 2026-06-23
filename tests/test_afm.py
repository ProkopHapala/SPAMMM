import pytest, numpy as np, os, datetime
from spammm.AtomicSystem import AtomicSystem
from tests.helpers.parity import overlay_plot, rmse, correlation

def _debug_dir(name='afm'):
    d = os.path.join('debug', f'{datetime.date.today()}_{name}')
    os.makedirs(d, exist_ok=True)
    return d

@pytest.mark.gpu
def test_afm_relax_convergence(xyz, substrate):
    """AFM probe relaxation on NaCl: finite forces, no NaN."""
    from spammm.SPM.AFM import AFMulator
    sub = AtomicSystem(fname=substrate('NaCl_1x1_L3.xyz'))
    tip = AtomicSystem(fname=xyz('CO.xyz'))
    afm = AFMulator()
    # TODO: setup afm with substrate + tip, run relaxation at a few points
    # Check: forces finite, no NaN, probe displacements reasonable
    pass

@pytest.mark.visual
@pytest.mark.gpu
def test_visual_afm_images(xyz, substrate):
    """2D AFM images at 3 z heights for NaCl."""
    from spammm.SPM.AFM import AFMulator
    save_dir = _debug_dir()
    # TODO: produce 2D AFM images at z = 3, 4, 5 A
    pass
