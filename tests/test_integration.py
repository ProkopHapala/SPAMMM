import pytest, numpy as np, os, datetime
from spammm.AtomicSystem import AtomicSystem
from tests.helpers.parity import overlay_plot, rmse, correlation
from tests.helpers.scan import compare_scans

def _debug_dir(name='integration'):
    d = os.path.join('debug', f'{datetime.date.today()}_{name}')
    os.makedirs(d, exist_ok=True)
    return d

@pytest.mark.gpu
@pytest.mark.slow
def test_relaxed_scan_water_nacl(xyz, substrate):
    """H2O on NaCl: relaxed scan z=[2,6] A, 20 pts. Smooth curve, min at 2.5-3.5 A."""
    # TODO: load H2O + NaCl, at each z height relax molecule, record energy
    # Check: energy curve smooth (no jumps), minimum in expected range
    pass

@pytest.mark.gpu
@pytest.mark.slow
def test_relaxed_scan_benzene_nacl(xyz, substrate):
    """Benzene on NaCl: relaxed scan. Min at 3.2-3.8 A (pi-stacking)."""
    pass

@pytest.mark.visual
@pytest.mark.gpu
@pytest.mark.slow
def test_visual_relaxed_scan(xyz, substrate):
    """Plot: E(z) rigid vs relaxed for water and benzene on NaCl."""
    save_dir = _debug_dir()
    # TODO: produce overlay plots of rigid vs relaxed scan
    pass
