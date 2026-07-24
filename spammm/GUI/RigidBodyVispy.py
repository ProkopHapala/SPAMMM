"""Vispy+PyQt5 visualization for RigidBodyPairFF molecular dynamics.

Architecture
------------
This is a **thin viewer** — it holds no physics state. All dynamics live on the
GPU in the RigidBodyPairFF OpenCL kernels (rigid.cl). The viewer's job is:
  1. Download atom positions each frame (one clEnqueueReadBuffer per frame)
  2. Upload anchor spring targets when the user drags atoms (mouse picking)
  3. Recompute a 2D potential map on the CPU when FF parameters change

The potential map is computed on CPU (not GPU) because it is a 2D grid scan
that changes infrequently (only on parameter edit), and doing it on CPU avoids
allocating a separate GPU kernel + buffer management for a ~100×100 grid.

Mouse picking uses **ray-nearest-point** (not ray-plane intersection) for the
anchor target. This ensures the spring force is always perpendicular to the
view ray, so dragging an atom in an orthographic top-down view produces a force
in the XY plane regardless of the atom's z-coordinate. See _nearest_point_on_ray.

Camera: TurntableCamera with fov=0 (orthographic), elevation=90 (top-down).
This gives a true XY orthographic projection. Mouse wheel adjusts scale_factor
(exponential zoom), arrow keys pan the camera center.

Features:
  - PyQt5 side panel: FF params, kernel mode, compact map probe (H+/O− presets + R0/E0/Q),
    dt, steps/frame
  - Orthographic top-down (XY) camera, white background
  - Mouse wheel zoom, arrow key pan
  - LMB click+drag on dynamic atoms for anchor spring picking
  - Multi-body: LMB click on any molecule (dyn or static) selects it as active —
    that body becomes mobile; others become frozen env; potential map recomputed
  - Potential map background: Morse + epair Hbond + sigma-hole (no Coulomb),
    matching ff_map.py's morseH_only mode. Presets H+(q=+0.4) / O−(q=-0.4);
    when ``rbd.faf_mode`` + ``rbd.faf_fit``, map is **PairFF + FAF** at active CoM z.
    R0, E0, Q editable; element combo still fills R0/E0 from AtomTypes.
  - Epairs rendered as cyan dots, sigma holes as magenta dots (same size, semi-transparent)
  - Faint dummy-bond lines from epairs (cyan) and sigma holes (magenta) to host atoms
  - Real atom bonds rendered as independent line segments (connect='segments', not strip)
  - Changing any FF parameter triggers live map recompute via paramsChanged signal
  - SPACE = run/stop, R = reset, F = FIRE (default ON), ESC = quit

Caveats:
  - Potential map does NOT include Coulomb (matches ff_map.py 'morseH_only' mode).
    The GPU kernel does include Coulomb, so the map is an approximation of the
    actual forces for visualization purposes only.
  - Bond computation uses a fixed 1.8 Å cutoff — may miss long bonds or include
    spurious ones for non-standard geometries.
  - Map plane z = active CoM (or ``rbd.map_z``); with ``faf_mode`` the map is
    PairFF(static) + FAF(probe). Without env molecules, PairFF layer is empty
    and only FAF (if enabled) fills the background.
"""

import numpy as np
from vispy import app, scene, gloo
from vispy.scene import visuals
import sys
import os

# Add repo root for imports
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from PyQt5 import QtWidgets, QtCore, QtGui
from spammm.forcefields.RigidBodyDynamics import _body_sites_world

# Element colors (CPK-like, RGBA)
ELEM_COLORS = {
    'H': (0.9, 0.9, 0.9, 1.0),
    'C': (0.2, 0.2, 0.2, 1.0),
    'N': (0.2, 0.2, 0.8, 1.0),
    'O': (0.8, 0.2, 0.2, 1.0),
    'F': (0.2, 0.8, 0.2, 1.0),
    'S': (0.8, 0.8, 0.2, 1.0),
    'Cl': (0.1, 0.6, 0.1, 1.0),
    'E':  (0.0, 0.8, 0.8, 0.7),   # cyan, semi-transparent
    'Sh': (0.8, 0.0, 0.8, 0.7),   # magenta, semi-transparent
}
ELEM_RADII = {
    'H': 0.25, 'C': 0.35, 'N': 0.35, 'O': 0.35, 'F': 0.30, 'S': 0.40, 'Cl': 0.40, 'E': 0.15, 'Sh': 0.15,
}

COULOMB_CONST = 14.3996448915
MORSE_BETA = 1.7


# ==================================================================
#  Potential map computation (simplified from ff_map.py)
# ==================================================================

def compute_potential_map(static_apos, static_REQ, static_enames, static_types,
                          probe_R0, probe_E0, probe_q, z_height=0.0, margin=4.0, step=0.1,
                          He=-1.0, Hs=0.0, rc=3.0, w=1.0, atom_types=None):
    """Compute 2D Morse + Hbond + SigmaHole energy map (no Coulomb).

    Probe specified by R0, E0 (EvdW depth), Q. Mixing uses e=sqrt(E0).
    """
    apos = np.asarray(static_apos, dtype=np.float64)

    xmin, ymin = apos[:, 0].min() - margin, apos[:, 1].min() - margin
    xmax, ymax = apos[:, 0].max() + margin, apos[:, 1].max() + margin
    xs = np.arange(xmin, xmax + step, step)
    ys = np.arange(ymin, ymax + step, step)
    nx, ny = len(xs), len(ys)

    probe_REQ = np.array([float(probe_R0), np.sqrt(max(float(probe_E0), 0.0)),
                          float(probe_q), 0.0], dtype=np.float64)

    # Mixed REQs (arithmetic R, geometric E, product Q, product H)
    REQs = np.asarray(static_REQ, dtype=np.float64)
    REQs_mixed = np.empty_like(REQs)
    REQs_mixed[:, 0] = probe_REQ[0] + REQs[:, 0]
    REQs_mixed[:, 1] = probe_REQ[1] * REQs[:, 1]
    REQs_mixed[:, 2] = probe_REQ[2] * REQs[:, 2]
    REQs_mixed[:, 3] = probe_REQ[3] * REQs[:, 3]
    REQs_mixed[REQs_mixed[:, 3] > 0, 3] = 0.0  # H <= 0

    X, Y = np.meshgrid(xs, ys)

    # --- Atom-atom: Morse only (no Coulomb), only real atoms (type=0) ---
    atom_mask = np.array([t == 0 for t in static_types])
    atom_idx = np.where(atom_mask)[0]
    if len(atom_idx) > 0:
        apos_atoms = apos[atom_idx]
        dx = X[:, :, None] - apos_atoms[None, None, :, 0]
        dy = Y[:, :, None] - apos_atoms[None, None, :, 1]
        dz = z_height - apos_atoms[None, None, :, 2]
        r2 = dx*dx + dy*dy + dz*dz
        r = np.sqrt(r2)
        e_exp = np.exp(-MORSE_BETA * (r - REQs_mixed[None, None, atom_idx, 0]))
        E_morse = REQs_mixed[None, None, atom_idx, 1] * (e_exp*e_exp - 2.0*e_exp)
        Emap = E_morse.sum(axis=2)
    else:
        Emap = np.zeros((ny, nx))

    # --- Hbond correction from epairs (type=1): coeff = min(0, probe_q * He) ---
    ep_mask = np.array([t == 1 for t in static_types])
    ep_indices = np.where(ep_mask)[0]
    if len(ep_indices) > 0 and He != 0:
        ep_pos = apos[ep_indices]
        dx_ep = X[:, :, None] - ep_pos[None, None, :, 0]
        dy_ep = Y[:, :, None] - ep_pos[None, None, :, 1]
        dz_ep = z_height - ep_pos[None, None, :, 2]
        r_ep = np.sqrt(dx_ep*dx_ep + dy_ep*dy_ep + dz_ep*dz_ep)
        x_cut = np.clip(1.0 - r_ep/rc, 0, 1)
        fcut = 3*x_cut**2 - 2*x_cut**3
        lorenc = 1.0 / (w*w + r_ep*r_ep)
        coeff = min(0.0, probe_q * He)
        Emap += coeff * (fcut * lorenc).sum(axis=2)

    # --- Sigma-hole correction (type=2): coeff = min(0, probe_q * Hs) ---
    sh_mask = np.array([t == 2 for t in static_types])
    sh_indices = np.where(sh_mask)[0]
    if len(sh_indices) > 0 and Hs != 0:
        sh_pos = apos[sh_indices]
        dx_s = X[:, :, None] - sh_pos[None, None, :, 0]
        dy_s = Y[:, :, None] - sh_pos[None, None, :, 1]
        dz_s = z_height - sh_pos[None, None, :, 2]
        r_s = np.sqrt(dx_s*dx_s + dy_s*dy_s + dz_s*dz_s)
        x_cut_s = np.clip(1.0 - r_s/rc, 0, 1)
        fcut_s = 3*x_cut_s**2 - 2*x_cut_s**3
        lorenc_s = 1.0 / (w*w + r_s*r_s)
        coeff_s = min(0.0, probe_q * Hs)
        Emap += coeff_s * (fcut_s * lorenc_s).sum(axis=2)

    extent = [xmin, xmax, ymin, ymax]
    return Emap, xs, ys, extent


def _load_compact_exp():
    """Import compact-exp helpers from fit_radial.py (shared reference)."""
    import importlib.util
    path = os.path.join(REPO_ROOT, 'examples', 'density_comparison', 'HBondFF', 'fit_radial.py')
    spec = importlib.util.spec_from_file_location('fit_radial_nbff', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.compact_exp_force_over_r, float(getattr(mod, 'MORSE_BETA', MORSE_BETA))


def compute_potential_map_unified(static_apos, static_REQ, static_enames, static_types,
                                  probe_R0, probe_E0, probe_q, z_height=0.0, margin=4.0, step=0.1,
                                  He=-1.0, Hs=0.0, w=0.7, beta=1.7, atom_types=None):
    """2D map for unified compact-exp PairFF (no Coulomb), matching GPU mixing."""
    compact_EF, _ = _load_compact_exp()
    apos = np.asarray(static_apos, dtype=np.float64)
    REQs = np.asarray(static_REQ, dtype=np.float64).copy()
    types = np.asarray(static_types, dtype=np.int32)
    for i, t in enumerate(types):
        if t == 1:
            REQs[i, 2] = He
            REQs[i, 3] = w
        elif t == 2:
            REQs[i, 2] = Hs
            REQs[i, 3] = w
        else:
            REQs[i, 3] = 0.0

    xmin, ymin = apos[:, 0].min() - margin, apos[:, 1].min() - margin
    xmax, ymax = apos[:, 0].max() + margin, apos[:, 1].max() + margin
    xs = np.arange(xmin, xmax + step, step)
    ys = np.arange(ymin, ymax + step, step)
    X, Y = np.meshgrid(xs, ys)
    Emap = np.zeros_like(X, dtype=np.float64)

    probe_R = float(probe_R0)
    probe_e = float(np.sqrt(max(float(probe_E0), 0.0)))
    gi, wi, Qi = 1.0, 0.0, float(probe_q)

    for j in range(len(types)):
        gj = 1.0 if types[j] == 0 else 0.0
        gij = gi * gj
        R0 = gij * (probe_R + REQs[j, 0])
        wij = wi + REQs[j, 3]
        alpha = gij
        attr = -min(0.0, Qi * REQs[j, 2])
        both_dummy = 1.0 - min(gi + gj, 1.0)
        E0 = (attr if gij < 0.5 else probe_e * REQs[j, 1]) * (1.0 - both_dummy)
        if E0 == 0.0:
            continue
        dx = X - apos[j, 0]
        dy = Y - apos[j, 1]
        dz = z_height - apos[j, 2]
        r = np.sqrt(dx * dx + dy * dy + dz * dz)
        V, _ = compact_EF(r, R0, E0, beta, power=8, alpha=alpha, w=wij, soft_kind='sqrt')
        Emap += V

    return Emap, xs, ys, [xmin, xmax, ymin, ymax]


def potential_to_rgba(Emap, vmin=None, vmax=None):
    """Convert energy map to RGBA using RdBu_r colormap (same as ff_map.py).
    Symmetric scale: vmax = max(|Emin|, 0.01), vmin = -vmax.
    Repulsion in atom centers is oversaturated (clipped) — that's OK."""
    import matplotlib.cm as cm
    vmax = max(abs(Emap.min()), 0.01)
    vmin = -vmax
    Enorm = np.clip((Emap - vmin) / (vmax - vmin + 1e-12), 0, 1)
    rgba = cm.get_cmap('RdBu_r')(Enorm).astype(np.float32)
    rgba[..., 3] = 0.6  # semi-transparent
    return rgba


# ==================================================================
#  PyQt5 Control Panel
# ==================================================================

class ControlPanel(QtWidgets.QWidget):
    """Side panel with parameter controls for RigidBodyPairFF.

    Design: all spinboxes emit paramsChanged → main viewer recomputes map +
    re-inits the GPU kernel. This is a full re-init (not incremental) because
    changing e.g. morse_alpha requires recompiling kernel args. The cost is
    negligible (~1ms) since it only fires on user interaction, not per-frame.

    The 'epair_dist' and 'sigma_dist' spins update host-side params but do NOT
    reposition existing epairs/sigma-holes on the GPU — they affect only the
    next from_two_molecules() call. This is a known limitation (see open issues).
    """

    paramsChanged = QtCore.pyqtSignal()

    def __init__(self, rbd, atom_types=None, parent=None):
        super().__init__(parent)
        self.rbd = rbd
        self.atom_types = atom_types or {}
        self.setWindowTitle("PairFF Controls")
        self.setFixedWidth(260)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        # --- Simulation controls ---
        grp_sim = QtWidgets.QGroupBox("Simulation")
        gl = QtWidgets.QGridLayout(grp_sim)
        gl.setSpacing(2)

        self.btn_run = QtWidgets.QPushButton("▶ Run")
        self.btn_run.setCheckable(True)
        self.btn_run.setChecked(True)
        gl.addWidget(self.btn_run, 0, 0)
        self.btn_reset = QtWidgets.QPushButton("Reset V")
        gl.addWidget(self.btn_reset, 0, 1)
        self.btn_fire = QtWidgets.QPushButton("FIRE")
        self.btn_fire.setCheckable(True)
        self.btn_fire.setChecked(True)  # default: FIRE (snappy relax); toggle off for damped MD
        gl.addWidget(self.btn_fire, 0, 2)

        self.lbl_status = QtWidgets.QLabel("FIRE")
        self.lbl_status.setStyleSheet("font-weight: bold; color: green;")
        gl.addWidget(self.lbl_status, 1, 0, 1, 3)

        self.lbl_E = QtWidgets.QLabel("E=0.0  |F|=0.0")
        gl.addWidget(self.lbl_E, 2, 0, 1, 3)

        self.lbl_active = QtWidgets.QLabel("Active: 0")
        self.lbl_active.setToolTip("Click any molecule to select it as the mobile body")
        gl.addWidget(self.lbl_active, 3, 0, 1, 3)

        gl.addWidget(QtWidgets.QLabel("Kernel:"), 4, 0)
        self.cmb_mode = QtWidgets.QComboBox()
        self.cmb_mode.addItem("Legacy (Morse+Lorentz)", "legacy")
        self.cmb_mode.addItem("Unified (compact exp)", "unified")
        cur_mode = (rbd.pairff_params_host or {}).get('mode', getattr(rbd, 'pairff_mode', 'legacy'))
        idx = self.cmb_mode.findData(cur_mode)
        if idx >= 0:
            self.cmb_mode.setCurrentIndex(idx)
        self.cmb_mode.currentIndexChanged.connect(self.on_param_changed)
        gl.addWidget(self.cmb_mode, 4, 1, 1, 2)

        layout.addWidget(grp_sim)

        # --- FF Parameters ---
        grp_ff = QtWidgets.QGroupBox("FF Parameters")
        fl = QtWidgets.QGridLayout(grp_ff)
        fl.setSpacing(2)

        self.spins = {}
        def add_spin(label, key, val, vmin=-100, vmax=100, step=0.01, fmt='%.4f'):
            row = len(self.spins)
            fl.addWidget(QtWidgets.QLabel(label), row, 0)
            sp = QtWidgets.QDoubleSpinBox()
            sp.setRange(vmin, vmax)
            sp.setSingleStep(step)
            sp.setDecimals(4)
            sp.setValue(val)
            sp.valueChanged.connect(self.on_param_changed)
            fl.addWidget(sp, row, 1)
            self.spins[key] = sp

        p = rbd.pairff_params_host or {}
        add_spin("He",      'He',      p.get('He', -1.0),   -10, 10, 0.01)
        add_spin("Hs",      'Hs',      p.get('Hs', 1.0),    -10, 10, 0.01)
        add_spin("rc",      'rc',      p.get('rc', 3.0),     0, 20, 0.1)
        add_spin("w",       'w',       p.get('w', 1.0),      0, 10, 0.05)
        add_spin("k_z",     'k_z',     p.get('k_z', 5.0),    0, 100, 0.5)
        add_spin("alpha",   'alpha',   p.get('morse_alpha', 1.8), 0, 10, 0.1)
        add_spin("z_target",'z_target',p.get('z_target', 0.0), -20, 20, 0.1)
        add_spin("L_epair", 'epair_dist', p.get('epair_dist', 1.4), 0, 5, 0.1)
        add_spin("L_sigma", 'sigma_dist', p.get('sigma_dist', 1.0), 0, 5, 0.1)
        add_spin("dt",      'dt',      0.02,  0, 1, 0.005)
        add_spin("steps/frame", 'spf', 10,    1, 500, 1)
        self.spins['spf'].setDecimals(0)

        layout.addWidget(grp_ff)

        # --- Probe for potential map (compact) ---
        grp_probe = QtWidgets.QGroupBox("Map probe")
        pl = QtWidgets.QGridLayout(grp_probe)
        pl.setContentsMargins(4, 4, 4, 4)
        pl.setSpacing(2)
        pl.setHorizontalSpacing(3)

        def _tiny_spin(vmin, vmax, step, decimals, val, maxw=56):
            sp = QtWidgets.QDoubleSpinBox()
            sp.setRange(vmin, vmax)
            sp.setSingleStep(step)
            sp.setDecimals(decimals)
            sp.setValue(val)
            sp.setMaximumWidth(maxw)
            sp.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
            sp.setAlignment(QtCore.Qt.AlignRight)
            return sp

        # Row 0: H+ | O− | type | map | ↻
        self.btn_probe_Hp = QtWidgets.QPushButton("H+")
        self.btn_probe_Om = QtWidgets.QPushButton("O−")
        for b in (self.btn_probe_Hp, self.btn_probe_Om):
            b.setCheckable(True)
            b.setFixedHeight(22)
            b.setMaximumWidth(36)
        self.btn_probe_Hp.setChecked(True)
        self.btn_probe_Hp.setToolTip("H donor  q=+0.4e")
        self.btn_probe_Om.setToolTip("O acceptor  q=-0.4e")
        pl.addWidget(self.btn_probe_Hp, 0, 0)
        pl.addWidget(self.btn_probe_Om, 0, 1)

        self.cmb_probe = QtWidgets.QComboBox()
        self.cmb_probe.addItems(['H', 'C', 'N', 'O', 'F', 'S', 'Cl'])
        self.cmb_probe.setCurrentText('H')
        self.cmb_probe.setMaximumWidth(48)
        self.cmb_probe.setToolTip("Element → fills R0, E0")
        pl.addWidget(self.cmb_probe, 0, 2)

        self.chk_map = QtWidgets.QCheckBox("map")
        self.chk_map.setChecked(True)
        pl.addWidget(self.chk_map, 0, 3)

        self.btn_recompute = QtWidgets.QPushButton("↻")
        self.btn_recompute.setFixedSize(28, 22)
        self.btn_recompute.setToolTip("Recompute map")
        pl.addWidget(self.btn_recompute, 0, 4)

        # Row 1: R0 E0 Q
        self.spin_probe_R0 = _tiny_spin(0.0, 5.0, 0.01, 3, 1.443, maxw=50)
        self.spin_probe_E0 = _tiny_spin(0.0, 1.0, 1e-4, 5, 0.00191, maxw=56)
        self.spin_probe_q = _tiny_spin(-5.0, 5.0, 0.05, 2, 0.40, maxw=42)
        row_req = QtWidgets.QHBoxLayout()
        row_req.setSpacing(2)
        row_req.setContentsMargins(0, 0, 0, 0)
        for lbl, sp in (("R0", self.spin_probe_R0), ("E0", self.spin_probe_E0), ("Q", self.spin_probe_q)):
            row_req.addWidget(QtWidgets.QLabel(lbl))
            row_req.addWidget(sp)
        row_req.addStretch(1)
        pl.addLayout(row_req, 1, 0, 1, 5)

        layout.addWidget(grp_probe)

        # --- Anchor spring ---
        grp_anchor = QtWidgets.QGroupBox("Anchor Spring")
        al = QtWidgets.QGridLayout(grp_anchor)
        al.setSpacing(2)
        al.addWidget(QtWidgets.QLabel("k_anchor:"), 0, 0)
        self.spin_anchor_k = QtWidgets.QDoubleSpinBox()
        self.spin_anchor_k.setRange(0, 100)
        self.spin_anchor_k.setValue(5.0)
        self.spin_anchor_k.setSingleStep(0.5)
        al.addWidget(self.spin_anchor_k, 0, 1)

        layout.addWidget(grp_anchor)
        layout.addStretch()

        # Connect
        self.btn_reset.clicked.connect(self.on_reset)
        self.btn_fire.clicked.connect(self.on_fire_toggle)
        self.btn_recompute.clicked.connect(self.on_recompute_map)
        self.btn_probe_Hp.clicked.connect(lambda: self.on_probe_preset('Hp'))
        self.btn_probe_Om.clicked.connect(lambda: self.on_probe_preset('Om'))
        self.cmb_probe.currentTextChanged.connect(self.on_probe_element_changed)
        self.spin_probe_R0.valueChanged.connect(self.on_probe_manual_edit)
        self.spin_probe_E0.valueChanged.connect(self.on_probe_manual_edit)
        self.spin_probe_q.valueChanged.connect(self.on_probe_manual_edit)
        self.chk_map.stateChanged.connect(self.on_recompute_map)

        self._apply_probe_from_element('H', q=0.4, preset='Hp')

    def _set_probe_spins(self, R0, E0, Q):
        for sp in (self.spin_probe_R0, self.spin_probe_E0, self.spin_probe_q):
            sp.blockSignals(True)
        self.spin_probe_R0.setValue(float(R0))
        self.spin_probe_E0.setValue(float(E0))
        self.spin_probe_q.setValue(float(Q))
        for sp in (self.spin_probe_R0, self.spin_probe_E0, self.spin_probe_q):
            sp.blockSignals(False)

    def _apply_probe_from_element(self, ename, q=None, preset=None):
        """Fill R0/E0 from AtomTypes; optionally set Q and preset button state."""
        R0, E0 = 1.5, 0.002
        if ename in self.atom_types:
            at = self.atom_types[ename]
            R0, E0 = float(at.RvdW), float(at.EvdW)
        if q is None:
            q = self.spin_probe_q.value()
        self.cmb_probe.blockSignals(True)
        self.cmb_probe.setCurrentText(ename)
        self.cmb_probe.blockSignals(False)
        self._set_probe_spins(R0, E0, q)
        if preset == 'Hp':
            self.btn_probe_Hp.setChecked(True)
            self.btn_probe_Om.setChecked(False)
        elif preset == 'Om':
            self.btn_probe_Hp.setChecked(False)
            self.btn_probe_Om.setChecked(True)
        elif preset is None:
            pass
        else:
            self.btn_probe_Hp.setChecked(False)
            self.btn_probe_Om.setChecked(False)

    def on_probe_preset(self, which):
        if which == 'Hp':
            self._apply_probe_from_element('H', q=0.4, preset='Hp')
        else:
            self._apply_probe_from_element('O', q=-0.4, preset='Om')
        self.on_recompute_map()

    def on_probe_element_changed(self, ename):
        self._apply_probe_from_element(ename, q=self.spin_probe_q.value(), preset=None)
        self.btn_probe_Hp.setChecked(False)
        self.btn_probe_Om.setChecked(False)
        self.on_recompute_map()

    def on_probe_manual_edit(self):
        self.btn_probe_Hp.setChecked(False)
        self.btn_probe_Om.setChecked(False)
        self.on_recompute_map()

    def on_param_changed(self):
        rbd = self.rbd
        He = self.spins['He'].value()
        Hs = self.spins['Hs'].value()
        rc = self.spins['rc'].value()
        w = self.spins['w'].value()
        k_z = self.spins['k_z'].value()
        alpha = self.spins['alpha'].value()
        z_target = self.spins['z_target'].value()
        epair_dist = self.spins['epair_dist'].value()
        sigma_dist = self.spins['sigma_dist'].value()
        mode = self.cmb_mode.currentData()
        beta = (rbd.pairff_params_host or {}).get('beta', 1.7)
        rbd.init_pairff(He=He, rc=rc, w=w, k_z=k_z, morse_alpha=alpha, z_target=z_target,
                        Hs=Hs, epair_dist=epair_dist, sigma_dist=sigma_dist, mode=mode, beta=beta)
        self.paramsChanged.emit()

    def get_pairff_mode(self):
        return self.cmb_mode.currentData()

    def on_reset(self):
        z = np.zeros((self.rbd.n_bodies, 4), dtype=np.float32)
        self.rbd.toGPU('vposs', z)
        self.rbd.toGPU('vrots', z)
        self.rbd.reset_optimizer_state()

    def on_fire_toggle(self):
        self.lbl_status.setText("FIRE" if self.btn_fire.isChecked() else "RUNNING")

    def on_recompute_map(self):
        self.paramsChanged.emit()

    def get_probe_params(self):
        """Return (R0, E0, Q) for the potential-map probe."""
        return (self.spin_probe_R0.value(), self.spin_probe_E0.value(), self.spin_probe_q.value())

    def get_dt(self):
        return self.spins['dt'].value()

    def get_steps_per_frame(self):
        return int(self.spins['spf'].value())

    def get_anchor_k(self):
        return self.spin_anchor_k.value()

    def is_running(self):
        return self.btn_run.isChecked()

    def is_fire(self):
        return self.btn_fire.isChecked()

    def show_map(self):
        return self.chk_map.isChecked()

    def set_active_label(self, body_id, n_bodies=None):
        if n_bodies is None:
            self.lbl_active.setText(f"Active: {body_id}")
        else:
            self.lbl_active.setText(f"Active: {body_id}/{n_bodies}")


# ==================================================================
#  Main Vispy+PyQt5 Window
# ==================================================================

class RigidBodyVispy:
    """Interactive Vispy+PyQt5 visualization for RigidBodyPairFF.

    Orthographic XY top-down view, white background.
    LMB click+drag on dynamic atoms activates anchor springs.
    Mouse wheel = zoom, Arrow keys = pan, SPACE = run/stop.

    Z-ordering: map_image (bottom) → static bonds → static dummy bonds →
    static markers → dyn bonds → dyn dummy bonds → dyn markers → anchor line →
    anchor marker (top). This ensures dynamic atoms are always visible above
    the potential map and static molecule.

    Frame loop: 60 Hz timer → run_pairff(N steps) → download positions →
    update visuals. The GPU runs N=steps_per_frame MD steps per timer tick,
    so effective simulation speed = N * 60 steps/sec. With dt=0.02 and N=10,
    that's 12 ps/sec of simulated time.
    """

    def __init__(self, rbd, dt=0.02, steps_per_frame=10, fire=True, title="RigidBodyPairFF"):
        self.rbd = rbd
        self.dt = dt
        self.steps_per_frame = steps_per_frame
        self.fire = fire
        self.dragging = False
        self.drag_atom = -1
        self.running = True
        self.frame_count = 0
        self._map_data = None  # cached potential map

        # --- Load atom types for potential map probe REQ ---
        from spammm.topology.FFparams import read_element_types, read_atom_types
        data_path = os.path.join(REPO_ROOT, 'data')
        etypes = read_element_types(os.path.join(data_path, 'ElementTypes.dat'))
        self._atom_types = read_atom_types(os.path.join(data_path, 'AtomTypes.dat'), etypes)

        # --- Camera params (from AtomScene) ---
        self._cam_zoom_speed = 0.12
        self._cam_zoom_min = 1e-4
        self._cam_zoom_max = 1e+4
        self._cam_pan_speed = 1.0

        # --- Build Qt app + window ---
        self.qt_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
        self.main_win = QtWidgets.QMainWindow()
        self.main_win.setWindowTitle(title)
        central = QtWidgets.QWidget()
        self.main_win.setCentralWidget(central)
        hlayout = QtWidgets.QHBoxLayout(central)
        hlayout.setContentsMargins(0, 0, 0, 0)
        hlayout.setSpacing(0)

        # --- Vispy canvas ---
        self.canvas = scene.SceneCanvas(keys='interactive', bgcolor='white', show=False,
                                        size=(800, 800))
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.TurntableCamera(fov=0, distance=15.0, elevation=90, azimuth=0)
        self.view.camera.interactive = False

        # Set range from all atom positions
        all_pos = self._get_all_positions()
        if all_pos.shape[0] > 0:
            margin = 3.0
            self.view.camera.set_range(
                (all_pos[:, 0].min() - margin, all_pos[:, 0].max() + margin),
                (all_pos[:, 1].min() - margin, all_pos[:, 1].max() + margin),
                (-5, 5))

        # --- Visuals (z-ordered like AtomScene) ---
        self.map_image = visuals.Image(parent=self.view.scene)
        self.map_image.set_gl_state('translucent', depth_test=False)
        self.static_bonds = visuals.Line(parent=self.view.scene, color=(0.3, 0.3, 0.3, 0.5), width=1.5, antialias=True, method='gl', connect='segments')
        self.dyn_bonds = visuals.Line(parent=self.view.scene, color=(0.2, 0.2, 0.8, 0.7),   width=2.0, antialias=True, method='gl', connect='segments')
        self.static_markers = visuals.Markers(parent=self.view.scene)
        self.dyn_markers = visuals.Markers(parent=self.view.scene)
        self.static_dummy_bonds = visuals.Line(parent=self.view.scene, color=(0.0, 0.8, 0.8, 0.3), width=1.0, antialias=True, method='gl', connect='segments')
        self.dyn_dummy_bonds = visuals.Line(parent=self.view.scene, color=(0.0, 0.8, 0.8, 0.3), width=1.0, antialias=True, method='gl', connect='segments')
        self.anchor_line = visuals.Line(parent=self.view.scene, color=(1, 0, 0, 0.8),   width=2.0, antialias=True, method='gl', connect='segments')
        self.anchor_marker = visuals.Markers(parent=self.view.scene)

        for v in (self.static_markers, self.dyn_markers, self.anchor_marker): v.set_gl_state('translucent', depth_test=False)
        for v in (self.static_bonds, self.dyn_bonds, self.static_dummy_bonds, self.dyn_dummy_bonds, self.anchor_line): v.set_gl_state('translucent', depth_test=False)
        for o, v in enumerate((self.map_image, self.static_bonds, self.static_dummy_bonds, self.static_markers, self.dyn_bonds, self.dyn_dummy_bonds, self.dyn_markers, self.anchor_line, self.anchor_marker)):
            if hasattr(v, 'order'): v.order = o

        # --- Build visuals ---
        self._build_static()
        self._build_dynamic()

        # --- Qt canvas widget ---
        canvas_native = self.canvas.native
        hlayout.addWidget(canvas_native, 1)

        # --- Control panel ---
        self.panel = ControlPanel(rbd, atom_types=self._atom_types)
        self.panel.btn_run.clicked.connect(self.on_run_toggle)
        self.panel.paramsChanged.connect(self._recompute_map)
        self.panel.btn_fire.setChecked(bool(self.fire))
        self.panel.lbl_status.setText("FIRE" if self.fire else "RUNNING")
        hlayout.addWidget(self.panel)
        self._update_active_label()

        # --- Recompute potential map (needs panel) ---
        self._recompute_map()

        # --- Timer ---
        self.timer = app.Timer(interval=1.0 / 60.0, connect=self.on_timer, start=True)

        # --- Connect vispy events ---
        self.canvas.events.mouse_press.connect(self.on_mouse_press)
        self.canvas.events.mouse_release.connect(self.on_mouse_release)
        self.canvas.events.mouse_move.connect(self.on_mouse_move)
        self.canvas.events.mouse_wheel.connect(self.on_mouse_wheel)
        self.canvas.events.key_press.connect(self.on_key_press)

        self.main_win.show()

    # --- Position helpers ---

    def _get_all_positions(self):
        pts = []
        if self.rbd.static_n > 0:
            pts.append(self.rbd.static_apos_host[:, :3])
        out = self.rbd.download_outputs()
        pts.append(out['atom_positions'][0, :, :3])
        return np.vstack(pts).astype(np.float32)

    def _get_dyn_pos(self):
        out = self.rbd.download_outputs()
        return out['atom_positions'][0, :, :3].astype(np.float32)

    # --- Color/size helpers ---

    def _get_colors_sizes(self, enames, types):
        """Compute per-atom RGBA colors and marker sizes.
        Real atoms: CPK colors from ELEM_COLORS, size from vdW radius.
        Epairs (type=1): cyan (0,0.8,0.8), small size — represents lone pair.
        Sigma holes (type=2): magenta (0.8,0,0.8), same small size — represents positive pole.
        Cyan/magenta are complementary colors, visually distinguishing epairs (negative,
        attract H) from sigma holes (positive, attract O/N)."""
        n = len(enames)
        colors = np.zeros((n, 4), dtype=np.float32)
        sizes = np.zeros(n, dtype=np.float32)
        for i, (e, t) in enumerate(zip(enames, types)):
            if t == 1:
                colors[i] = ELEM_COLORS['E']
                sizes[i] = ELEM_RADII['E'] * 20 + 5
            elif t == 2:
                colors[i] = ELEM_COLORS['Sh']
                sizes[i] = ELEM_RADII['Sh'] * 20 + 5
            else:
                colors[i] = ELEM_COLORS.get(e, (0.5, 0.5, 0.5, 1.0))
                sizes[i] = ELEM_RADII.get(e, 0.3) * 20 + 5
        return colors, sizes

    # --- Build visuals ---

    def _compute_bonds(self, apos, types, max_dist=1.8):
        """Compute bonds between real atoms (type=0) by distance cutoff.
        Returns array of (2,3) line segments for vispy Line."""
        real_mask = types == 0
        real_idx = np.where(real_mask)[0]
        if len(real_idx) < 2:
            return np.zeros((0, 3), dtype=np.float32)
        pos = apos[real_idx]
        bonds = []
        for i in range(len(real_idx)):
            for j in range(i + 1, len(real_idx)):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < max_dist:
                    bonds.append([pos[i], pos[j]])
        if not bonds:
            return np.zeros((0, 3), dtype=np.float32)
        return np.array(bonds, dtype=np.float32).reshape(-1, 3)

    def _compute_dummy_bonds(self, apos, types, enames):
        """Compute lines from epairs (type=1) and sigma holes (type=2) to nearest real atom.
        Returns (segments, colors) where colors are per-segment RGBA."""
        dummy_idx = np.where(types != 0)[0]
        real_idx = np.where(types == 0)[0]
        if len(dummy_idx) == 0 or len(real_idx) == 0:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 4), dtype=np.float32)
        segments = []
        colors = []
        for i in dummy_idx:
            # Find nearest real atom
            best_j = None
            best_d = 1e9
            for j in real_idx:
                d = np.linalg.norm(apos[i] - apos[j])
                if d < best_d:
                    best_d = d
                    best_j = j
            if best_j is not None:
                segments.append([apos[i], apos[best_j]])
                if types[i] == 1:
                    colors.append((0.0, 0.8, 0.8, 0.3))  # cyan, faint
                else:
                    colors.append((0.8, 0.0, 0.8, 0.3))  # magenta, faint
        if not segments:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 4), dtype=np.float32)
        seg_arr = np.array(segments, dtype=np.float32).reshape(-1, 3)
        col_arr = np.repeat(np.array(colors, dtype=np.float32), 2, axis=0)  # per-vertex
        return seg_arr, col_arr

    def _build_static(self):
        rbd = self.rbd
        n = int(getattr(rbd, 'static_n', 0) or 0)
        if n == 0:
            empty = np.zeros((0, 3), dtype=np.float32)
            self.static_markers.set_data(pos=empty)
            self.static_bonds.set_data(empty)
            self.static_dummy_bonds.set_data(empty)
            return
        apos = rbd.static_apos_host[:, :3]
        types = rbd.static_type_host
        enames = rbd.static_enames if hasattr(rbd, 'static_enames') else ['?'] * n
        colors, sizes = self._get_colors_sizes(enames, types)
        self.static_markers.set_data(pos=apos.astype(np.float32), face_color=colors,
                                      size=sizes, edge_width=0.5, edge_color='black', symbol='disc')
        self.static_colors = colors
        self.static_sizes = sizes
        # Bonds
        bond_segs = self._compute_bonds(apos, types)
        self.static_bonds.set_data(bond_segs)
        # Dummy bonds (epair/sigma to host)
        dummy_segs, dummy_colors = self._compute_dummy_bonds(apos, types, enames)
        self.static_dummy_bonds.set_data(dummy_segs, color=dummy_colors)

    def _build_dynamic(self):
        pos = self._get_dyn_pos()
        types = self.rbd.dyn_type_host
        enames = self.rbd.enames
        colors, sizes = self._get_colors_sizes(enames, types)
        self.dyn_markers.set_data(pos=pos, face_color=colors,
                                   size=sizes, edge_width=0.5, edge_color='black', symbol='disc')
        self.dyn_colors = colors
        self.dyn_sizes = sizes
        # Bonds (in body frame, will be updated each frame)
        bond_segs = self._compute_bonds(pos, types)
        self.dyn_bonds.set_data(bond_segs)
        # Dummy bonds (epair/sigma to host)
        dummy_segs, dummy_colors = self._compute_dummy_bonds(pos, types, enames)
        self.dyn_dummy_bonds.set_data(dummy_segs, color=dummy_colors)

    def _update_dynamic(self):
        pos = self._get_dyn_pos()
        self.dyn_markers.set_data(pos=pos, face_color=self.dyn_colors,
                                   size=self.dyn_sizes, edge_width=0.5, edge_color='black', symbol='disc')
        # Update dynamic bonds
        bond_segs = self._compute_bonds(pos, self.rbd.dyn_type_host)
        self.dyn_bonds.set_data(bond_segs)
        # Update dummy bonds
        dummy_segs, dummy_colors = self._compute_dummy_bonds(pos, self.rbd.dyn_type_host, self.rbd.enames)
        self.dyn_dummy_bonds.set_data(dummy_segs, color=dummy_colors)
        out = self.rbd.download_outputs()
        a = int(getattr(self.rbd, 'active_body', 0))
        E = float(out['atom_positions'][0, :, 3].sum())
        F = out['body_force'][a, :3]
        Fmag = float(np.linalg.norm(F))
        self.panel.lbl_E.setText(f"E={E:.4f}  |F|={Fmag:.4f}")

    # --- Potential map ---

    def _map_z_height(self):
        """Map plane z: active CoM if available, else rbd.map_z / 0."""
        rbd = self.rbd
        try:
            out = rbd.download_selected(('pos',))
            a = int(getattr(rbd, 'active_body', 0))
            return float(out['pos'][a, 2])
        except Exception:
            return float(getattr(rbd, 'map_z', 0.0) or 0.0)

    def _recompute_map(self):
        if not self.panel.show_map():
            self.map_image.visible = False
            return
        probe_R0, probe_E0, probe_q = self.panel.get_probe_params()
        He = self.panel.spins['He'].value()
        Hs = self.panel.spins['Hs'].value()
        rc = self.panel.spins['rc'].value()
        w = self.panel.spins['w'].value()
        mode = self.panel.get_pairff_mode()
        z_height = self._map_z_height()
        rbd = self.rbd
        n_static = int(getattr(rbd, 'static_n', 0) or 0)
        if n_static > 0:
            if mode == 'unified':
                beta = (rbd.pairff_params_host or {}).get('beta', 1.7)
                Emap, xs, ys, extent = compute_potential_map_unified(
                    rbd.static_apos_host, rbd.static_REQ_host,
                    rbd.static_enames, rbd.static_type_host,
                    probe_R0, probe_E0, probe_q, z_height=z_height, margin=4.0, step=0.1,
                    He=He, Hs=Hs, w=w, beta=beta)
            else:
                Emap, xs, ys, extent = compute_potential_map(
                    rbd.static_apos_host, rbd.static_REQ_host,
                    rbd.static_enames, rbd.static_type_host,
                    probe_R0, probe_E0, probe_q, z_height=z_height, margin=4.0, step=0.1,
                    He=He, Hs=Hs, rc=rc, w=w)
        else:
            # No env molecules: empty PairFF layer; extent from active sites or FAF default
            out = rbd.download_outputs()
            apos = out['atom_positions'][0, :, :3]
            margin, step = 4.0, 0.1
            xmin, ymin = float(apos[:, 0].min() - margin), float(apos[:, 1].min() - margin)
            xmax, ymax = float(apos[:, 0].max() + margin), float(apos[:, 1].max() + margin)
            xs = np.arange(xmin, xmax + step, step)
            ys = np.arange(ymin, ymax + step, step)
            Emap = np.zeros((len(ys), len(xs)), dtype=np.float64)
            extent = [xmin, xmax, ymin, ymax]

        # Compose FAF substrate at the same z (diagnostic: E_PairFF + E_FAF)
        fit = getattr(rbd, 'faf_fit', None)
        if fit is not None and getattr(rbd, 'faf_mode', False):
            from spammm.surfaces.FoldedRigid import eval_folded_potential_grid, faf_type_idx_for_probe
            ityp = faf_type_idx_for_probe(fit, probe_R0, probe_E0, probe_q)
            Efaf = eval_folded_potential_grid(fit, ityp, xs, ys, z_height)
            Emap = Emap + Efaf

        rgba = potential_to_rgba(Emap)
        self.map_image.set_data(rgba)
        # Set transform to map image pixels to world coordinates
        # vispy Image renders row 0 at bottom by default, matching our data (ys[0]=ymin)
        x0, x1, y0, y1 = extent
        dx = (x1 - x0) / max(len(xs) - 1, 1)
        dy = (y1 - y0) / max(len(ys) - 1, 1)
        from vispy.visuals.transforms import STTransform
        self.map_image.transform = STTransform(
            translate=(x0, y0, -0.1), scale=(dx, dy, 1))
        self.map_image.visible = True
        self._map_data = (Emap, extent)
        self.canvas.update()

    # --- Camera control (from AtomScene/VispyUtils.py) ---

    def _cam_zoom(self, delta):
        cam = self.view.camera
        if cam is None: return
        z0 = float(getattr(cam, 'scale_factor', 1.0))
        s = float(np.exp(-float(delta) * float(self._cam_zoom_speed)))
        z1 = z0 * s
        z1 = max(self._cam_zoom_min, min(self._cam_zoom_max, z1))
        cam.scale_factor = z1
        self.canvas.update()

    def _cam_pan(self, dx, dy):
        cam = self.view.camera
        if cam is None: return
        center = np.array(cam.center, dtype=np.float64)
        center[0] += float(dx) * float(self._cam_pan_speed)
        center[1] += float(dy) * float(self._cam_pan_speed)
        cam.center = tuple(center)
        self.canvas.update()

    # --- Ray-plane picking (from AtomScene/VispyUtils.py) ---

    def _ray_from_mouse(self, mouse_pos, z0=0.0, z1=1.0):
        view_pos = np.array(mouse_pos) - self.view.pos[:2]
        tr = self.view.scene.transform
        p0 = np.array(tr.imap((view_pos[0], view_pos[1], float(z0)))[:3], dtype=np.float32)
        p1 = np.array(tr.imap((view_pos[0], view_pos[1], float(z1)))[:3], dtype=np.float32)
        d = p1 - p0
        dn = float(np.linalg.norm(d))
        if dn <= 1e-20:
            d = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        else:
            d /= dn
        return p0, d

    def _intersect_ray_plane(self, r0, rd, p0, n):
        denom = float(np.dot(rd, n))
        if abs(denom) < 1e-12: return None
        t = float(np.dot((p0 - r0), n) / denom)
        return r0 + rd * t

    def _nearest_point_on_ray(self, mouse_pos, atom_pos):
        """Compute nearest point on mouse ray to atom_pos.

        Key design decision: instead of intersecting the mouse ray with the
        z=0 plane (which would give wrong results for atoms above/below z=0),
        we project the atom position onto the mouse ray. This gives the point
        on the ray closest to the atom, ensuring:
          1. The spring force (atom→anchor) is perpendicular to the view ray
          2. Dragging works regardless of the atom's z-coordinate
          3. In orthographic top-down view, the anchor tracks the mouse in XY

        Math: t = dot(atom_pos - r0, rd) / dot(rd, rd). Since rd is unit,
        this simplifies to t = dot(atom_pos - r0, rd). Anchor = r0 + t*rd.
        """
        r0, rd = self._ray_from_mouse(mouse_pos)
        # t = dot(atom_pos - r0, rd) / dot(rd, rd)  (rd is unit, so dot(rd,rd)=1)
        t = float(np.dot(atom_pos - r0, rd))
        return r0 + rd * t

    def _mouse_to_world_xy(self, mouse_pos):
        """Intersect mouse ray with z=0 plane — used for picking only."""
        r0, rd = self._ray_from_mouse(mouse_pos)
        pt = self._intersect_ray_plane(r0, rd, np.zeros(3), np.array([0, 0, 1]))
        return pt

    def _pick_atom(self, mouse_pos):
        """Pick nearest dynamic atom (classic 1+1 mode). Returns atom index or -1."""
        pt = self._mouse_to_world_xy(mouse_pos)
        if pt is None: return -1
        xy = pt[:2]
        pos = self._get_dyn_pos()
        d2 = np.sum((pos[:, :2] - xy[None, :]) ** 2, axis=1)
        idx = int(np.argmin(d2))
        if d2[idx] < 1.0: return idx
        return -1

    def _is_multibody(self):
        rbd = self.rbd
        return bool(getattr(rbd, '_mb_packs', None) is not None and (
            getattr(rbd, 'allmol_mode', False) or getattr(rbd, 'env_mode', False)))

    def _update_active_label(self):
        rbd = self.rbd
        if self._is_multibody():
            self.panel.set_active_label(rbd.active_body, len(rbd._mb_packs))
        else:
            self.panel.set_active_label(0)

    def _world_sites_all_bodies(self):
        """World-frame sites for every multi-body molecule. Syncs active pose from GPU."""
        rbd = self.rbd
        rbd.sync_active_pose_from_gpu()
        sites = []
        for j, pack in enumerate(rbd._mb_packs):
            world = _body_sites_world(pack['rel'], rbd._mb_pos[j], rbd._mb_quat[j])
            sites.append(world)
        return sites

    def _pick_body_atom(self, mouse_pos, thresh2=1.0):
        """Pick nearest site across all multi-body molecules. Returns (body_id, atom_idx) or (-1, -1)."""
        pt = self._mouse_to_world_xy(mouse_pos)
        if pt is None:
            return -1, -1
        xy = pt[:2]
        best_d2 = thresh2
        best_body, best_atom = -1, -1
        for j, world in enumerate(self._world_sites_all_bodies()):
            d2 = np.sum((world[:, :2] - xy[None, :]) ** 2, axis=1)
            ia = int(np.argmin(d2))
            if d2[ia] < best_d2:
                best_d2 = float(d2[ia])
                best_body, best_atom = j, ia
        return best_body, best_atom

    def _select_active_body(self, body_id):
        """Make body_id mobile; freeze others; refresh markers + potential map."""
        rbd = self.rbd
        if not self._is_multibody():
            return
        if int(body_id) == int(rbd.active_body):
            return
        # Clear any drag anchors before active switch
        if self.dragging:
            self._clear_anchor(self.drag_atom)
            self.dragging = False
            self.drag_atom = -1
            self.anchor_line.set_data(np.zeros((0, 3), dtype=np.float32))
            self.anchor_marker.set_data(np.zeros((0, 3), dtype=np.float32))
        rbd.set_active_body(int(body_id))
        self.panel.rbd = rbd
        self._build_static()
        self._build_dynamic()
        self._update_active_label()
        self._recompute_map()
        self.canvas.update()

    # --- Mouse events ---

    def on_mouse_press(self, event):
        if event.button != 1: return
        if self._is_multibody():
            body_id, atom_idx = self._pick_body_atom(event.pos)
            if body_id < 0:
                return
            if body_id != self.rbd.active_body:
                self._select_active_body(body_id)
                # After switch, atom_idx is still valid in the new active body frame
            # Start drag on the (now) active molecule
            self.dragging = True
            self.drag_atom = atom_idx
            atom_pos = self._get_dyn_pos()[atom_idx]
            world = self._nearest_point_on_ray(event.pos, atom_pos)
            self._set_anchor(atom_idx, world)
            return
        idx = self._pick_atom(event.pos)
        if idx >= 0:
            self.dragging = True
            self.drag_atom = idx
            atom_pos = self._get_dyn_pos()[idx]
            world = self._nearest_point_on_ray(event.pos, atom_pos)
            self._set_anchor(idx, world)

    def on_mouse_release(self, event):
        if self.dragging:
            self._clear_anchor(self.drag_atom)
            self.dragging = False
            self.drag_atom = -1
            self.anchor_line.set_data(np.zeros((0, 3), dtype=np.float32))
            self.anchor_marker.set_data(np.zeros((0, 3), dtype=np.float32))

    def on_mouse_move(self, event):
        if not self.dragging or self.drag_atom < 0: return
        atom_pos = self._get_dyn_pos()[self.drag_atom]
        world = self._nearest_point_on_ray(event.pos, atom_pos)
        self._set_anchor(self.drag_atom, world)

    def on_mouse_wheel(self, event):
        delta = None
        if hasattr(event, 'delta') and event.delta is not None:
            try: delta = float(event.delta[1])
            except: delta = float(event.delta)
        if delta is None: return
        if abs(delta) > 50: delta /= 120.0
        self._cam_zoom(delta)

    def _set_anchor(self, atom_idx, world_pos):
        """Upload anchor spring for one atom. anchors[i].w > 0 = active spring.

        atom_idx is local to the active molecule. In allmol_mode the GPU buffer
        is flat over all molecules, so we write at mol_offsets[active]+atom_idx.
        """
        rbd = self.rbd
        k = self.panel.get_anchor_k()
        anchors = np.zeros((rbd.total_atoms, 4), dtype=np.float32)
        anchors[:, 3] = -1.0
        i0 = 0
        if getattr(rbd, 'allmol_mode', False) and getattr(rbd, 'mol_offsets', None) is not None:
            i0 = int(rbd.mol_offsets[int(rbd.active_body)])
        gi = i0 + int(atom_idx)
        anchors[gi, :3] = world_pos
        anchors[gi, 3] = k
        rbd.anchors = anchors
        rbd.upload_anchors()
        atom_pos = self._get_dyn_pos()[atom_idx]
        line_pos = np.array([atom_pos, world_pos[:3]], dtype=np.float32)
        self.anchor_line.set_data(line_pos)
        self.anchor_marker.set_data(pos=world_pos[:3].reshape(1, 3).astype(np.float32),
                                     face_color=(1, 0, 0, 1), size=10, edge_width=0,
                                     symbol='cross')

    def _clear_anchor(self, atom_idx):
        rbd = self.rbd
        anchors = np.zeros((rbd.total_atoms, 4), dtype=np.float32)
        anchors[:, 3] = -1.0
        rbd.anchors = anchors
        rbd.upload_anchors()

    # --- Key events ---

    def on_key_press(self, event):
        key = event.key
        if key == 'Escape':
            self.main_win.close()
        elif key == ' ':
            self.running = not self.running
            self.panel.btn_run.setChecked(self.running)
            self.panel.lbl_status.setText("RUNNING" if self.running else "PAUSED")
            self.panel.lbl_status.setStyleSheet(
                f"font-weight: bold; color: {'green' if self.running else 'red'};")
        elif key in ('r', 'R'):
            self.panel.on_reset()
        elif key in ('f', 'F'):
            self.fire = not self.fire
            self.panel.btn_fire.setChecked(self.fire)
            self.panel.on_fire_toggle()
        elif key in ('Up', 'ArrowUp'):
            self._cam_pan(0, 1)
        elif key in ('Down', 'ArrowDown'):
            self._cam_pan(0, -1)
        elif key in ('Left', 'ArrowLeft'):
            self._cam_pan(-1, 0)
        elif key in ('Right', 'ArrowRight'):
            self._cam_pan(1, 0)

    def on_run_toggle(self):
        self.running = self.panel.is_running()
        self.panel.lbl_status.setText("RUNNING" if self.running else "PAUSED")
        self.panel.lbl_status.setStyleSheet(
            f"font-weight: bold; color: {'green' if self.running else 'red'};")

    # --- Timer ---

    def on_timer(self, event):
        self.fire = self.panel.is_fire()
        self.dt = self.panel.get_dt()
        self.steps_per_frame = self.panel.get_steps_per_frame()
        if self.running:
            self.rbd.run_pairff(self.steps_per_frame, self.dt, fire=self.fire)
        self._update_dynamic()
        self.canvas.update()
        self.frame_count += 1

    def run(self):
        self.main_win.show()
        self.qt_app.exec_()
