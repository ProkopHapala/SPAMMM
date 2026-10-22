"""pme_lattice.py — Periodic rectangular-lattice PME8 (Temirov charging-ring model).

Compute backend for the ChargeRings "Lattice PME8 (periodic)" GUI mode. Port of
the export-repo workhorse ``scripts/temirov_pme.py`` (TemirovPME) generalized to
a preset-driven lattice with full tip multipole coefficients. No Qt here.

Conventions (identical to the reference implementation):
- geometry.json supplies ``cell_A`` (rows a,b), ``basis_A``, ``origin_pixel``,
  ``pixels_per_A`` and ``screen_transform_xy_to_image``. Map row-vector physical
  xy -> pixel via ``xy @ T * pixels_per_A + origin_pixel``; inverse for scans.
  Image y points down.
- Supercell = [n1*a, n2*b]; sites = basis + i*a + j*b, ordering
  ``[i,j] for j in range(n2) for i in range(n1)`` per basis site (basis outer).
  Sites sit at z=0, rots identity. Total sites <= 8 (PME8 kernel limit).
- Nearest-replica PBC: the kernel picks ONE translated image per physical site
  (tip_cell / tip_periodic_sites) — replicas are never summed. The supercell is
  an occupation proxy: opposite supercell images share one site occupation.
- Wij policy: minimum-image pair distances INSIDE the supercell
  (delta -= rint(delta @ inv(supercell)) @ supercell); one contact per pair,
  selected by distance shell |a| (long), |b| (short), |a+b| = |a-b| (diagonal).
  W_a/W_b/W_ab are given in meV in the preset and converted to eV here.
- Site energy (occupied-environment convention): E_i = E_ref - W.sum(axis=1).
  Set ``occupied_env_energy=False`` to use E_i = E_ref directly.
- Voltage = tip chemical potential mu_tip = gate (Vtips_gate = Vtips = bias).
- Tip multipole: 10 kernel coefficients cs (order=2), see make_multipole_cs.
  Dipole/quadrupole entries are VOLTAGE-COUPLED kernel coefficients in length
  units relative to Q0 — not permanent moments in Debye. The mirror term uses
  the SAME coefficients (eval(d_mirror) is subtracted, kernel convention).
- Current solver: PauliSolverCL8, NVIDIA required, GMRES/double stationary
  solve, apply_validity_cut=False. dI/dV = (I(V+dV)-I(V-dV))/(2 dV).
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.interpolate import RegularGridInterpolator

from spammm.quantum import pauli_scan as ps


def image_ridge_features(image, pixel_step, widths_px=(2.,4.,6.), background_px=12., border_px=12., polarity='signed'):
    """Normalized signed band-pass images; all lengths are original image pixels.

    Same operation on experiment and PME, without translation or circle fitting.
    Optional magnitude features ignore polarity only in the score, never in
    the saved physical signal. Smoothing supplies a positional
    tolerance, not a free registration. Exclude the outer border from the score.
    (Ported verbatim from export scripts/temirov_pme.py.)
    """
    image=np.asarray(image,float)
    background=gaussian_filter(image,background_px/pixel_step)
    border=int(np.ceil(border_px/pixel_step));features=[]
    for width in widths_px:
        a=(gaussian_filter(image,width/pixel_step)-background)[border:-border,border:-border].copy()
        a-=a.mean()
        if polarity=='magnitude':a=np.abs(a);a-=a.mean()
        elif polarity!='signed':raise ValueError('Unknown image feature polarity')
        norm=np.linalg.norm(a)
        if not np.isfinite(norm) or norm<1e-30:raise FloatingPointError('No finite image contrast for matching')
        features.append(a.ravel()/norm)
    return np.array(features)


def bandpass_image(image, pixel_step, width_px=4., background_px=12.):
    """Full-size signed band-pass display image (same filters as ridge features)."""
    image = np.asarray(image, float)
    a = gaussian_filter(image, width_px/pixel_step) - gaussian_filter(image, background_px/pixel_step)
    a -= a.mean()
    return a / max(np.linalg.norm(a), 1e-30)


def _norm2d(a):
    a = a - a.mean(); return a / max(np.linalg.norm(a), 1e-30)


def shift_tolerant_corr(a, b, max_shift_px=8):
    """Max Pearson correlation of 2D feature maps over integer shifts, overlap region only.
    Same normalized band-pass correlation as the MC objective but position-tolerant.
    Returns (best_corr, (dy, dx)) with shift applied to a."""
    best, bshift = -2., (0, 0); H, W = a.shape
    for dy in range(-max_shift_px, max_shift_px + 1):
        for dx in range(-max_shift_px, max_shift_px + 1):
            ya, xa = slice(max(0, dy), H + min(0, dy)), slice(max(0, dx), W + min(0, dx))
            yb, xb = slice(max(0, -dy), H + min(0, -dy)), slice(max(0, -dx), W + min(0, -dx))
            c = float((_norm2d(a[ya, xa]) * _norm2d(b[yb, xb])).sum())
            if c > best: best, bshift = c, (dy, dx)
    return best, bshift


def ridge_chamfer_score(a, b, thresh_frac=0.35, sigma_px=3.):
    """Symmetric signed ridge-proximity score in (0,1]; 1 = every ridge of each map lies on a same-sign ridge of the other.

    a, b: signed band-pass feature maps (any consistent normalization). Ridges = pixels above
    thresh_frac * max(|map|), matched separately for positive and negative polarity, so bright
    features must align with bright and NDR with NDR. A polarity absent from both maps is skipped;
    present in only one it contributes 0 (wrong-sign/missing features penalized). exp(-d/sigma_px)
    gives a few-px positional tolerance; rewards feature geometry (shape + sign + coarse position),
    not pixel-exact amplitude overlap."""
    from scipy.ndimage import distance_transform_edt
    parts = []
    for sgn in (1., -1.):
        ma, mb = (a * sgn > thresh_frac * np.abs(a).max()), (b * sgn > thresh_frac * np.abs(b).max())
        if not ma.any() and not mb.any(): continue                  # polarity absent from both maps: nothing to match
        if not ma.any() or not mb.any(): parts += [0., 0.]; continue  # polarity on one side only: penalize both directions
        da, db = distance_transform_edt(~ma), distance_transform_edt(~mb)
        parts += [float(np.exp(-db[ma] / sigma_px).mean()), float(np.exp(-da[mb] / sigma_px).mean())]
    return float(np.mean(parts)) if parts else 0.


def power_spectrum_corr(a, b):
    """Pearson correlation of log 2D Fourier power maps (periodicity + anisotropy, position-free)."""
    pa = _norm2d(np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(a)))))
    pb = _norm2d(np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(b)))))
    return float((pa * pb).sum())


def branch_signal(a):
    """Signed spatial ridge contrast; same operation for experiment and PME.
    (Ported verbatim from export scripts/fit_temirov_interactions.py.)"""
    smoothed=gaussian_filter(a,(.7,.7))
    signal=smoothed-gaussian_filter(smoothed,(0,5.))
    # Equalize voltage-column weighting without amplifying weak/noisy columns.
    norm=np.linalg.norm(signal,axis=1,keepdims=True)
    floor=.2*np.percentile(norm,75)
    signal=signal/np.maximum(norm,floor)
    return signal/(np.linalg.norm(signal)+1e-30)


def make_multipole_cs(Q0, p=(0., 0., 0.), Qxx=0., Qyy=0., Qzz=0., Qxy=0., Qxz=0., Qyz=0.):
    """PME8 multipole coefficient array (10 floats, order=2).

    Kernel slots (kernels/PME8.cl eval_multipole), d = tip - site:
      E = (1/r) * ( cs0
        + (cs1 dx + cs2 dy + cs3 dz) / r^2
        + (cs4 dx^2 + cs5 dy^2 + cs6 dz^2 + cs9 dx dy + cs7 dy dz + cs8 dz dx) / r^4 )
    Off-diagonal quadrupoles (cs7,cs8,cs9) are SINGLE coefficients — the kernel
    pairs them as (cs9 dx dy + cs7 dy dz + cs8 dz dx), i.e. they are not doubled.
    Temirov's axial quadrupole lives in cs6 = Qzz (dz^2), NOT cs5 — do not use
    pauli_scan.make_quadrupole_Coeffs for this mode (it puts Qzz in cs5=dy^2).
    """
    return np.array([Q0, p[0], p[1], p[2], Qxx, Qyy, Qzz, Qyz, Qxz, Qxy], dtype=np.float32), 2


def load_lattice_params(path):
    """Load a lattice preset JSON; resolve geometry and verify its sha256."""
    p = json.loads(Path(path).read_text())
    p['_preset_path'] = str(path)
    if p.get('geometry_file'):
        gpath = Path(p['geometry_file'])
        if not gpath.is_absolute():
            gpath = Path(path).parent / gpath
        p['geometry_file'] = str(gpath)
        if p.get('geometry_sha256'):
            sha = hashlib.sha256(gpath.read_bytes()).hexdigest()
            if sha != p['geometry_sha256']:
                raise ValueError(f'geometry sha256 mismatch: {gpath} ({sha} != {p["geometry_sha256"]})')
    return p


def available_mV(exp_dir, pattern='{mV}mv.txt'):
    """Sorted integer mV labels with an existing file in exp_dir."""
    out = []
    if not os.path.isdir(exp_dir):
        return np.array([], int)
    prefix, _, suffix = pattern.partition('{mV}')
    for f in os.listdir(exp_dir):
        if f.startswith(prefix) and f.endswith(suffix):
            try:
                out.append(int(f[len(prefix):len(f) - len(suffix)]))
            except ValueError:
                pass
    return np.array(sorted(out))


def load_experiment_image(params, bias):
    """(raw image, mV_used) for the nearest available 5 mV file; raises if missing."""
    exp_dir = params.get('exp_dir')
    if not exp_dir or not os.path.isdir(exp_dir):
        raise FileNotFoundError(f'experiment dir missing: {exp_dir}')
    pattern = params.get('exp_pattern', '{mV}mv.txt')
    mV = available_mV(exp_dir, pattern)
    if len(mV) == 0:
        raise FileNotFoundError(f'no {pattern} files in {exp_dir}')
    use = int(mV[np.argmin(np.abs(mV - round(bias * 1000)))])
    return np.loadtxt(os.path.join(exp_dir, pattern.format(mV=use))), use


class LatticePME:
    """Periodic rectangular-lattice PME8 model (Temirov). See module docstring."""

    def __init__(self, params, solver=None, solvers=None, max_iter=20000):
        self.params = dict(params)
        self.max_iter = max_iter
        # Solver cache keyed (kind, method, precision, tol); 'solvers' lets a
        # caller share solvers across model rebuilds (GUI supercell changes).
        self._solvers = solvers if solvers is not None else {}
        self._padded = {}
        if solver is not None:
            self._solvers[('pme8', 'gmres', 'double', 1e-12, 1 << 30)] = solver
        g = json.loads(Path(params['geometry_file']).read_text())
        self.geometry = g
        self.cell = np.array(g['cell_A'], float)
        self.basis = np.array(g['basis_A'], float)
        self.T = np.array(g['screen_transform_xy_to_image'], float)
        self.scale = float(g['pixels_per_A'])
        self.origin = np.array(g['origin_pixel'], float)
        n1, n2 = (int(x) for x in params.get('supercell', [2, 4]))
        self.ns = (n1, n2)
        self.supercell = np.array([n1 * self.cell[0], n2 * self.cell[1]])
        idx = [(k, i, j) for k in range(len(self.basis)) for j in range(n2) for i in range(n1)]
        sites = np.zeros((len(idx), 4))
        sites[:, :2] = np.array([self.basis[k] + i * self.cell[0] + j * self.cell[1] for k, i, j in idx])
        if len(sites) > 8:
            raise ValueError(f'lattice mode supports at most 8 sites (got {len(sites)})')
        self.sites = sites
        self.rots = np.tile(np.eye(3), (len(sites), 1, 1))
        # Minimum-image pair distances inside the supercell; distance shells
        # |a|, |b|, |a+b| pick long/short/diagonal contacts (one per pair).
        delta = sites[:, None, :2] - sites[None, :, :2]
        delta -= np.rint(delta @ np.linalg.inv(self.supercell)) @ self.supercell
        distance = np.linalg.norm(delta, axis=2)
        self.distance = distance
        a, b = self.cell
        # Contact vectors for the W_a/W_b/W_ab shells: default |a|,|b|,|a+b|; geometry may override
        # (e.g. a centered basis makes |(a+b)/2| the nearest neighbour -> list it first).
        vecs = np.array(g.get('contact_vectors_A', [a, b, a + b]), float)
        self.contact_vectors = vecs
        self.contact_masks = np.array([np.isclose(distance, np.linalg.norm(v), atol=1e-6) for v in vecs], float)
        self.W = np.zeros((len(sites), len(sites)))
        self.configure(params)

    # --- solver selection ---------------------------------------------------
    def _solver_kind(self):
        kind = self.params.get('solver', 'auto')
        if kind == 'auto':
            kind = 'pme4' if len(self.sites) <= 4 else 'pme8'
        if kind not in ('pme4', 'pme8'):
            raise ValueError(f'unknown lattice solver kind {kind!r}')
        if kind == 'pme4' and len(self.sites) > 4:
            raise ValueError(f'pme4 supports at most 4 lattice sites (got {len(self.sites)}; reduce supercell or use pme8)')
        return kind

    def _solver_key(self):
        p = self.params
        kind = self._solver_kind()
        if kind == 'pme4':
            return (kind, 'direct', 'single', 0.)
        prec = p.get('stationary_precision', 'double')
        tol = float(p.get('tol', 1e-12 if prec == 'double' else 1e-6))
        return (kind, p.get('stationary_method', 'gmres'), prec, tol, int(p.get('gmres_workspace_bytes', 1 << 30)))

    def get_solver(self):
        """Lazy solver cache; pme8 honours stationary_method/precision/tol."""
        key = self._solver_key()
        sol = self._solvers.get(key)
        if sol is None:
            if key[0] == 'pme4':
                from spammm.quantum.PauliSolverCL import PauliSolverCL
                sol = PauliSolverCL(nSingle=4, preferred_vendor='nvidia', bPrint=False)
            else:
                from spammm.quantum.PauliSolverCL8 import PauliSolverCL8
                sol = PauliSolverCL8(preferred_vendor='nvidia', bPrint=False, max_iter=self.max_iter, tol=key[3],
                                     stationary_method=key[1], stationary_precision=key[2],
                                     gmres_workspace_bytes=key[4])
            if 'nvidia' not in sol.queue.device.vendor.lower():
                raise RuntimeError('NVIDIA GPU required for lattice PME')
            self._solvers[key] = sol
        return sol

    def _pad(self, kind):
        """Sites/rots/Wij padded to the solver's nSingle with far +E spectators."""
        key = self._solver_kind() if kind is None else kind
        n_single = 4 if key == 'pme4' else 8
        cached = self._padded.get(key)
        if cached is not None:
            return cached
        n = len(self.sites)
        sites = np.full((n_single, 4), 1e3)
        sites[:n] = self.sites
        for i in range(n, n_single):
            sites[i] = [1e3 * (i + 1), 1e3 * (i + 1), 0.0, 10.0]  # far away, +10 eV => always empty
        rots = np.tile(np.eye(3), (n_single, 1, 1))
        mask = np.zeros(n_single, bool)
        mask[:n] = True
        W = np.zeros((n_single, n_single))
        W[:n, :n] = self.W
        cached = dict(n_single=n_single, sites=sites, rots=rots, mask=mask, W=W)
        self._padded[key] = cached
        return cached

    def configure(self, params):
        """Update everything that depends on params (no solver rebuild)."""
        p = self.params = dict(params)
        self.order_cs = make_multipole_cs(float(p.get('Q0', 1.)), p=(float(p.get('px', 0.)), float(p.get('py', 0.)), float(p.get('pz', 0.))),
                                        Qxx=float(p.get('Qxx', 0.)), Qyy=float(p.get('Qyy', 0.)), Qzz=float(p.get('Qzz', 0.)),
                                        Qxy=float(p.get('Qxy', 0.)), Qxz=float(p.get('Qxz', 0.)), Qyz=float(p.get('Qyz', 0.)))
        self.cs, self.order = self.order_cs
        base = ps.ruslan_default_params(nsite=len(self.sites), Temp=float(p.get('Temp', 1.)), Rtip=float(p.get('Rtip', 3.)), zV0=float(p.get('zV0', -.9)),
                                      decay=float(p.get('decay', .3)), GammaS=float(p.get('GammaT', .01)), GammaT=float(p.get('GammaT', .01)),
                                      Esite=0, W=0, bMirror=bool(p.get('bMirror', True)), bRamp=bool(p.get('bRamp', True)))
        w = np.array([float(p.get('W_a_meV', 0.)), float(p.get('W_b_meV', 0.)), float(p.get('W_ab_meV', 0.))]) / 1000.
        self.W = np.tensordot(w, self.contact_masks, axes=1)
        self.sites[:, 3] = float(p.get('E_ref', 0.)) - (self.W.sum(axis=1) if bool(p.get('occupied_env_energy', True)) else 0.)
        for pad in self._padded.values():
            pad['sites'][:len(self.sites)] = self.sites
            pad['W'][:len(self.sites), :len(self.sites)] = self.W
        self.solver = self.get_solver()
        self.device = self.solver.queue.device.name
        ps.configure_leads(self.solver, base)
        self.cpp = ps.make_cpp_params(dict(base, zVd=float(p['zVd'])))
        self.cpp[6] = 0

    # --- coordinate maps ----------------------------------------------------
    def image_to_physical(self, pixel_xy):
        return (np.asarray(pixel_xy, float) - self.origin) @ np.linalg.inv(self.T) / self.scale

    def physical_to_image(self, xy):
        return np.asarray(xy, float) @ self.T * self.scale + self.origin

    # --- core compute -------------------------------------------------------
    def current(self, bias, xy):
        xy = np.asarray(xy, float).reshape(-1, 2)
        bias = np.broadcast_to(np.asarray(bias, float), len(xy))
        tips = np.column_stack([xy, np.full(len(xy), float(self.params['z_charge']))])
        pad = self._pad(self._solver_kind())
        result = self.solver.scan_current_tip(tips, bias, pad['sites'], self.cpp, self.order, self.cs, rots=pad['rots'], Wij=pad['W'],
                                              Vtips_gate=bias, tip_cell=self.supercell, tip_periodic_sites=pad['mask'],
                                              return_probs=False, return_state_energies=False, apply_validity_cut=False)
        if not np.isfinite(result[0]).all():
            raise FloatingPointError('Nonfinite PME lattice current')
        return result

    def derivative(self, bias, xy, dV=None):
        dV = float(dV if dV is not None else self.params.get('dV', .001))
        bias = np.asarray(bias, float)
        return (self.current(bias + dV, xy)[0] - self.current(bias - dV, xy)[0]) / (2 * dV)

    def _grid_xy(self, npix, width_px, height_px):
        px = np.linspace(0, width_px - 1, int(npix))
        ny = int(npix) if height_px == width_px else max(3, int(round(npix * (height_px - 1) / (width_px - 1))))
        py = np.linspace(0, height_px - 1, ny)
        xx, yy = np.meshgrid(px, py)
        pixelxy = np.column_stack([xx.ravel(), yy.ravel()])
        return px, py, pixelxy, self.image_to_physical(pixelxy)

    def scan_map(self, bias, npix, width_px, height_px):
        """One solver call each at V, V±dV over the whole map. -> dict(I,dIdV,pixels,xy)."""
        px, py, pixelxy, xy = self._grid_xy(npix, width_px, height_px)
        dV = float(self.params.get('dV', .001))
        I = self.current(bias, xy)[0]
        dIdV = (self.current(bias + dV, xy)[0] - self.current(bias - dV, xy)[0]) / (2 * dV)
        return dict(I=I.reshape(len(py), len(px)), dIdV=dIdV.reshape(len(py), len(px)), pixels=(px, py), pixelxy=pixelxy, xy=xy, bias=bias)

    # families: (name, [start0,start1], [dir0,dir1]) — exp averages the two lines
    def _cut_families(self, nfrac):
        a, b = self.cell
        fr = np.linspace(0, 1, int(nfrac))[:, None]
        fams = [('long', [np.zeros(2), b], [a, a]), ('short', [np.zeros(2), a], [b, b]), ('diagonal', [np.zeros(2), b], [a + b, a - b])]
        ends = [a, b, a + b]
        lines = [fr * e for e in ends]
        pix = [[self.physical_to_image(s + fr * d) for s, d in zip(starts, dirs)] for _, starts, dirs in fams]
        return fams, ends, np.vstack(lines), pix

    def scan_cuts(self, voltages, nfrac):
        """dI/dV and I on long/short/diagonal cuts. -> dict(names,dIdV[3,nV,nfrac],I,ends,fractions,pixels)."""
        voltages = np.asarray(voltages, float)
        fams, ends, line, pix = self._cut_families(nfrac)
        allxy = np.tile(line, (len(voltages), 1))
        allv = np.repeat(voltages, len(line))
        dV = float(self.params.get('dV', .001))  # cut_dV is the voltage-grid spacing, not the derivative step
        d = (self.current(allv + dV, allxy)[0] - self.current(allv - dV, allxy)[0]) / (2 * dV)
        I = self.current(allv, allxy)[0]
        nf = len(line) // 3
        return dict(names=[n for n, _, _ in fams], dIdV=d.reshape(len(voltages), 3, nf).transpose(1, 0, 2),
                    I=I.reshape(len(voltages), 3, nf).transpose(1, 0, 2), ends=np.array(ends),
                    fractions=np.linspace(0, 1, nf), voltages=voltages, pixels=pix)

    def site_pixels(self, width, height, margin_cells=1):
        """Pixel coords of all lattice replicas of all basis sites inside the image."""
        n1, n2 = self.ns
        corners_pix = np.array([[0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]])
        fc = self.image_to_physical(corners_pix) @ np.linalg.inv(self.cell)
        i0, i1 = int(np.floor(fc[:, 0].min())) - margin_cells, int(np.ceil(fc[:, 0].max())) + margin_cells
        j0, j1 = int(np.floor(fc[:, 1].min())) - margin_cells, int(np.ceil(fc[:, 1].max())) + margin_cells
        xy = np.array([self.basis[k] + i * self.cell[0] + j * self.cell[1] for k in range(len(self.basis)) for j in range(j0, j1 + 1) for i in range(i0, i1 + 1)])
        pix = self.physical_to_image(xy)
        keep = (pix[:, 0] >= -1) & (pix[:, 0] <= width) & (pix[:, 1] >= -1) & (pix[:, 1] <= height)
        return pix[keep]

    # --- experiment ---------------------------------------------------------
    def experiment_image(self, bias):
        """(raw, mV_used) or (None, None) if exp_dir missing — GUI treats missing as non-fatal."""
        exp_dir = self.params.get('exp_dir')
        if not exp_dir or not os.path.isdir(exp_dir):
            return None, None
        return load_experiment_image(self.params, bias)

    def experiment_on_grid(self, raw, pixelxy, shape):
        """Interpolate raw experiment image onto the scan pixel grid."""
        interp = RegularGridInterpolator((np.arange(raw.shape[0]), np.arange(raw.shape[1])), raw)
        return interp(pixelxy[:, ::-1]).reshape(shape)

    def sample_experiment_cuts(self, voltages, nfrac):
        """Experimental x-V cuts averaged over the two symmetry-related lines.

        Voltages snap to the nearest existing 5 mV TXT file (cached). Returns
        (exp[3,nV,nfrac], mV_used[nV]) or (None, None) if exp_dir missing.
        """
        exp_dir = self.params.get('exp_dir')
        if not exp_dir or not os.path.isdir(exp_dir):
            return None, None
        pattern = self.params.get('exp_pattern', '{mV}mv.txt')
        mV_avail = available_mV(exp_dir, pattern)
        mV_used = np.array([int(mV_avail[np.argmin(np.abs(mV_avail - round(v * 1000)))]) for v in voltages])
        cache = getattr(self, '_exp_txt_cache', None)
        if cache is None:
            cache = self._exp_txt_cache = {}
        _, _, _, pix = self._cut_families(nfrac)
        exp = np.zeros((3, len(voltages), int(nfrac)))
        for iv, mv in enumerate(mV_used):
            if mv not in cache:
                cache[mv] = np.loadtxt(os.path.join(exp_dir, pattern.format(mV=mv)))
            img = cache[mv]
            for f in range(3):
                exp[f, iv] = np.array([map_coordinates(img, [p[:, 1], p[:, 0]], order=1) for p in pix[f]]).mean(axis=0)
        return exp, mV_used

    def image_scores(self, sim, exp, pixel_step):
        """Band-pass correlation scores vs experiment -> dict(signed, magnitude)."""
        return {pol: float(np.mean(np.sum(image_ridge_features(sim, pixel_step, polarity=pol) * image_ridge_features(exp, pixel_step, polarity=pol), axis=1))) for pol in ('signed', 'magnitude')}

    def cut_correlations(self, sim_dIdV, exp):
        """Branch-signal correlation per family -> [3] floats."""
        return [float(np.sum(branch_signal(s) * branch_signal(e))) for s, e in zip(sim_dIdV, exp)]
