#!/usr/bin/env python3
# testplot_pme_tiles.py — L1/L2 visual diagnostics for the contact_pme tiled
# evaluator: system map + bucket(PIC) cell occupancy + per-tile atom/halo
# decomposition + fit-time breakdown. Artifacts -> debug/testplot_pme_tiles/
#
# usage:
#   python tests/SPM/testplot_pme_tiles.py --mol data/xyz/PTCDA.xyz --replica 4 4
#   python tests/SPM/testplot_pme_tiles.py --graphene 40 40 --layers 2 --defect N
#   python tests/SPM/testplot_pme_tiles.py --scale     # fit+scan vs system size
#   python tests/SPM/testplot_pme_tiles.py --gridff    # GridFF voxel build vs PME
#   python tests/SPM/testplot_pme_tiles.py --heights   # relaxed df: GridFF vs 1x/2x/3x
import os, sys, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.environ.setdefault('PYOPENCL_CTX', '0')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   'debug', 'testplot_pme_tiles')
os.makedirs(OUT, exist_ok=True)

A_CC = 1.42

def make_graphene_xyz(nx, ny, layers=1, defect=None, z_gap=3.4, passivate=True, out_path=None):
    """Honeycomb graphene flake (nx,ny unit cells), optional center N defect,
    `layers` sheets stacked at z_gap. Edge atoms passivated with H."""
    a = np.sqrt(3) * A_CC                                   # 2.46 A lattice const
    a1 = np.array([a, 0.0]); a2 = np.array([a / 2, a * np.sqrt(3) / 2])
    basis = [np.array([0.0, 0.0]), np.array([a / 2, a * np.sqrt(3) / 6])]
    pos, el = [], []
    R_C = 1.1 * A_CC                                        # neighbor reach
    raw = []
    for i in range(nx):
        for j in range(ny):
            for b in basis:
                raw.append(i * a1 + j * a2 + b)
    raw = np.array(raw)
    # keep atoms with >=2 neighbors (drop dangling edge atoms, passivate 2-coord ones)
    d2 = ((raw[:, None, :] - raw[None, :, :]) ** 2).sum(-1)
    nnb = ((d2 > 1e-9) & (d2 < R_C ** 2)).sum(axis=1)
    keep = nnb >= 2
    raw = raw[keep]; nnb = nnb[keep]
    raw -= raw.mean(axis=0)
    # defect at the atom nearest the flake center
    idx_c = np.argmin((raw ** 2).sum(axis=1))
    for L in range(layers):
        z = L * z_gap
        for k, p in enumerate(raw):
            e = 'C'
            if defect and L == 0 and k == idx_c:
                e = defect
            pos.append([p[0], p[1], z]); el.append(e)
        if passivate:
            d2k = ((raw[:, None, :] - raw[None, :, :]) ** 2).sum(-1)
            for k, p in enumerate(raw):
                if nnb[k] == 2:
                    nb = np.flatnonzero((d2k[k] > 1e-9) & (d2k[k] < R_C ** 2))
                    v = (raw[nb] - p).mean(axis=0); v /= np.linalg.norm(v)
                    pos.append([p[0] - 1.09 * v[0], p[1] - 1.09 * v[1], z]); el.append('H')
    pos = np.array(pos)
    if out_path:
        with open(out_path, 'w') as f:
            f.write(f"{len(el)}\n\n")
            for e, p in zip(el, pos):
                f.write(f"{e:2s} {p[0]:12.6f} {p[1]:12.6f} {p[2]:12.6f}\n")
        print(f"  wrote {out_path}: {len(el)} atoms", flush=True)
    return pos, el

def pic_stats(afm, tiles):
    """Atom-count decomposition per scan tile: atoms inside the pixel footprint
    vs halo (needed only via r_b reach), plus bucket-cell occupancy."""
    p = afm.cpm
    apos = np.asarray(p.atom_pos); r_b = np.asarray(p.core_fit.r_lo) + p.core_d_span
    nloc = np.diff(tiles['wg_atom_offsets'])
    ntx, nty, tx, ty = tiles['ntx'], tiles['nty'], tiles['tx'], tiles['ty']
    # in-cell atoms: position inside the tile pixel box (without halo)
    scan_p0, scan_da, scan_db = afm._scan_grid_auto((48, 48))
    from spammm.SPM.AFM import build_scan_xy_points_vectorized
    pts = build_scan_xy_points_vectorized(scan_p0, scan_da, scan_db, 48, 48)[:, :3].reshape(48, 48, 3)
    inside = np.zeros_like(nloc)
    for tyi in range(nty):
        for txi in range(ntx):
            t = tyi * ntx + txi
            q = pts[txi * tx:min(txi * tx + tx, 48), tyi * ty:min(tyi * ty + ty, 48)].reshape(-1, 3)
            a0, a1 = tiles['wg_atom_offsets'][t], tiles['wg_atom_offsets'][t + 1]
            ids = tiles['wg_atom_ids'][a0:a1]
            ap = apos[ids]
            m = ((ap[:, 0] >= q[:, 0].min()) & (ap[:, 0] <= q[:, 0].max())
                 & (ap[:, 1] >= q[:, 1].min()) & (ap[:, 1] <= q[:, 1].max()))
            inside[t] = int(m.sum())
    # bucket occupancy
    nb = int(p.bucket_nbx) * int(p.bucket_nby)
    occ = np.diff(np.asarray(p.bucket_offsets)[:nb + 1])
    return dict(nloc=nloc, inside=inside, occ=occ, r_b=r_b, cell=p.bucket_cell_size)

def _kb(n):
    return n / 1024.0

def run_box_partition(afm, mol, rho=1.0, nxy=(48, 48), nz=60, dtip=-0.1, tx=8, ty=4, nzL=6,
                      delta_bs=(2.0, 1.2, 0.6)):
    """In-box / rho-margin / Rcut-halo per WG tile, and the local-memory allocation that follows.

    Recommended Rcut is Δ_b=1.2 (r_b≈R0+1.2). Δ_b=2.0 is the current default ("now").
    Δ_b=0.6 is the more aggressive locality point from the 2026-10-04 sweep.
    The tile kernel allocates nloc_max, not the mean: every workgroup pays for the worst tile.
    """
    import pyopencl as cl
    from spammm.SPM.AFM import build_scan_xy_points_vectorized
    dev = afm.ctx.devices[0]
    lmem = int(dev.get_info(cl.device_info.LOCAL_MEM_SIZE))
    print(f"DEVICE: {dev.name.strip()}  local_mem={lmem} B ({_kb(lmem):.1f} KB)", flush=True)
    nx_s, ny_s = nxy
    scan_p0, scan_da, scan_db = afm._scan_grid_auto(nxy)
    pts = build_scan_xy_points_vectorized(scan_p0, scan_da, scan_db, nx_s, ny_s)
    stem = os.path.basename(mol)[:-4]
    rows = []
    pack = {}
    for db in delta_bs:
        print(f"fit delta_b={db} ...", flush=True)
        t0 = time.perf_counter()
        p = afm.fit_contact_pme(h_mesh=1.0, delta_b=db, bPrint=False)
        print(f"  fit {time.perf_counter()-t0:.2f}s  na={p.na}  r_b=R0+{db}  "
              f"r_b_max={float(np.max(np.asarray(p.core_fit.r_lo)+p.core_d_span)):.3f}", flush=True)
        tiles = afm._pme_build_wg_tiles(p, pts, nx_s, ny_s, nz, dtip, tx=tx, ty=ty, rho_max=rho, nzL=nzL)
        n_in, n_margin, n_halo = afm._pme_tile_inbox_halo(p, tiles)
        nloc = np.diff(tiles['wg_atom_offsets'])
        if not np.array_equal(n_in + n_margin + n_halo, nloc):
            raise RuntimeError("in-box + margin + halo != tile atom list")
        mem = afm._pme_tile_local_bytes(tiles)
        nb = int(p.bucket_nbx) * int(p.bucket_nby)
        occ = np.diff(np.asarray(p.bucket_offsets)[:nb + 1])
        # 3x3 candidates around the cell that owns the origin (bucket path, not local memory)
        x0b, y0b = float(p.bucket_bounds[0]), float(p.bucket_bounds[1])
        cs = float(p.bucket_cell_size)
        bx = min(max(int((0.0 - x0b) / cs), 1), int(p.bucket_nbx) - 2)
        by = min(max(int((0.0 - y0b) / cs), 1), int(p.bucket_nby) - 2)
        nbx = int(p.bucket_nbx)
        cand = 0
        for iy in range(by - 1, by + 2):
            for ix in range(bx - 1, bx + 2):
                cand += int(occ[iy * nbx + ix])
        apos = np.asarray(p.atom_pos)
        ztop = float(apos[:, 2].max())
        below = apos[:, 2] < ztop - 1.0
        n_below = np.zeros(len(nloc), np.int32)
        offs, ids = tiles['wg_atom_offsets'], tiles['wg_atom_ids']
        for t in range(len(nloc)):
            sel = ids[int(offs[t]):int(offs[t + 1])]
            n_below[t] = int(below[sel].sum()) if len(sel) else 0
        row = dict(db=db, r_b=float(np.max(np.asarray(p.core_fit.r_lo) + p.core_d_span)),
                   na=int(p.na), n_in=n_in, n_margin=n_margin, n_halo=n_halo, nloc=nloc, n_below=n_below,
                   mem=mem, occ=occ, cs=cs, cand3x3=cand, lmem=lmem,
                   local_all=int(p.na) * 36)
        rows.append(row)
        pack[db] = (p, tiles, apos, ztop)
        print(f"  tiles {tiles['ntx']}x{tiles['nty']}  "
              f"in {n_in.min()}/{n_in.mean():.1f}/{n_in.max()}  "
              f"margin {n_margin.min()}/{n_margin.mean():.1f}/{n_margin.max()}  "
              f"halo {n_halo.min()}/{n_halo.mean():.1f}/{n_halo.max()}  "
              f"nloc max={nloc.max()}  below-layer max={n_below.max()}  "
              f"local {mem['total']} B ({100.0*mem['total']/lmem:.1f}% of device)", flush=True)

    # ---------- table ----------
    hdr = (f"{'Δ_b':>5} {'r_b':>6} | {'in mean/max':>12} {'margin m/max':>13} {'halo m/max':>12} "
           f"{'nloc max':>8} {'below max':>9} | {'atoms':>7} {'coeffs':>7} {'mesh':>7} {'total':>8} {'%lmem':>6} | "
           f"{'PIC cell':>8} {'PIC max':>7} {'3x3':>5} {'all-atom':>8}")
    print(hdr, flush=True)
    lines = [hdr]
    for r in rows:
        line = (f"{r['db']:5.2f} {r['r_b']:6.2f} | "
                f"{r['n_in'].mean():5.1f}/{r['n_in'].max():<5d} "
                f"{r['n_margin'].mean():6.1f}/{r['n_margin'].max():<5d} "
                f"{r['n_halo'].mean():5.1f}/{r['n_halo'].max():<5d} "
                f"{r['nloc'].max():8d} {r['n_below'].max():9d} | "
                f"{_kb(r['mem']['atoms']):6.2f}K {_kb(r['mem']['coeffs']):6.2f}K {_kb(r['mem']['mesh']):6.2f}K "
                f"{_kb(r['mem']['total']):7.2f}K {100.0*r['mem']['total']/r['lmem']:5.1f}% | "
                f"{r['cs']:8.2f} {int(r['occ'].max()):7d} {r['cand3x3']:5d} {_kb(r['local_all']):7.2f}K")
        print(line, flush=True)
        lines.append(line)
    p0, tiles0, apos0, ztop0 = pack[delta_bs[1] if 1.2 in pack else delta_bs[0]]
    pix, sw = tiles0['tile_pix'], tiles0['tile_sweep']
    geo = (f"geometry: scan {nx_s}x{ny_s} nz={nz} dtip={dtip} tile {tx}x{ty} rho={rho} nzL={nzL}  "
           f"pixel box {pix[0,1]-pix[0,0]:.2f}x{pix[0,3]-pix[0,2]:.2f} A  "
           f"swept xy {sw[0,1]-sw[0,0]:.2f}x{sw[0,3]-sw[0,2]:.2f} A  "
           f"z-sweep [{sw[0,4]:.2f},{sw[0,5]:.2f}]  mesh window max {tiles0['nxL_max']}x{tiles0['nyL_max']}x{nzL}")
    print(geo, flush=True)
    lines.append(geo)
    tab_path = os.path.join(OUT, f"partition_table_{stem}.txt")
    with open(tab_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"REVIEW: {tab_path}", flush=True)

    # ---------- maps: rows = Δ_b, cols = in-box, margin, Rcut halo ----------
    rec = rows[1] if len(rows) > 1 else rows[0]
    ntx, nty = tiles0['ntx'], tiles0['nty']
    extent = [float(pix[:, 0].min()), float(pix[:, 1].max()), float(pix[:, 2].min()), float(pix[:, 3].max())]
    cols = [('n_in', 'in-box (center in pixel rectangle)'),
            ('n_margin', 'margin (center in swept box, outside pixels)'),
            ('n_halo', 'Rcut halo (sphere reaches swept box)')]
    vmax = [max(int(r[k].max()) for r in rows) for k, _ in cols]
    fig, axs = plt.subplots(len(rows), 3, figsize=(14.5, 3.3 * len(rows)), squeeze=False)
    for i, r in enumerate(rows):
        for j, (k, lab) in enumerate(cols):
            ax = axs[i, j]
            data = r[k].reshape(nty, ntx)
            im = ax.imshow(data, origin='lower', cmap='viridis', vmin=0, vmax=max(vmax[j], 1),
                           extent=extent, aspect='equal')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            tag = '  recommended' if abs(r['db'] - 1.2) < 1e-9 else ('  current default' if abs(r['db'] - 2.0) < 1e-9 else '')
            ax.set_title(f"Δ_b={r['db']:.2f}{tag}\n{lab}\nmean {r[k].mean():.1f}  max {r[k].max()}")
            ax.set_xlabel('x [Å]'); ax.set_ylabel('y [Å]')
    fig.suptitle(f"{stem}  WG tiles {tx}×{ty} px, ρ={rho} Å   color = atoms preloaded into that workgroup", y=1.01)
    fig.tight_layout()
    fn_map = os.path.join(OUT, f"partition_maps_{stem}.png")
    fig.savefig(fn_map, dpi=140, bbox_inches='tight'); plt.close(fig)
    print(f"REVIEW: {fn_map}", flush=True)

    # ---------- one interior tile at the recommended Rcut ----------
    db_rec = 1.2 if 1.2 in pack else delta_bs[0]
    p, tiles, apos, ztop = pack[db_rec]
    rrow = next(r for r in rows if abs(r['db'] - db_rec) < 1e-9)
    t = int(np.argmax(rrow['nloc']))   # the tile that sets the local-memory allocation
    offs, ids = tiles['wg_atom_offsets'], tiles['wg_atom_ids']
    sel = ids[int(offs[t]):int(offs[t + 1])]
    ap = apos[sel]
    xl, xh, yl, yh = tiles['tile_pix'][t]
    sxl, sxh, syl, syh, zbl, zbh = tiles['tile_sweep'][t]
    in_pix = (ap[:, 0] >= xl) & (ap[:, 0] <= xh) & (ap[:, 1] >= yl) & (ap[:, 1] <= yh)
    in_sw = (ap[:, 0] >= sxl) & (ap[:, 0] <= sxh) & (ap[:, 1] >= syl) & (ap[:, 1] <= syh)
    r_b = np.asarray(p.core_fit.r_lo) + p.core_d_span
    listed = np.zeros(len(apos), bool); listed[sel] = True
    fig, ax2 = plt.subplots(1, 2, figsize=(14.5, 7.0))
    th = np.linspace(0, 2 * np.pi, 80)
    for ax, j, ylab in ((ax2[0], 1, 'y [Å]'), (ax2[1], 2, 'z [Å]')):
        ax.scatter(apos[~listed, 0], apos[~listed, j], s=10, c='0.82', label=f'excluded  n={int((~listed).sum())}', zorder=1)
        ax.scatter(ap[~in_sw, 0], ap[~in_sw, j], s=22, c='darkorange', label=f'Rcut halo  n={int((~in_sw).sum())}', zorder=3)
        ax.scatter(ap[in_sw & ~in_pix, 0], ap[in_sw & ~in_pix, j], s=22, c='deepskyblue',
                   label=f'ρ margin  n={int((in_sw & ~in_pix).sum())}', zorder=3)
        ax.scatter(ap[in_pix, 0], ap[in_pix, j], s=28, c='seagreen', label=f'in-box  n={int(in_pix.sum())}', zorder=4)
        ax.set_aspect('equal'); ax.set_xlabel('x [Å]'); ax.set_ylabel(ylab)
    for ia in np.asarray(sel)[in_pix]:
        ax2[0].plot(apos[ia, 0] + r_b[ia] * np.cos(th), apos[ia, 1] + r_b[ia] * np.sin(th),
                    color='seagreen', lw=0.5, alpha=0.7)
    for ia in np.asarray(sel)[in_sw & ~in_pix]:
        ax2[0].plot(apos[ia, 0] + r_b[ia] * np.cos(th), apos[ia, 1] + r_b[ia] * np.sin(th),
                    color='deepskyblue', lw=0.35, alpha=0.45)
    for tt in range(len(tiles['tile_pix'])):
        q = tiles['tile_pix'][tt]
        ax2[0].add_patch(plt.Rectangle((q[0], q[2]), q[1] - q[0], q[3] - q[2], fill=False, ec='0.6', lw=0.4))
    ax2[0].add_patch(plt.Rectangle((xl, yl), xh - xl, yh - yl, fill=False, ec='crimson', lw=1.8, label='pixel box (this WG)'))
    ax2[0].add_patch(plt.Rectangle((sxl, syl), sxh - sxl, syh - syl, fill=False, ec='deepskyblue', lw=1.2, ls='--',
                                   label='swept box (pixels + ρ)'))
    ax2[1].add_patch(plt.Rectangle((xl, zbl), xh - xl, zbh - zbl, fill=False, ec='crimson', lw=1.6, label='pixel x × PP z-box'))
    ax2[1].axhline(ztop, color='0.3', lw=0.7, ls=':', label=f'top atom z={ztop:.2f}')
    rb = float(r_b.max())
    ax2[0].set_xlim(sxl - rb - 0.4, sxh + rb + 0.4)
    ax2[0].set_ylim(syl - rb - 0.4, syh + rb + 0.4)
    ax2[1].set_xlim(sxl - rb - 0.4, sxh + rb + 0.4)
    ax2[0].legend(loc='upper left', fontsize=7, framealpha=0.9)
    ax2[1].legend(loc='upper right', fontsize=7, framealpha=0.9)
    ax2[0].set_title(f'worst WG (sets local mem) t={t}  Δ_b={db_rec}  r_b={rb:.2f} Å\n'
                     f'in-box {int(in_pix.sum())} + margin {int((in_sw & ~in_pix).sum())} + halo {int((~in_sw).sum())} '
                     f'= {len(sel)} preloaded   (na={p.na})')
    ax2[1].set_title(f'XZ  PP z-box [{zbl:.2f}, {zbh:.2f}] Å\n'
                     f'of these, z < {ztop-1:.1f} (lower layer): {int((ap[:,2] < ztop-1).sum())}')
    fig.tight_layout()
    fn_tile = os.path.join(OUT, f"partition_tile_db{db_rec}_{stem}.png")
    fig.savefig(fn_tile, dpi=140); plt.close(fig)
    print(f"REVIEW: {fn_tile}", flush=True)

    # ---------- local memory bars ----------
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    labels, b_at, b_co, b_me = [], [], [], []
    for r in rows:
        labels.append(f"Δ_b={r['db']:.1f}\nnloc={r['nloc'].max()}")
        b_at.append(_kb(r['mem']['atoms'])); b_co.append(_kb(r['mem']['coeffs'])); b_me.append(_kb(r['mem']['mesh']))
    labels.append(f"all-atom\nLocal kernel\nna={rows[0]['na']}")
    b_at.append(_kb(rows[0]['na'] * 16)); b_co.append(_kb(rows[0]['na'] * 20)); b_me.append(0.0)
    x = np.arange(len(labels))
    ax.bar(x, b_at, color='seagreen', label='LATOMS  nloc×16 B')
    ax.bar(x, b_co, bottom=b_at, color='darkorange', label='LCOEFFS  nloc×5×4 B')
    bot = np.array(b_at) + np.array(b_co)
    ax.bar(x, b_me, bottom=bot, color='steelblue', label=f'LMESH  {tiles0["nxL_max"]}×{tiles0["nyL_max"]}×{nzL}×4 B')
    ax.axhline(_kb(lmem), color='crimson', ls='--', lw=1.2, label=f'device local mem {_kb(lmem):.0f} KB')
    for i, r in enumerate(rows):
        ax.text(i, _kb(r['mem']['total']) + 0.4, f"{_kb(r['mem']['total']):.1f} KB", ha='center', fontsize=8)
    ax.text(len(rows), _kb(rows[0]['local_all']) + 0.4, f"{_kb(rows[0]['local_all']):.1f} KB", ha='center', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel('local memory [KB]')
    ax.set_title(f'{stem}: tile-kernel local allocation (worst tile, not the mean)\n'
                 f'mesh slab does not shrink with Rcut')
    ax.legend(fontsize=8, loc='upper right')
    fig.tight_layout()
    fn_mem = os.path.join(OUT, f"local_mem_{stem}.png")
    fig.savefig(fn_mem, dpi=140); plt.close(fig)
    print(f"REVIEW: {fn_mem}", flush=True)
    np.savez(os.path.join(OUT, f"partition_{stem}.npz"),
             delta_b=np.array([r['db'] for r in rows]),
             n_in=np.stack([r['n_in'] for r in rows]),
             n_margin=np.stack([r['n_margin'] for r in rows]),
             n_halo=np.stack([r['n_halo'] for r in rows]),
             nloc=np.stack([r['nloc'] for r in rows]))
    plot_count_proof(pack, stem)
    print("partition done", flush=True)


def _offset_rect(xl, xh, yl, yh, r, n=48):
    """Closed curve of points at distance r outside the rectangle (straight sides, circular corners)."""
    if r <= 0:
        return np.array([xl, xh, xh, xl, xl]), np.array([yl, yl, yh, yh, yl])
    a = np.linspace(-np.pi / 2, 0, n)
    b = np.linspace(0, np.pi / 2, n)
    c = np.linspace(np.pi / 2, np.pi, n)
    d = np.linspace(np.pi, 3 * np.pi / 2, n)
    xs = np.concatenate([
        [xl], np.linspace(xl, xh, 2), xh + r * np.cos(a),
        np.full(2, xh + r), xh + r * np.cos(b),
        np.linspace(xh, xl, 2), xl + r * np.cos(c),
        np.full(2, xl - r), xl + r * np.cos(d),
    ])
    ys = np.concatenate([
        [yl - r], np.full(2, yl - r), yl + r * np.sin(a),
        np.linspace(yl, yh, 2), yh + r * np.sin(b),
        np.full(2, yh + r), yh + r * np.sin(c),
        np.linspace(yh, yl, 2), yl + r * np.sin(d),
    ])
    return xs, ys


def plot_count_proof(pack, stem):
    """One tile, every loaded atom drawn, plus a real 12×12 Å square on the same lattice.

    The 145 figure is the Rcut halo of the Δ_b=2 tile, not the atoms inside a 4 Å box.
    """
    db = 2.0 if 2.0 in pack else next(iter(pack))
    p, tiles, apos, ztop = pack[db]
    apos = np.asarray(apos, np.float64)
    r_b = np.asarray(p.core_fit.r_lo, np.float64) + float(p.core_d_span)
    rb = float(np.max(r_b))
    n_in, n_margin, n_halo = None, None, None
    # recompute split the same way the table did
    offs, ids = tiles['wg_atom_offsets'], tiles['wg_atom_ids']
    nloc = np.diff(offs)
    pix, sw = tiles['tile_pix'], tiles['tile_sweep']
    n_halo = np.zeros(len(nloc), np.int32)
    for t in range(len(nloc)):
        sel = ids[int(offs[t]):int(offs[t + 1])]
        if len(sel) == 0:
            continue
        ap = apos[sel]
        in_sw = (ap[:, 0] >= sw[t, 0]) & (ap[:, 0] <= sw[t, 1]) & (ap[:, 1] >= sw[t, 2]) & (ap[:, 1] <= sw[t, 3])
        n_halo[t] = int((~in_sw).sum())
    t = int(np.argmax(n_halo))
    sel = np.asarray(ids[int(offs[t]):int(offs[t + 1])])
    xl, xh, yl, yh = pix[t]
    sxl, sxh, syl, syh, zbl, zbh = sw[t]
    top = apos[:, 2] > ztop - 1.0
    in_list = np.zeros(len(apos), bool); in_list[sel] = True
    in_sw = (apos[:, 0] >= sxl) & (apos[:, 0] <= sxh) & (apos[:, 1] >= syl) & (apos[:, 1] <= syh)
    in_pix = (apos[:, 0] >= xl) & (apos[:, 0] <= xh) & (apos[:, 1] >= yl) & (apos[:, 1] <= yh)
    # independent 3D distance to the swept box, per-atom r_b — must match the list
    c = np.array([(sxl + sxh) / 2, (syl + syh) / 2, (zbl + zbh) / 2])
    hw = np.array([(sxh - sxl) / 2, (syh - syl) / 2, (zbh - zbl) / 2])
    d = np.maximum(np.abs(apos - c) - hw, 0.0)
    reach = (d * d).sum(axis=1) < r_b * r_b
    if not np.array_equal(np.flatnonzero(reach), np.sort(sel)):
        raise RuntimeError(f"list/reach mismatch tile {t}: list {len(sel)} reach {int(reach.sum())}")
    cx, cy = 0.5 * (sxl + sxh), 0.5 * (syl + syh)
    sq = 6.0
    in12 = (np.abs(apos[:, 0] - cx) <= sq) & (np.abs(apos[:, 1] - cy) <= sq)
    def _n(mask):
        return int((mask & top).sum()), int((mask & ~top).sum()), int(mask.sum())
    nt, nb, nall = _n(in_list)
    ht, hb, hall = _n(in_list & ~in_sw)
    pt, pb, pall = _n(in_list & in_pix)
    st, sb, sall = _n(in12)
    print(f"PROOF tile t={t} Δ_b={db} r_b={rb:.2f}", flush=True)
    print(f"  pixel box {xh-xl:.2f} x {yh-yl:.2f} Å   atoms with center inside it: {pall} (top {pt}, bottom {pb})", flush=True)
    print(f"  swept box {sxh-sxl:.2f} x {syh-syl:.2f} Å", flush=True)
    print(f"  preloaded {nall} = top {nt} + bottom {nb}", flush=True)
    print(f"  of those, halo (center outside swept box) {hall} = top {ht} + bottom {hb}", flush=True)
    print(f"  12x12 Å square on the same center: {sall} = top {st} + bottom {sb}", flush=True)
    # bonds inside the view, same layer, C–C ~1.42
    view = (np.abs(apos[:, 0] - cx) < (sxh - sxl) / 2 + rb + 1.5) & (np.abs(apos[:, 1] - cy) < (syh - syl) / 2 + rb + 1.5)
    iv = np.flatnonzero(view)
    bonds = []
    if len(iv):
        xy = apos[iv, :2]
        zz = apos[iv, 2]
        d2 = (xy[:, None, 0] - xy[None, :, 0]) ** 2 + (xy[:, None, 1] - xy[None, :, 1]) ** 2
        same = np.abs(zz[:, None] - zz[None, :]) < 0.5
        iu, ju = np.where(np.triu(same & (d2 > 0.5) & (d2 < 1.7 ** 2), 1))
        bonds = (iv[iu], iv[ju])

    z_bot = float(np.median(apos[~top, 2])) if np.any(~top) else 0.0
    r_bot = float(np.sqrt(max(rb * rb - max(zbl - z_bot, 0.0) ** 2, 0.0)))

    def _panel(ax, layer, mask_draw, title, draw_12):
        lay = top if layer == 'top' else ~top
        reach_r = rb if layer == 'top' else r_bot
        if len(bonds):
            for ia, ja in zip(*bonds):
                if lay[ia] and lay[ja]:
                    ax.plot([apos[ia, 0], apos[ja, 0]], [apos[ia, 1], apos[ja, 1]], color='0.75', lw=0.7, zorder=1)
        m = mask_draw & lay
        ax.scatter(apos[m, 0], apos[m, 1], s=42, c=('darkorange' if layer == 'top' else 'royalblue'),
                   marker='o', zorder=4)
        ax.add_patch(plt.Rectangle((xl, yl), xh - xl, yh - yl, fill=False, ec='crimson', lw=2.0, zorder=5))
        ax.add_patch(plt.Rectangle((sxl, syl), sxh - sxl, syh - syl, fill=False, ec='deepskyblue', lw=1.4, ls='--', zorder=5))
        xs, ys = _offset_rect(sxl, sxh, syl, syh, reach_r)
        ax.plot(xs, ys, color='k', lw=1.1, zorder=5)
        if draw_12:
            ax.add_patch(plt.Rectangle((cx - sq, cy - sq), 2 * sq, 2 * sq, fill=False, ec='green', lw=1.8, zorder=6))
        ax.set_aspect('equal')
        pad = rb + 1.6
        ax.set_xlim(cx - (sxh - sxl) / 2 - pad, cx + (sxh - sxl) / 2 + pad)
        ax.set_ylim(cy - (syh - syl) / 2 - pad, cy + (syh - syl) / 2 + pad)
        ax.set_xlabel('x [Å]'); ax.set_ylabel('y [Å]')
        ax.set_title(title)

    fig, ax = plt.subplots(2, 2, figsize=(15.5, 14.5))
    _panel(ax[0, 0], 'top', in_list, f'TOP layer, atoms this tile preloads: {nt}\n(halo outside blue box: {ht}; inside red box: {pt})', False)
    _panel(ax[1, 0], 'bottom', in_list, f'BOTTOM layer, same tile: {nb}\n(halo outside blue box: {hb}; inside red box: {pb})\nAA stack — same XY as the top layer, so it was hidden if drawn together', False)
    _panel(ax[0, 1], 'top', in12, f'TOP layer inside the green 12×12 Å square: {st}', True)
    _panel(ax[1, 1], 'bottom', in12, f'BOTTOM layer inside that same square: {sb}\n12×12 both layers = {sall}', True)
    fig.suptitle(f'{stem}  Δ_b={db}   preload {nall} = {nt} top + {nb} bottom    halo {hall} = {ht} top + {hb} bottom\n'
                 f'red = pixel strip {xh-xl:.1f}×{yh-yl:.1f} Å    blue = swept box {sxh-sxl:.1f}×{syh-syl:.1f} Å    '
                 f'black = {rb:.1f} Å outside the blue box (bottom panel {r_bot:.1f} Å, the sheet is {zbl-z_bot:.1f} Å below the box)    green = 12×12 Å\n'
                 f'layers are drawn apart because this flake is AA-stacked: same XY, so one picture hides half the atoms',
                 fontsize=11)
    fig.tight_layout()
    fn = os.path.join(OUT, f'count_proof_{stem}.png')
    fig.savefig(fn, dpi=140)
    plt.close(fig)
    print(f'REVIEW: {fn}', flush=True)


def plot_pic_kernel(afm, mol, delta_b=1.2):
    """The partition cs_pme_core_eval_at actually walks.

    Square PIC cells of side r_b (the compact-core cutoff). A query reads the
    cell it sits in plus the 8 neighbors. Contact core is the top sheet only.
    """
    print(f'PIC kernel partition  delta_b={delta_b}', flush=True)
    p = afm.fit_contact_pme(h_mesh=1.0, delta_b=delta_b, bPrint=False)
    apos = np.asarray(p.atom_pos, np.float64)
    ztop = float(apos[:, 2].max())
    top = apos[:, 2] > ztop - 1.0
    cell = float(p.bucket_cell_size)
    x0, y0 = float(p.bucket_bounds[0]), float(p.bucket_bounds[1])
    nbx, nby = int(p.bucket_nbx), int(p.bucket_nby)
    bx = np.clip(((apos[:, 0] - x0) / cell).astype(int), 0, nbx - 1)
    by = np.clip(((apos[:, 1] - y0) / cell).astype(int), 0, nby - 1)
    occ = np.zeros((nby, nbx), np.int32)
    for iy, ix in zip(by[top], bx[top]):
        occ[iy, ix] += 1
    # interior cell with the most top-sheet atoms, so the 3x3 is complete
    best, t_iy, t_ix = -1, nby // 2, nbx // 2
    for iy in range(1, nby - 1):
        for ix in range(1, nbx - 1):
            if occ[iy, ix] > best:
                best, t_iy, t_ix = int(occ[iy, ix]), iy, ix
    home = top & (bx == t_ix) & (by == t_iy)
    halo = top & (np.abs(bx - t_ix) <= 1) & (np.abs(by - t_iy) <= 1) & ~home
    block = home | halo
    bot_in_block = (~top) & (np.abs(bx - t_ix) <= 1) & (np.abs(by - t_iy) <= 1)
    print(f'  cell = r_b = {cell:.3f} Å    3x3 side = {3 * cell:.2f} Å', flush=True)
    print(f'  home cell ({t_ix},{t_iy}): {int(home.sum())} top-sheet atoms', flush=True)
    print(f'  8 neighbor cells: {int(halo.sum())} top-sheet atoms', flush=True)
    print(f'  kernel candidate list, top sheet: {int(block.sum())}', flush=True)
    print(f'  bottom-sheet atoms sitting in the same 9 cells (not contact core): {int(bot_in_block.sum())}', flush=True)
    print('  3x3 occupancy (top sheet):', flush=True)
    print(occ[t_iy - 1:t_iy + 2, t_ix - 1:t_ix + 2], flush=True)

    # bonds of the top sheet, so the lattice spacing is visible
    iv = np.flatnonzero(top)
    xy, zz = apos[iv, :2], apos[iv, 2]
    d2 = (xy[:, None, 0] - xy[None, :, 0]) ** 2 + (xy[:, None, 1] - xy[None, :, 1]) ** 2
    same = np.abs(zz[:, None] - zz[None, :]) < 0.5
    iu, ju = np.where(np.triu(same & (d2 > 0.5) & (d2 < 1.7 ** 2), 1))

    def _draw(ax, xlim, ylim):
        for ia, ja in zip(iv[iu], iv[ju]):
            ax.plot([apos[ia, 0], apos[ja, 0]], [apos[ia, 1], apos[ja, 1]], color='0.82', lw=0.6, zorder=1)
        for iy in range(nby + 1):
            ax.plot([x0, x0 + nbx * cell], [y0 + iy * cell, y0 + iy * cell], color='0.75', lw=0.4, zorder=2)
        for ix in range(nbx + 1):
            ax.plot([x0 + ix * cell, x0 + ix * cell], [y0, y0 + nby * cell], color='0.75', lw=0.4, zorder=2)
        ax.add_patch(plt.Rectangle((x0 + (t_ix - 1) * cell, y0 + (t_iy - 1) * cell), 3 * cell, 3 * cell,
                                   fill=False, ec='k', lw=1.8, zorder=5))
        ax.add_patch(plt.Rectangle((x0 + t_ix * cell, y0 + t_iy * cell), cell, cell,
                                   fill=True, fc='yellow', alpha=0.35, ec='crimson', lw=2.0, zorder=3))
        rest = top & ~block
        ax.scatter(apos[rest, 0], apos[rest, 1], s=18, c='0.55', zorder=4)
        ax.scatter(apos[halo, 0], apos[halo, 1], s=28, c='darkorange', zorder=4)
        ax.scatter(apos[home, 0], apos[home, 1], s=36, c='seagreen', zorder=5)
        ax.set_aspect('equal')
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_xlabel('x [Å]'); ax.set_ylabel('y [Å]')

    fig, ax = plt.subplots(1, 2, figsize=(14.5, 7.2))
    _draw(ax[0], (float(apos[top, 0].min()) - 1, float(apos[top, 0].max()) + 1),
          (float(apos[top, 1].min()) - 1, float(apos[top, 1].max()) + 1))
    ax[0].set_title(f'top sheet only, every PIC cell is {cell:.2f}×{cell:.2f} Å\n'
                    f'grey = other cells    orange = 8 neighbors    green = home cell')
    m = 0.4
    _draw(ax[1], (x0 + (t_ix - 1) * cell - m, x0 + (t_ix + 2) * cell + m),
          (y0 + (t_iy - 1) * cell - m, y0 + (t_iy + 2) * cell + m))
    ax[1].set_title(f'what one query reads: home {int(home.sum())} + neighbors {int(halo.sum())} = {int(block.sum())} atoms\n'
                    f'black square is the 3×3, side {3 * cell:.2f} Å')
    fig.suptitle(f'OpenCL cs_pme_core_eval_at    cell = r_b = {cell:.2f} Å (Δ_b={delta_b})    '
                 f'bottom sheet is not in this picture ({int((~top).sum())} atoms, not contact core)',
                 fontsize=12)
    fig.tight_layout()
    stem = os.path.basename(mol)[:-4]
    base = os.path.join(OUT, f'pic_3x3_{stem}')
    fig.savefig(base + '.png', dpi=140)
    fig.savefig(base + '.svg')
    fig.savefig(base + '.pdf')
    plt.close(fig)
    html = os.path.join(OUT, f'pic_3x3_{stem}.html')
    with open(html, 'w') as f:
        f.write('<!DOCTYPE html><meta charset="utf-8"><title>PIC 3x3</title>\n'
                '<body style="font-family:sans-serif">\n'
                f'<h3>cell = {cell:.2f} Å, 3×3 = {3*cell:.2f} Å, '
                f'home {int(home.sum())} + halo {int(halo.sum())} = {int(block.sum())} top-sheet atoms</h3>\n'
                f'<img src="pic_3x3_{stem}.svg" width="1200">\n</body>\n')
    for ext in ('.png', '.svg', '.pdf', '.html'):
        print(f'REVIEW: {base}{ext}' if ext != '.html' else f'REVIEW: {html}', flush=True)


def _factor_tiles(n):
    """(tx, ty) with tx*ty == n, most compact first in pixel count (aspect closest to 1)."""
    pairs = []
    for tx in range(1, n + 1):
        if n % tx:
            continue
        ty = n // tx
        pairs.append((max(tx, ty) / min(tx, ty), tx, ty))
    pairs.sort()
    return [(tx, ty) for _, tx, ty in pairs]


def sweep_compact_tiles(afm, mol, delta_b=1.2, rho=1.0, nxy=(48, 48), nz=60, dtip=-0.1, nzL=6):
    """Local memory vs workgroup shape. A CU holds ~1024 threads and 48 KB local.

    32 threads/WG → 32 resident WGs → ~1.5 KB each. 256 threads/WG → 4 WGs → ~12 KB each.
    The pixel tile must be compact in Å, not a long rectangle: every thread in the WG
    shares one atom list and one B-spline slab.
    """
    from spammm.SPM.AFM import build_scan_xy_points_vectorized
    p = afm.fit_contact_pme(h_mesh=1.0, delta_b=delta_b, bPrint=False)
    nx_s, ny_s = nxy
    scan_p0, scan_da, scan_db = afm._scan_grid_auto(nxy)
    pts = build_scan_xy_points_vectorized(scan_p0, scan_da, scan_db, nx_s, ny_s)
    dx, dy = abs(float(scan_da[0])), abs(float(scan_db[1]))
    apos = np.asarray(p.atom_pos, np.float64)
    ztop = float(apos[:, 2].max())
    top = apos[:, 2] > ztop - 1.0
    print(f'pixel pitch dx={dx:.3f} Å  dy={dy:.3f} Å   r_b={p.bucket_cell_size:.2f} Å', flush=True)
    shapes = [(8, 4)]
    for n in (32, 64, 128, 256):
        for tx, ty in _factor_tiles(n):
            if (tx, ty) not in shapes:
                shapes.append((tx, ty))
    # keep the square-pixel one and the most square-in-Å one per thread count, plus 8x4
    keep = {(8, 4)}
    for n in (32, 64, 128, 256):
        fac = _factor_tiles(n)
        keep.add(fac[0])  # closest to square in pixels
        best = min(fac, key=lambda wh: abs(np.log((wh[0] * dx) / (wh[1] * dy))))
        keep.add(best)
    rows = []
    print(f'{"tx":>4} {"ty":>4} {"thr":>5} {"WxH Å":>14} {"aspect":>7} | {"nloc":>5} {"top":>5} | '
          f'{"mesh":>8} {"atoms":>8} {"total":>8} | {"WGs/CU":>7} {"budget48":>8}', flush=True)
    for tx, ty in shapes:
        if (tx, ty) not in keep:
            continue
        tiles = afm._pme_build_wg_tiles(p, pts, nx_s, ny_s, nz, dtip, tx=tx, ty=ty, rho_max=rho, nzL=nzL)
        mem = afm._pme_tile_local_bytes(tiles)
        offs, ids = tiles['wg_atom_offsets'], tiles['wg_atom_ids']
        nloc = np.diff(offs)
        t = int(np.argmax(nloc))
        sel = ids[int(offs[t]):int(offs[t + 1])]
        n_top = int(top[sel].sum()) if len(sel) else 0
        pix = tiles['tile_pix'][t]
        Wx, Hy = float(pix[1] - pix[0]), float(pix[3] - pix[2])
        aspect = max(Wx, Hy) / max(min(Wx, Hy), 1e-9)
        thr = tx * ty
        nwg = max(1, 1024 // thr)
        budget = 48 * 1024 / nwg
        row = dict(tx=tx, ty=ty, thr=thr, Wx=Wx, Hy=Hy, aspect=aspect, nloc=int(nloc.max()),
                   n_top=n_top, mem=mem, budget=budget, nwg=nwg, tiles=tiles, t=t)
        rows.append(row)
        print(f'{tx:4d} {ty:4d} {thr:5d} {Wx:6.2f}×{Hy:<6.2f} {aspect:7.2f} | {int(nloc.max()):5d} {n_top:5d} | '
              f'{mem["mesh"]/1024:7.2f}K { (mem["atoms"]+mem["coeffs"])/1024:7.2f}K {mem["total"]/1024:7.2f}K | '
              f'{nwg:7d} {budget/1024:7.1f}K', flush=True)
    # figure: current 8x4 vs the 256-thread tile closest to square in Å
    cur = next(r for r in rows if (r['tx'], r['ty']) == (8, 4))
    big = [r for r in rows if r['thr'] == 256]
    sq = min(big, key=lambda r: r['aspect'])
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 6.4))
    for a, r, title in ((ax[0], cur, 'current 8×4'), (ax[1], sq, f'compact {sq["tx"]}×{sq["ty"]}')):
        tiles, t = r['tiles'], r['t']
        xl, xh, yl, yh = tiles['tile_pix'][t]
        sxl, sxh, syl, syh = tiles['tile_sweep'][t, :4]
        offs, ids = tiles['wg_atom_offsets'], tiles['wg_atom_ids']
        sel = np.asarray(ids[int(offs[t]):int(offs[t + 1])])
        sel = sel[top[sel]] if len(sel) else sel
        a.scatter(apos[top, 0], apos[top, 1], s=8, c='0.8')
        if len(sel):
            a.scatter(apos[sel, 0], apos[sel, 1], s=16, c='darkorange')
        a.add_patch(plt.Rectangle((xl, yl), xh - xl, yh - yl, fill=False, ec='crimson', lw=2))
        a.add_patch(plt.Rectangle((sxl, syl), sxh - sxl, syh - syl, fill=False, ec='deepskyblue', lw=1.2, ls='--'))
        a.set_aspect('equal')
        a.set_xlim(sxl - p.bucket_cell_size - 0.5, sxh + p.bucket_cell_size + 0.5)
        a.set_ylim(syl - p.bucket_cell_size - 0.5, syh + p.bucket_cell_size + 0.5)
        a.set_xlabel('x [Å]'); a.set_ylabel('y [Å]')
        a.set_title(f'{title}\n{r["Wx"]:.2f}×{r["Hy"]:.2f} Å   top-sheet atoms in worst WG: {r["n_top"]}\n'
                    f'local {r["mem"]["total"]/1024:.1f} KB   CU share at {r["thr"]} threads: {r["budget"]/1024:.1f} KB')
    fig.suptitle('red = pixel tile shared by one workgroup    blue = tile + ρ    orange = top-sheet atoms that tile preloads', fontsize=11)
    fig.tight_layout()
    fn = os.path.join(OUT, 'tile_compact_graphene_10x10x2L_N.png')
    fig.savefig(fn, dpi=140)
    plt.close(fig)
    print(f'REVIEW: {fn}', flush=True)
    return rows


def plot_pitch_tile(afm, mol, pitch=0.1, tx=16, ty=16, delta_b=1.2, rho=1.0, nz=60, dtip=-0.1, nzL=6):
    """One workgroup tile on a real AFM grid: pitch 0.1 Å, so 16×16 pixels = 1.6×1.6 Å."""
    from spammm.SPM.AFM import build_scan_xy_points_vectorized
    p = afm.fit_contact_pme(h_mesh=1.0, delta_b=delta_b, bPrint=False)
    apos = np.asarray(p.atom_pos, np.float64)
    ztop = float(apos[:, 2].max())
    # enough pixels that a full tx×ty tile sits in the interior
    n = max(tx, ty) * 3
    span = (n - 1) * pitch
    z0 = ztop + 5.0 + abs(float(afm.dpos0[2]))
    p0 = np.array([-span / 2, -span / 2, z0], np.float32)
    da = np.array([pitch, 0.0, 0.0], np.float32)
    db = np.array([0.0, pitch, 0.0], np.float32)
    pts = build_scan_xy_points_vectorized(p0, da, db, n, n)
    tiles = afm._pme_build_wg_tiles(p, pts, n, n, nz, dtip, tx=tx, ty=ty, rho_max=rho, nzL=nzL)
    nloc = np.diff(tiles['wg_atom_offsets'])
    # interior tile: full tx by ty, closest to the origin
    ntx, nty = tiles['ntx'], tiles['nty']
    best, td = 1e9, 0
    for t in range(ntx * nty):
        xl, xh, yl, yh = tiles['tile_pix'][t]
        if abs((xh - xl) - (tx - 1) * pitch) > 1e-4 or abs((yh - yl) - (ty - 1) * pitch) > 1e-4:
            continue
        d = abs(0.5 * (xl + xh)) + abs(0.5 * (yl + yh))
        if d < best:
            best, td = d, t
    xl, xh, yl, yh = tiles['tile_pix'][td]
    # pixel block is n * pitch (each sample owns pitch/2 on either side)
    box = (xl - 0.5 * pitch, xh + 0.5 * pitch, yl - 0.5 * pitch, yh + 0.5 * pitch)
    offs, ids = tiles['wg_atom_offsets'], tiles['wg_atom_ids']
    sel = np.asarray(ids[int(offs[td]):int(offs[td + 1])])
    top = apos[:, 2] > ztop - 1.0
    n_top = int(top[sel].sum()) if len(sel) else 0
    mem = afm._pme_tile_local_bytes(tiles)
    Wx, Hy = box[1] - box[0], box[3] - box[2]
    print(f'pitch={pitch} Å   tile {tx}×{ty} pixels   block {Wx:.2f}×{Hy:.2f} Å', flush=True)
    print(f'  sample span (first to last) {(xh-xl):.2f}×{(yh-yl):.2f} Å = ({tx}-1)×{pitch}', flush=True)
    print(f'  worst-list nloc_max={int(nloc.max())}   this tile n={len(sel)} top-sheet={n_top} bottom={len(sel)-n_top}', flush=True)
    print(f'  mesh {tiles["nxL_max"]}×{tiles["nyL_max"]}×{nzL}   local {mem["total"]/1024:.2f} KB '
          f'(atoms+coeffs {(mem["atoms"]+mem["coeffs"])/1024:.2f} KB, mesh {mem["mesh"]/1024:.2f} KB)', flush=True)
    iv = np.flatnonzero(top)
    xy = apos[iv, :2]
    d2 = (xy[:, None, 0] - xy[None, :, 0]) ** 2 + (xy[:, None, 1] - xy[None, :, 1]) ** 2
    iu, ju = np.where(np.triu((d2 > 0.5) & (d2 < 1.7 ** 2), 1))
    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    for ia, ja in zip(iv[iu], iv[ju]):
        ax.plot([apos[ia, 0], apos[ja, 0]], [apos[ia, 1], apos[ja, 1]], color='0.8', lw=0.7, zorder=1)
    ax.scatter(apos[top, 0], apos[top, 1], s=18, c='0.65', zorder=2)
    if len(sel):
        ax.scatter(apos[sel][top[sel], 0], apos[sel][top[sel], 1], s=28, c='darkorange', zorder=3, label=f'top sheet preloaded  {n_top}')
    ax.add_patch(plt.Rectangle((box[0], box[2]), Wx, Hy, fill=False, ec='crimson', lw=2.0, zorder=4,
                               label=f'{tx}×{ty} pixels × {pitch} Å  =  {Wx:.1f}×{Hy:.1f} Å'))
    ax.set_aspect('equal')
    m = p.bucket_cell_size + 0.8
    ax.set_xlim(0.5 * (box[0] + box[1]) - m, 0.5 * (box[0] + box[1]) + m)
    ax.set_ylim(0.5 * (box[2] + box[3]) - m, 0.5 * (box[2] + box[3]) + m)
    ax.set_xlabel('x [Å]'); ax.set_ylabel('y [Å]')
    ax.set_title(f'{pitch} Å / pixel    {tx}×{ty} workgroup is {Wx:.1f}×{Hy:.1f} Å\n'
                 f'orange = top-sheet atoms whose core reaches this tile ({n_top})')
    ax.legend(loc='upper right', fontsize=8)
    fig.tight_layout()
    fn = os.path.join(OUT, f'tile_{tx}x{ty}_pitch{pitch:.2f}.png')
    fig.savefig(fn, dpi=150)
    plt.close(fig)
    print(f'REVIEW: {fn}', flush=True)


def write_xyz(path, pos, el):
    with open(path, 'w') as f:
        f.write(f"{len(el)}\n\n")
        for e, p in zip(el, pos):
            f.write(f"{e:2s} {p[0]:12.6f} {p[1]:12.6f} {p[2]:12.6f}\n")
    print(f"  wrote {path}: {len(el)} atoms", flush=True)


def square_graphene(side=24.0, layers=2):
    """Honeycomb clipped to a square of side `side` Å. Two sheets, no edge H."""
    n = int(side / 1.3) + 6
    pos, el = make_graphene_xyz(n, n, layers=layers, defect=None, passivate=False)
    m = np.maximum(np.abs(pos[:, 0]), np.abs(pos[:, 1])) <= side / 2
    return pos[m], [el[i] for i in range(len(el)) if m[i]]


def make_nacl(n=12, layers=2, a=5.64):
    """NaCl(001): square checkerboard, spacing a/2, layers stacked by a/2."""
    d = a / 2
    pos, el = [], []
    for L in range(layers):
        for i in range(n):
            for j in range(n):
                pos.append((i * d, j * d, L * d))
                el.append('Na' if (i + j + L) % 2 == 0 else 'Cl')
    pos = np.array(pos, np.float64)
    pos[:, 0] -= pos[:, 0].mean(); pos[:, 1] -= pos[:, 1].mean()
    return pos, el


def plot_loaded_cells(afm, mol, pitch=0.1, nxy=192, tx=16, ty=16, delta_b=1.2, rho=1.0):
    """PIC cells that hold atoms this workgroup actually preloads, plus the mesh nodes copied with them."""
    from spammm.SPM.AFM import build_scan_xy_points_vectorized
    p = afm.fit_contact_pme(h_mesh=1.0, delta_b=delta_b, bPrint=False)
    apos = np.asarray(p.atom_pos, np.float64)
    ztop, zbot = float(apos[:, 2].max()), float(apos[:, 2].min())
    top = apos[:, 2] > ztop - 1.0
    rb = float(p.bucket_cell_size)
    L = abs(float(afm.dpos0[3]))
    pp_lo = max(ztop + 2.6, zbot + rb + 0.5)
    pp_hi = pp_lo + 2.0
    nz = int(round((pp_hi - pp_lo) / pitch)) + 1
    dtip = -pitch
    span = (nxy - 1) * pitch
    cxy = apos[top, :2].mean(axis=0)
    p0 = np.array([cxy[0] - span / 2, cxy[1] - span / 2, pp_hi + L], np.float32)
    da = np.array([pitch, 0., 0.], np.float32)
    db = np.array([0., pitch, 0.], np.float32)
    pts = build_scan_xy_points_vectorized(p0, da, db, nxy, nxy)
    tiles = afm._pme_build_wg_tiles(p, pts, nxy, nxy, nz, dtip, tx=tx, ty=ty, rho_max=rho, nzL=6)
    nloc = np.diff(tiles['wg_atom_offsets'])
    full = [k for k, (xl, xh, yl, yh) in enumerate(tiles['tile_pix'])
            if abs((xh - xl) - (tx - 1) * pitch) < 1e-3 and abs((yh - yl) - (ty - 1) * pitch) < 1e-3]
    t = full[int(np.argmax(nloc[full]))]
    xl, xh, yl, yh = tiles['tile_pix'][t]
    sel = tiles['wg_atom_ids'][int(tiles['wg_atom_offsets'][t]):int(tiles['wg_atom_offsets'][t + 1])]
    sel = sel[top[sel]]
    x0, y0 = float(p.bucket_bounds[0]), float(p.bucket_bounds[1])
    nbx, nby = int(p.bucket_nbx), int(p.bucket_nby)
    cell = rb
    bx = np.clip(((apos[:, 0] - x0) / cell).astype(int), 0, nbx - 1)
    by = np.clip(((apos[:, 1] - y0) / cell).astype(int), 0, nby - 1)
    loaded = np.zeros((nby, nbx), np.int32)
    for iy, ix in zip(by[sel], bx[sel]):
        loaded[iy, ix] += 1
    occ = np.zeros((nby, nbx), np.int32)
    for iy, ix in zip(by[top], bx[top]):
        occ[iy, ix] += 1
    touched = np.argwhere(loaded > 0)
    print(f"tile {t}  {tx}×{ty}  preloaded top atoms {len(sel)}  PIC cells touched {len(touched)}", flush=True)
    for iy, ix in touched:
        print(f"  cell ({ix},{iy})  loaded {int(loaded[iy, ix])} of {int(occ[iy, ix])} top atoms in that cell", flush=True)
    fig, ax = plt.subplots(1, 2, figsize=(13.2, 6.4))
    # left: PIC cells. Filled = this workgroup copied at least one atom from the cell.
    ax[0].set_aspect('equal')
    for iy in range(nby):
        for ix in range(nbx):
            if occ[iy, ix] == 0 and loaded[iy, ix] == 0:
                continue
            cx, cy = x0 + ix * cell, y0 + iy * cell
            if cx + cell < xl - 2 * cell or cx > xh + 2 * cell or cy + cell < yl - 2 * cell or cy > yh + 2 * cell:
                continue
            face = '#f6c28b' if loaded[iy, ix] else 'none'
            ax[0].add_patch(plt.Rectangle((cx, cy), cell, cell, fill=loaded[iy, ix] > 0,
                                          fc=face, ec='0.45', lw=0.8))
            if loaded[iy, ix]:
                ax[0].text(cx + cell / 2, cy + cell / 2, str(int(loaded[iy, ix])),
                           ha='center', va='center', fontsize=11, color='saddlebrown')
    ax[0].scatter(apos[top, 0], apos[top, 1], s=10, c='0.75', zorder=3)
    ax[0].scatter(apos[sel, 0], apos[sel, 1], s=22, c='darkorange', zorder=4)
    ax[0].add_patch(plt.Rectangle((xl - 0.5 * pitch, yl - 0.5 * pitch), tx * pitch, ty * pitch,
                                  fill=False, ec='crimson', lw=2, zorder=5))
    rad = 1.6 * cell
    ax[0].set_xlim(0.5 * (xl + xh) - rad, 0.5 * (xl + xh) + rad)
    ax[0].set_ylim(0.5 * (yl + yh) - rad, 0.5 * (yl + yh) + rad)
    ax[0].set_xlabel('x [Å]'); ax[0].set_ylabel('y [Å]')
    ax[0].set_title(f'PIC cells, side {cell:.2f} Å\n'
                    f'number = top-sheet atoms this tile copies from that cell\n'
                    f'{len(touched)} cells, {len(sel)} atoms')
    # right: mesh nodes the same tile copies into local memory (h = 1 Å)
    i0x, i0y, nxL, nyL = tiles['tile_desc'][t]
    ox, oy = float(p.mesh_origin[0]), float(p.mesh_origin[1])
    h = float(p.mesh_h)
    xs = ox + (i0x + np.arange(nxL)) * h
    ys = oy + (i0y + np.arange(nyL)) * h
    XX, YY = np.meshgrid(xs, ys)
    ax[1].set_aspect('equal')
    ax[1].scatter(XX, YY, s=18, c='teal', zorder=3)
    ax[1].add_patch(plt.Rectangle((xs[0] - 0.5 * h, ys[0] - 0.5 * h), nxL * h, nyL * h,
                                  fill=False, ec='teal', lw=1.2))
    ax[1].add_patch(plt.Rectangle((xl - 0.5 * pitch, yl - 0.5 * pitch), tx * pitch, ty * pitch,
                                  fill=False, ec='crimson', lw=2, zorder=5))
    ax[1].scatter(apos[sel, 0], apos[sel, 1], s=22, c='darkorange', zorder=4)
    ax[1].set_xlim(xs[0] - 2 * h, xs[-1] + 2 * h)
    ax[1].set_ylim(ys[0] - 2 * h, ys[-1] + 2 * h)
    ax[1].set_xlabel('x [Å]'); ax[1].set_ylabel('y [Å]')
    ax[1].set_title(f'mesh nodes copied with the atoms\n'
                    f'{nxL}×{nyL} at h={h:.0f} Å, ×{tiles["nzL"]} in z\n'
                    f'red = {tx}×{ty} pixel tile, {tx*pitch:.1f}×{ty*pitch:.1f} Å')
    fig.tight_layout()
    stem = os.path.basename(mol)[:-4]
    fn = os.path.join(OUT, f'loaded_cells_{stem}_{tx}x{ty}.png')
    fig.savefig(fn, dpi=140)
    plt.close(fig)
    print(f'REVIEW: {fn}', flush=True)
    dev = afm.ctx.devices[0]
    import pyopencl as cl
    mult = afm.prg.relaxStrokesTiltedContactPMETileSph.get_work_group_info(
        cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE, dev)
    kmax = afm.prg.relaxStrokesTiltedContactPMETileSph.get_work_group_info(
        cl.kernel_work_group_info.WORK_GROUP_SIZE, dev)
    print(f'kernel preferred multiple {mult}   kernel max workgroup {kmax}   '
          f'device max {dev.max_work_group_size}', flush=True)


def square_sheet(side, layers=2, z_gap=3.4):
    """Honeycomb clipped to a square, both layers. No neighbor pass."""
    a = np.sqrt(3) * A_CC
    a1 = np.array([a, 0.0]); a2 = np.array([a / 2, a * np.sqrt(3) / 2])
    n = int(side / a) + 3
    ii, jj = np.meshgrid(np.arange(-n, n + 1), np.arange(-n, n + 1), indexing='ij')
    b0 = ii[..., None] * a1 + jj[..., None] * a2
    b1 = b0 + np.array([a / 2, a * np.sqrt(3) / 6])
    xy = np.concatenate([b0.reshape(-1, 2), b1.reshape(-1, 2)], axis=0)
    xy = xy[np.maximum(np.abs(xy[:, 0]), np.abs(xy[:, 1])) <= side / 2]
    pos = np.vstack([np.column_stack([xy, np.full(len(xy), L * z_gap)]) for L in range(layers)])
    return pos, ['C'] * len(pos)


def _basis_line(n_coarse, s):
    """Collocation of the cardinal cubic basis. Row k is the fine node at coarse coordinate k/s."""
    from spammm.surfaces.CoarseMesh import _basis
    n_fine = s * (n_coarse - 1) + 1
    B = np.zeros((n_fine, n_coarse), np.float64)
    for k in range(n_fine):
        t = k / s
        i0 = int(np.floor(t))
        w = _basis(t - i0)
        for a in range(4):
            j = i0 - 1 + a
            if 0 <= j < n_coarse:
                B[k, j] += w[a]
    return B


def _ls_spline(samples, s):
    """Least-squares cubic spline on the coarse lattice, from samples at spacing h/s."""
    nc = [(n - 1) // s + 1 for n in samples.shape]
    assert tuple(s * (c - 1) + 1 for c in nc) == samples.shape, (samples.shape, nc)
    Bx, By, Bz = (_basis_line(c, s) for c in nc)
    rhs = np.tensordot(Bx.T, samples, axes=(1, 0))          # (ncx, nyf, nzf)
    rhs = np.tensordot(rhs, By, axes=(1, 0))                # (ncx, nzf, ncy)
    rhs = np.tensordot(rhs, Bz, axes=(1, 0))                # (ncx, ncy, ncz)
    for G, axis in ((Bz.T @ Bz, 2), (By.T @ By, 1), (Bx.T @ Bx, 0)):
        n = G.shape[0]
        flat = np.moveaxis(rhs, axis, -1).reshape(-1, n)
        sol = np.linalg.solve(G, flat.T).T
        rhs = np.moveaxis(sol.reshape(np.moveaxis(rhs, axis, -1).shape), -1, axis)
    return rhs


def plot_supersample_heights(side=24.0):
    """Relaxed PP-AFM df strip with the SPM_CLI scan contract.

    K_LAT=0.5 N/m, K_RAD=20 eV/Å², L=3 Å, FIRE, df at 3.7–4.7 Å, amp=1 Å.
    Gray, per-panel range, repulsive (positive df) bright. Tip charges are off
    on every row: the contact-PME fit rejects the CLI two-site tip.
    """
    from dataclasses import replace
    from spammm.SPM.AFM import AFMulator, stiffness_Nm_to_eVA2
    from spammm.SPM.AFM_utils import (ScanSpec, afm_df_height_stacks, make_fdbm_grid_com_zsym,
                                      plot_afm_variant_height_strip, scan_extent, shared_postprocess)
    # SPM_CLI defaults (run_spm.py afm)
    step, margin, scan_margin = 0.1, 4.0, 2.0
    h_min, h_max, h_step, amp = 3.7, 4.7, 0.1, 1.0
    K_LAT_Nm, K_RAD, bond = 0.5, 20.0, 3.0
    K_LAT = stiffness_Nm_to_eVA2(K_LAT_Nm)
    pos, el = square_sheet(side)
    mol = os.path.join(OUT, f"graphene_square{side:.0f}_2L.xyz")
    write_xyz(mol, pos, el)
    ztop = float(pos[:, 2].max())
    h_df, h_Fz, h_scan = afm_df_height_stacks(h_min, h_max, h_step, amp=amp, amp_align=True)
    scan_xs = np.arange(float(pos[:, 0].min() - scan_margin), float(pos[:, 0].max() + scan_margin), step, dtype=np.float32)
    scan_ys = np.arange(float(pos[:, 1].min() - scan_margin), float(pos[:, 1].max() + scan_margin), step, dtype=np.float32)
    spec = ScanSpec(scan_xs=scan_xs, scan_ys=scan_ys, h_df=h_df, h_Fz=h_Fz, h_scan=h_scan,
                    amplitude=amp, osc_dir=(0., 0., 1.), K_LAT=K_LAT, K_RAD=K_RAD,
                    bond_length=bond, scan_margin=scan_margin)
    print(f"PP-AFM  K_LAT={K_LAT_Nm} N/m ({K_LAT:.4f} eV/Å²)  K_RAD={K_RAD} eV/Å²  L={bond} Å  FIRE  "
          f"df {h_min}–{h_max} dz={h_step} amp={amp}  scan={len(scan_xs)}×{len(scan_ys)}", flush=True)
    # GridFF: same relaxStrokes path as run_morse_pp_afm. z vacuum must cover the bilayer top.
    z_vac = max(6.0, (ztop - float(pos[:, 2].mean())) + float(h_scan[-1]) + 1.0)
    _, origin, ngrid, step = make_fdbm_grid_com_zsym(pos, step, margin, z_vac=z_vac)
    afm = AFMulator(use_morse=True, nloc=32, use_fire=True)
    afm.load_molecule(mol)
    afm.assign_params(params_path='data/ElementTypes.dat')
    afm.tipQs[:] = 0.0
    L = np.array([ngrid[0] * step, ngrid[1] * step, ngrid[2] * step], np.float32)
    afm.setup_grid_world(origin, L, ngrid)
    print(f"  GridFF relax {ngrid[0]}×{ngrid[1]}×{ngrid[2]} ...", flush=True)
    afm.make_forcefield()
    afm.setup_fdbm_grid_from_img(afm.img_FF, ngrid, origin, step)
    FEg, disp_g = afm.scan_fdbm(scan_xs, scan_ys, h_scan, mol_z=ztop, ppm_mode=True, use_fire=True,
                                K_LAT=K_LAT, K_RAD=K_RAD, bond_length=bond)
    g = shared_postprocess(FEg, spec, backend_name='morse')
    print(f"  GridFF |dxy| max={np.hypot(disp_g['dx'], disp_g['dy']).max():.3f} Å", flush=True)
    # PME: same springs and the same scan grid. Supersample only the V_L spline.
    afm.dpos0 = np.array([0., 0., -bond, bond], np.float32)
    afm.stiffness = np.array([-K_LAT, -K_LAT, -K_LAT, -K_RAD], np.float32)
    z_pad = 1.5
    qbounds = np.array([[float(scan_xs.min()), float(scan_xs.max())],
                        [float(scan_ys.min()), float(scan_ys.max())],
                        [ztop + float(h_scan.min()) - z_pad, ztop + float(h_scan.max()) + z_pad]])
    afm.fit_contact_pme(query_bounds=qbounds, h_mesh=1.0, halo_nodes=6, bPrint=False)
    p = afm.cpm
    O = np.asarray(p.mesh_origin, np.float64)
    hi = O + (np.array(p.mesh_shape) - 1) * p.mesh_h
    qbox = np.stack([O, hi], axis=1)
    nx_s, ny_s, nz_s = len(scan_xs), len(scan_ys), len(h_scan)
    dz = float(h_scan[1] - h_scan[0])
    scan_p0 = np.array([float(scan_xs[0]), float(scan_ys[0]), ztop + float(h_scan.max()) + bond], np.float32)
    dfs = {'gridff': g.df}
    for s in (1, 2, 3):
        print(f"  PME relax {s}×{s}×{s} ...", flush=True)
        fine = afm._pme_build_coarse_mesh_gpu(np.asarray(p.atom_pos), p.split_params, qbox,
                                              h_mesh=p.mesh_h / s, halo_nodes=0, prefilter=False)
        ps = replace(p, mesh_coeffs=np.ascontiguousarray(_ls_spline(fine.coeffs, s)))
        afm.cpm = ps
        FEs, _ = afm.run_scan_contact_pme(nxy=(nx_s, ny_s), nz=nz_s, dtip=-abs(dz),
                                          scan_p0=scan_p0, scan_da=np.array([step, 0, 0], np.float32),
                                          scan_db=np.array([0, step, 0], np.float32), core_backend='local')
        r = shared_postprocess(FEs[:, :, ::-1, :], spec, backend_name='contact_pme')
        dfs[f's{s}'] = r.df
    variants = {k: {'df': v} for k, v in dfs.items()}
    rows = [('df', 'gridff', 'GridFF\nK_LAT=0.5 N/m', 'gray'),
            ('df', 's1', '1×1×1', 'gray'), ('df', 's2', '2×2×2', 'gray'), ('df', 's3', '3×3×3', 'gray')]
    plot_afm_variant_height_strip(
        variants, rows, h_df, os.path.join(OUT, 'heights_df_strip.png'),
        scale='per_image', title=f'df  PP-AFM  K_LAT={K_LAT_Nm} N/m  K_RAD={K_RAD}  L={bond}Å',
        extent=scan_extent(scan_xs, scan_ys), amp=amp, amp_align=True,
        long_axis_vertical=True, tight=True)


def bench_gridff(sides=(16, 24, 32, 48), dx=0.1, pitch=0.1):
    """GridFF voxel build versus the PME nodal raster, plus 2³ and 3³ spline refits."""
    from spammm.SPM.AFM import AFMulator
    from spammm.surfaces.CoarseMesh import CoarseMesh, eval_mesh, eval_mesh_direct
    afm = AFMulator(use_morse=True, use_fire=False)
    rows = []
    print(f"{'side':>5} {'na':>6} | {'PME nodes':>10} {'core pts':>8} {'mesh ms':>8} | "
          f"{'FF voxels':>10} {'FF ms':>8} {'FF scan':>8} {'PME scan':>8}", flush=True)
    for side in sides:
        pos, el = square_sheet(side)
        mol = os.path.join(OUT, f"graphene_square{side:.0f}_2L.xyz")
        write_xyz(mol, pos, el)
        afm.load_molecule(mol)
        afm.assign_params(params_path='data/ElementTypes.dat')
        afm.tipQs[:] = 0.0
        afm.fit_contact_pme(bPrint=False)
        afm.fit_contact_pme(bPrint=False)
        tm = afm.last_pme_times
        p = afm.cpm
        nodes = int(np.prod(tm['mesh_shape']))
        # same sheet scan as bench_scale
        apos = np.asarray(p.atom_pos, np.float64)
        ztop, zbot = float(apos[:, 2].max()), float(apos[:, 2].min())
        top = apos[:, 2] > ztop - 1.0
        rb = float(p.bucket_cell_size)
        Llen = abs(float(afm.dpos0[3]))
        pp_hi = max(ztop + 2.6, zbot + rb + 0.5) + 2.0
        nz = int(round(2.0 / pitch)) + 1
        nxy = int(np.ceil((side / pitch + 1) / 16.0) * 16)
        span = (nxy - 1) * pitch
        cxy = apos[top, :2].mean(axis=0)
        p0 = np.array([cxy[0] - span / 2, cxy[1] - span / 2, pp_hi + Llen], np.float32)
        da = np.array([pitch, 0., 0.], np.float32)
        db = np.array([0., pitch, 0.], np.float32)
        kw = dict(nxy=(nxy, nxy), nz=nz, dtip=-pitch, scan_p0=p0, scan_da=da, scan_db=db,
                  relax_mode='sph', core_backend='tile', tile_xy=(16, 16), rho_max=1.0, nzL=6)
        afm.run_scan_contact_pme(**kw)
        afm.queue.finish()
        t0 = time.perf_counter()
        afm.run_scan_contact_pme(**kw, bAlloc=False)
        afm.queue.finish()
        pme_scan = (time.perf_counter() - t0) * 1e3
        # GridFF over the whole molecule, 0.1 Å voxels — a fresh load so atom coords stay put
        afm.load_molecule(mol)
        afm.assign_params(params_path='data/ElementTypes.dat')
        afm.tipQs[:] = 0.0
        a = afm.atoms_arr[:, :3]
        mn, mx = a.min(0), a.max(0)
        margin, z_top = 3.0, 8.0
        Lg = np.array([(mx[0] - mn[0]) + 2 * margin, (mx[1] - mn[1]) + 2 * margin,
                       (mx[2] - mn[2]) + margin + z_top], np.float64)
        ng = tuple(max(2, int(np.round(Lg[i] / dx))) for i in range(3))
        afm.setup_grid(n=ng, L=Lg.astype(np.float32), margin=margin, z_top=z_top, shift_atoms=False)
        print(f"  GridFF {side:.0f} Å  voxels {ng[0]}×{ng[1]}×{ng[2]} ...", flush=True)
        afm.make_forcefield()
        afm.queue.finish()
        t0 = time.perf_counter()
        afm.make_forcefield(bAlloc=False)
        afm.queue.finish()
        ff_ms = (time.perf_counter() - t0) * 1e3
        afm.run_scan(nxy=(nxy, nxy), nz=nz, dtip=-pitch, scan_p0=p0, scan_da=da, scan_db=db)
        afm.queue.finish()
        t0 = time.perf_counter()
        afm.run_scan(nxy=(nxy, nxy), nz=nz, dtip=-pitch, scan_p0=p0, scan_da=da, scan_db=db, bAlloc=False)
        afm.queue.finish()
        ff_scan = (time.perf_counter() - t0) * 1e3
        nvox = int(np.prod(ng))
        row = dict(side=side, na=int(tm['na']), nodes=nodes, core_pts=int(tm['na']) * 32,
                   mesh_ms=tm['mesh_ms'], pme_scan=pme_scan, voxels=nvox, ff_ms=ff_ms, ff_scan=ff_scan,
                   mesh_shape=tm['mesh_shape'])
        rows.append(row)
        print(f"{side:5.0f} {row['na']:6d} | {nodes:10d} {row['core_pts']:8d} {row['mesh_ms']:8.1f} | "
              f"{nvox:10d} {ff_ms:8.1f} {ff_scan:8.1f} {pme_scan:8.1f}", flush=True)
    # Supersample the 32 Å sheet: evaluate V_L at h, h/2, h/3, fit the h=1 coefficients
    side = 32.0
    afm.load_molecule(os.path.join(OUT, "graphene_square32_2L.xyz"))
    afm.assign_params(params_path='data/ElementTypes.dat')
    afm.tipQs[:] = 0.0
    afm.fit_contact_pme(bPrint=False)
    p = afm.cpm
    apos = np.asarray(p.atom_pos, np.float64)
    zmax = float(apos[:, 2].max())
    qb = np.array([[apos[:, 0].min() - 2, apos[:, 0].max() + 2],
                   [apos[:, 1].min() - 2, apos[:, 1].max() + 2],
                   [zmax + 3.0, zmax + 8.0]])
    coarse = afm._pme_build_coarse_mesh_gpu(apos, p.split_params, qb, h_mesh=1.0, halo_nodes=6)
    O = np.asarray(coarse.origin, np.float64)
    nc = coarse.coeffs.shape
    hi = O + (np.array(nc) - 1) * 1.0
    sup_rows = []
    rng = np.random.default_rng(1)
    inset = O + 3.0
    out = hi - 3.0
    qry = np.column_stack([rng.uniform(inset[i], out[i], 80) for i in range(3)])
    E_ref, F_ref = eval_mesh_direct(qry, apos, p.split_params)
    print("supersample of V_L onto the 1 Å coefficients, 32 Å sheet", flush=True)
    for s in (1, 2, 3):
        h = 1.0 / s
        qf = np.stack([O, hi], axis=1)
        print(f"  raster s={s}  h={h:.3f} Å ...", flush=True)
        fine = afm._pme_build_coarse_mesh_gpu(apos, p.split_params, qf, h_mesh=h, halo_nodes=0, prefilter=False)
        samples = fine.coeffs
        assert samples.shape == tuple(s * (c - 1) + 1 for c in nc), (samples.shape, nc)
        t0 = time.perf_counter()
        coeffs = _ls_spline(samples, s)
        ls_ms = (time.perf_counter() - t0) * 1e3
        if s == 1:
            print(f"  s=1 LS vs nodal prefilter  max|Δc|={np.max(np.abs(coeffs - coarse.coeffs)):.3e}", flush=True)
        mesh = CoarseMesh(coeffs=coeffs, origin=O, h=1.0, halo=6,
                          query_interior=coarse.query_interior)
        E, F = eval_mesh(mesh, qry)
        dE = float(np.max(np.abs(E - E_ref)))
        dF = float(np.max(np.abs(F - F_ref)))
        fill = afm.last_raster_ms
        sup_rows.append(dict(s=s, n=int(samples.size), fill_ms=fill['fill_ms'], ls_ms=ls_ms, dE=dE, dF=dF))
        print(f"  s={s}  samples {samples.size:8d}  eval {fill['fill_ms']:7.1f} ms  "
              f"LS {ls_ms:7.1f} ms  max|dE| {dE:.3e}  max|dF| {dF:.3e}", flush=True)
    fig, ax = plt.subplots(1, 2, figsize=(12.4, 5.2))
    na = [r['na'] for r in rows]
    ax[0].plot(na, [r['ff_ms'] for r in rows], 'o-', label='GridFF build 0.1 Å')
    ax[0].plot(na, [r['ff_scan'] for r in rows], 's-', label='GridFF scan')
    ax[0].plot(na, [r['mesh_ms'] for r in rows], '^-', label='PME mesh (1 Å nodes)')
    ax[0].plot(na, [r['pme_scan'] for r in rows], 'D-', label='PME scan')
    ax[0].set_xlabel('atoms')
    ax[0].set_ylabel('time [ms]')
    ax[0].set_yscale('log')
    ax[0].legend()
    ax[0].set_title('GridFF voxels versus PME nodal samples')
    xs = [r['s'] for r in sup_rows]
    ax[1].plot(xs, [r['dF'] for r in sup_rows], 'o-', color='C3')
    for r in sup_rows:
        ax[1].annotate(f"{r['n']} pts\n{r['fill_ms']:.0f} ms", (r['s'], r['dF']),
                       textcoords='offset points', xytext=(6, 6), fontsize=8)
    ax[1].set_xticks([1, 2, 3])
    ax[1].set_xticklabels(['1×1×1', '2×2×2', '3×3×3'])
    ax[1].set_ylabel('max |ΔF| of V_L [eV/Å]')
    ax[1].set_title('coarse spline fit to denser V_L samples\n32 Å sheet, 80 off-node points')
    fig.tight_layout()
    fn = os.path.join(OUT, 'gridff_vs_pme.png')
    fig.savefig(fn, dpi=140)
    plt.close(fig)
    print(f'REVIEW: {fn}', flush=True)
    return rows, sup_rows


def bench_scale(sides=(16, 24, 32, 48, 64, 80), pitch=0.1):
    """Fit (B-spline mesh + core) and a 0.1 Å scan of the whole sheet, versus size."""
    from spammm.SPM.AFM import AFMulator, build_scan_xy_points_vectorized
    afm = AFMulator(use_morse=True, use_fire=False)
    rows = []
    print(f"{'side':>5} {'na':>6} {'scan':>9} | {'mesh':>8} {'core':>8} {'fit':>8} {'scan':>8} {'total':>8}", flush=True)
    for side in sides:
        print(f"  building {side:.0f} Å sheet ...", flush=True)
        pos, el = square_sheet(side)
        mol = os.path.join(OUT, f"graphene_square{side:.0f}_2L.xyz")
        write_xyz(mol, pos, el)
        afm.load_molecule(mol)
        afm.assign_params(params_path='data/ElementTypes.dat')
        afm.tipQs[:] = 0.0
        afm.fit_contact_pme(bPrint=False)                 # allocate
        afm.fit_contact_pme(bPrint=False)
        tm = afm.last_pme_times
        p = afm.cpm
        apos = np.asarray(p.atom_pos, np.float64)
        ztop, zbot = float(apos[:, 2].max()), float(apos[:, 2].min())
        top = apos[:, 2] > ztop - 1.0
        rb = float(p.bucket_cell_size)
        L = abs(float(afm.dpos0[3]))
        pp_lo = max(ztop + 2.6, zbot + rb + 0.5)
        pp_hi = pp_lo + 2.0
        nz = int(round((pp_hi - pp_lo) / pitch)) + 1
        nxy = int(np.ceil((side / pitch + 1) / 16.0) * 16)
        span = (nxy - 1) * pitch
        cxy = apos[top, :2].mean(axis=0)
        p0 = np.array([cxy[0] - span / 2, cxy[1] - span / 2, pp_hi + L], np.float32)
        da = np.array([pitch, 0., 0.], np.float32)
        db = np.array([0., pitch, 0.], np.float32)
        print(f"  scan {side:.0f} Å  na={len(pos)}  {nxy}×{nxy}×{nz} ...", flush=True)
        kw = dict(nxy=(nxy, nxy), nz=nz, dtip=-pitch, scan_p0=p0, scan_da=da, scan_db=db,
                  relax_mode='sph', core_backend='tile', tile_xy=(16, 16), rho_max=1.0, nzL=6)
        afm.queue.finish()
        afm.run_scan_contact_pme(**kw)
        afm.queue.finish()
        t0 = time.perf_counter()
        afm.run_scan_contact_pme(**kw, bAlloc=False)
        afm.queue.finish()
        scan_ms = (time.perf_counter() - t0) * 1e3
        total = tm['fit_ms'] + scan_ms
        row = dict(side=side, na=int(tm['na']), nxy=nxy, nz=nz, mesh=tm['mesh_shape'],
                   mesh_ms=tm['mesh_ms'], core_ms=tm['core_ms'], fit_ms=tm['fit_ms'],
                   scan_ms=scan_ms, total_ms=total)
        rows.append(row)
        print(f"{side:5.0f} {row['na']:6d} {nxy:4d}×{nz:<3d} | {row['mesh_ms']:8.1f} {row['core_ms']:8.2f} "
              f"{row['fit_ms']:8.1f} {scan_ms:8.1f} {total:8.1f}", flush=True)
    na = np.array([r['na'] for r in rows])
    fig, ax = plt.subplots(1, 2, figsize=(12.4, 5.4))
    ax[0].plot(na, [r['mesh_ms'] for r in rows], 'o-', label='B-spline mesh')
    ax[0].plot(na, [r['core_ms'] for r in rows], 's-', label='atom core')
    ax[0].plot(na, [r['scan_ms'] for r in rows], '^-', label='scan')
    ax[0].plot(na, [r['total_ms'] for r in rows], 'kD-', label='fit + scan')
    ax[0].set_xlabel('atoms (bilayer)')
    ax[0].set_ylabel('time [ms]')
    ax[0].set_title('square graphene, 0.1 Å scan of the sheet\n16×16 tile, second run')
    ax[0].legend()
    labels = [f"{r['side']:.0f} Å\n{r['na']}" for r in rows]
    x = np.arange(len(rows))
    ax[1].bar(x, [r['mesh_ms'] for r in rows], label='B-spline mesh')
    ax[1].bar(x, [r['core_ms'] for r in rows], bottom=[r['mesh_ms'] for r in rows], label='atom core')
    bot = [r['fit_ms'] for r in rows]
    ax[1].bar(x, [r['scan_ms'] for r in rows], bottom=bot, label='scan')
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(labels)
    ax[1].set_ylabel('time [ms]')
    ax[1].set_title('same times, stacked')
    ax[1].legend()
    fig.tight_layout()
    fn = os.path.join(OUT, 'scale_graphene.png')
    fig.savefig(fn, dpi=140)
    plt.close(fig)
    print(f'REVIEW: {fn}', flush=True)
    return rows


def bench_wg(afm, mol, pitch=0.1, nxy=192, delta_b=1.2, rho=1.0,
             shapes=((8, 8), (12, 12), (16, 16), (24, 24), (32, 32))):
    """0.1 Å pixels. Square workgroups. Atoms loaded per tile, PIC atoms per cell, scan time."""
    from spammm.SPM.AFM import build_scan_xy_points_vectorized
    p = afm.fit_contact_pme(h_mesh=1.0, delta_b=delta_b, bPrint=False)
    apos = np.asarray(p.atom_pos, np.float64)
    ztop, zbot = float(apos[:, 2].max()), float(apos[:, 2].min())
    top = apos[:, 2] > ztop - 1.0
    rb = float(p.bucket_cell_size)
    # PP window stays far enough above the lower sheet that its cores are dead
    L = abs(float(afm.dpos0[3]))
    pp_lo = max(ztop + 2.6, zbot + rb + 0.5)
    pp_hi = pp_lo + 2.0
    nz = int(round((pp_hi - pp_lo) / pitch)) + 1
    dtip = -pitch
    z_tip0 = pp_hi + L
    span = (nxy - 1) * pitch
    cxy = apos[top, :2].mean(axis=0)
    p0 = np.array([cxy[0] - span / 2, cxy[1] - span / 2, z_tip0], np.float32)
    da = np.array([pitch, 0., 0.], np.float32)
    db = np.array([0., pitch, 0.], np.float32)
    pts = build_scan_xy_points_vectorized(p0, da, db, nxy, nxy)
    # PIC cell occupancy, top sheet only (the contact core)
    x0, y0 = float(p.bucket_bounds[0]), float(p.bucket_bounds[1])
    nbx, nby = int(p.bucket_nbx), int(p.bucket_nby)
    cell = rb
    bx = np.clip(((apos[:, 0] - x0) / cell).astype(int), 0, nbx - 1)
    by = np.clip(((apos[:, 1] - y0) / cell).astype(int), 0, nby - 1)
    occ = np.zeros((nby, nbx), np.int32)
    for iy, ix in zip(by[top], bx[top]):
        occ[iy, ix] += 1
    interior = occ[1:-1, 1:-1]
    interior = interior[interior > 0]
    pic_mean = float(interior.mean()) if len(interior) else 0.0
    pic_max = int(occ.max())
    bb = apos[top, :2].max(axis=0) - apos[top, :2].min(axis=0)
    print(f"  top-sheet bbox {bb[0]:.1f} × {bb[1]:.1f} Å", flush=True)
    print(f"{os.path.basename(mol)}  na={p.na} top={int(top.sum())}  pitch={pitch} Å  "
          f"scan {nxy}×{nxy} = {span:.1f} Å  nz={nz}  PP h above top "
          f"[{pp_lo-ztop:.2f},{pp_hi-ztop:.2f}] Å", flush=True)
    print(f"  PIC cell = r_b = {cell:.2f} Å   top-sheet atoms/cell  mean={pic_mean:.1f}  max={pic_max}", flush=True)
    hdr = (f"{'WG':>7} {'Å':>11} {'thr':>5} | {'atoms/tile':>10} {'top':>5} {'bot':>4} | "
           f"{'mesh':>8} {'local':>7} | {'ms':>8} {'µs/px':>8}")
    print(hdr, flush=True)
    rows = []
    for tx, ty in shapes:
        tiles = afm._pme_build_wg_tiles(p, pts, nxy, nxy, nz, dtip, tx=tx, ty=ty, rho_max=rho, nzL=6)
        nloc = np.diff(tiles['wg_atom_offsets'])
        t = int(np.argmax(nloc))
        # prefer a full interior tile
        full = []
        for k, (xl, xh, yl, yh) in enumerate(tiles['tile_pix']):
            if abs((xh - xl) - (tx - 1) * pitch) < 1e-3 and abs((yh - yl) - (ty - 1) * pitch) < 1e-3:
                full.append(k)
        if full:
            t = full[int(np.argmax(nloc[full]))]
        offs, ids = tiles['wg_atom_offsets'], tiles['wg_atom_ids']
        sel = ids[int(offs[t]):int(offs[t + 1])]
        n_top = int(top[sel].sum()) if len(sel) else 0
        n_bot = int(len(sel) - n_top)
        mem = afm._pme_tile_local_bytes(tiles)
        xl, xh, yl, yh = tiles['tile_pix'][t]
        Wx, Hy = (xh - xl) + pitch, (yh - yl) + pitch
        print(f"  scan {tx}×{ty} ...", flush=True)
        try:
            afm.queue.finish()
            t0 = time.perf_counter()
            afm.run_scan_contact_pme(nxy=(nxy, nxy), nz=nz, dtip=dtip, scan_p0=p0, scan_da=da, scan_db=db,
                                     relax_mode='sph', core_backend='tile', tile_xy=(tx, ty), rho_max=rho, nzL=6)
            afm.queue.finish()
            t1 = time.perf_counter()
            afm.run_scan_contact_pme(nxy=(nxy, nxy), nz=nz, dtip=dtip, scan_p0=p0, scan_da=da, scan_db=db,
                                     relax_mode='sph', core_backend='tile', tile_xy=(tx, ty), rho_max=rho, nzL=6, bAlloc=False)
            afm.queue.finish()
            ms = (time.perf_counter() - t1) * 1e3
            warm = 1e3 * (t1 - t0)
        except Exception as e:
            print(f"{tx:2d}×{ty:<2d} {Wx:5.2f}×{Hy:<5.2f} {tx*ty:5d} | {n_top+n_bot:10d} {n_top:5d} {n_bot:4d} | "
                  f"{tiles['nxL_max']}×{tiles['nyL_max']}×6 {mem['total']/1024:6.2f}K | LAUNCH FAILED {type(e).__name__}", flush=True)
            continue
        us = ms * 1e3 / (nxy * nxy)
        row = dict(tx=tx, ty=ty, Wx=Wx, Hy=Hy, thr=tx * ty, n_top=n_top, n_bot=n_bot,
                   nloc=int(nloc[t]), mesh=f"{tiles['nxL_max']}×{tiles['nyL_max']}×6",
                   kb=mem['total'] / 1024, ms=ms, us=us, pic_mean=pic_mean, pic_max=pic_max, cell=cell)
        rows.append(row)
        print(f"{tx:2d}×{ty:<2d} {Wx:5.2f}×{Hy:<5.2f} {tx*ty:5d} | {n_top+n_bot:10d} {n_top:5d} {n_bot:4d} | "
              f"{row['mesh']:>8} {row['kb']:6.2f}K | {ms:8.1f} {us:8.2f}   (warmup {1e3*(t1-t0):.0f} ms)", flush=True)
    # figure: 16×16 tile on the top sheet + timing bars
    r16 = next(r for r in rows if (r['tx'], r['ty']) == (16, 16))
    tiles = afm._pme_build_wg_tiles(p, pts, nxy, nxy, nz, dtip, tx=16, ty=16, rho_max=rho, nzL=6)
    nloc = np.diff(tiles['wg_atom_offsets'])
    full = [k for k, (xl, xh, yl, yh) in enumerate(tiles['tile_pix'])
            if abs((xh - xl) - 15 * pitch) < 1e-3 and abs((yh - yl) - 15 * pitch) < 1e-3]
    t = full[int(np.argmax(nloc[full]))]
    xl, xh, yl, yh = tiles['tile_pix'][t]
    box = (xl - 0.5 * pitch, yl - 0.5 * pitch, 16 * pitch, 16 * pitch)
    sel = tiles['wg_atom_ids'][int(tiles['wg_atom_offsets'][t]):int(tiles['wg_atom_offsets'][t + 1])]
    fig, ax = plt.subplots(1, 2, figsize=(12.8, 6.2))
    ax[0].scatter(apos[top, 0], apos[top, 1], s=12, c='0.75')
    if len(sel):
        m = top[sel]
        ax[0].scatter(apos[sel][m, 0], apos[sel][m, 1], s=22, c='darkorange')
    ax[0].add_patch(plt.Rectangle((box[0], box[1]), box[2], box[3], fill=False, ec='crimson', lw=2))
    ax[0].set_aspect('equal')
    rad = rb + 1.0
    ax[0].set_xlim(xl + 0.75 - rad, xl + 0.75 + rad)
    ax[0].set_ylim(yl + 0.75 - rad, yl + 0.75 + rad)
    ax[0].set_xlabel('x [Å]'); ax[0].set_ylabel('y [Å]')
    ax[0].set_title(f'{os.path.basename(mol)}\n16×16 × {pitch} Å = 1.6×1.6 Å\n'
                    f'top-sheet atoms loaded: {r16["n_top"]}    PIC cell {cell:.2f} Å holds ~{pic_mean:.0f}')
    labels = [f'{r["tx"]}×{r["ty"]}\n{r["thr"]} thr' for r in rows]
    x = np.arange(len(rows))
    ax[1].bar(x, [r['ms'] for r in rows], color='steelblue')
    for i, r in enumerate(rows):
        ax[1].text(i, r['ms'], f'{r["n_top"]} atoms', ha='center', va='bottom', fontsize=8)
    ax[1].set_xticks(x); ax[1].set_xticklabels(labels)
    ax[1].set_ylabel(f'scan time [ms]   {nxy}×{nxy}×{nz}')
    ax[1].set_title('square workgroups at 0.1 Å/pixel\nlabel = top-sheet atoms in the tile')
    fig.tight_layout()
    stem = os.path.basename(mol)[:-4]
    fn = os.path.join(OUT, f'wg_bench_{stem}.png')
    fig.savefig(fn, dpi=140)
    plt.close(fig)
    print(f'REVIEW: {fn}', flush=True)
    return rows


def bench_sph(side=24.0):
    """Local-memory split, then FIRE vs sphere-constrained Newton on the CLI scan."""
    from spammm.SPM.AFM import AFMulator, stiffness_Nm_to_eVA2, compact_scan_tile
    from spammm.SPM.AFM_utils import (ScanSpec, afm_df_height_stacks,
                                      plot_afm_variant_height_strip, scan_extent, shared_postprocess)
    step, scan_margin = 0.1, 2.0
    h_min, h_max, h_step, amp = 3.7, 4.7, 0.1, 1.0
    K_LAT = stiffness_Nm_to_eVA2(0.5)
    K_RAD, bond = 20.0, 3.0
    pos, el = square_sheet(side)
    mol = os.path.join(OUT, f"graphene_square{side:.0f}_2L.xyz")
    write_xyz(mol, pos, el)
    ztop = float(pos[:, 2].max())
    h_df, h_Fz, h_scan = afm_df_height_stacks(h_min, h_max, h_step, amp=amp, amp_align=True)
    scan_xs = np.arange(float(pos[:, 0].min() - scan_margin), float(pos[:, 0].max() + scan_margin), step, dtype=np.float32)
    scan_ys = np.arange(float(pos[:, 1].min() - scan_margin), float(pos[:, 1].max() + scan_margin), step, dtype=np.float32)
    spec = ScanSpec(scan_xs=scan_xs, scan_ys=scan_ys, h_df=h_df, h_Fz=h_Fz, h_scan=h_scan,
                    amplitude=amp, osc_dir=(0., 0., 1.), K_LAT=K_LAT, K_RAD=K_RAD,
                    bond_length=bond, scan_margin=scan_margin)
    afm = AFMulator(use_morse=True, nloc=32, use_fire=True)
    afm.load_molecule(mol)
    afm.assign_params(params_path='data/ElementTypes.dat')
    afm.tipQs[:] = 0.0
    afm.dpos0 = np.array([0., 0., -bond, bond], np.float32)
    afm.stiffness = np.array([-K_LAT, -K_LAT, -K_LAT, -K_RAD], np.float32)
    z_pad = 1.5
    qbounds = np.array([[float(scan_xs.min()), float(scan_xs.max())],
                        [float(scan_ys.min()), float(scan_ys.max())],
                        [ztop + float(h_scan.min()) - z_pad, ztop + float(h_scan.max()) + z_pad]])
    afm.fit_contact_pme(query_bounds=qbounds, h_mesh=1.0, halo_nodes=6, bPrint=False)
    p = afm.cpm
    na = int(p.na)
    nx_s, ny_s, nz_s = len(scan_xs), len(scan_ys), len(h_scan)
    b_atoms, b_coeff = na * 16, na * 5 * 4
    print(f"local kernel (the df strip, and sph): ALL {na} atoms in local, mesh stays global", flush=True)
    print(f"  cores {b_atoms/1024:.2f} KB + coeffs {b_coeff/1024:.2f} KB = {(b_atoms+b_coeff)/1024:.2f} KB   mesh local 0", flush=True)
    print(f"  global mesh {p.mesh_shape} h={p.mesh_h} Å  {np.prod(p.mesh_shape)*4/1024:.1f} KB", flush=True)
    z0 = ztop + float(h_scan.max()) + bond
    gx, gy = np.meshgrid(scan_xs, scan_ys, indexing='ij')
    pts = np.zeros((nx_s * ny_s, 4), np.float32)
    pts[:, 0], pts[:, 1], pts[:, 2] = gx.ravel(), gy.ravel(), z0
    tx, ty = compact_scan_tile(step, step)
    tiles = afm._pme_build_wg_tiles(p, pts, nx_s, ny_s, nz_s, -abs(float(h_scan[1] - h_scan[0])), tx=tx, ty=ty, nzL=5)
    mem = afm._pme_tile_local_bytes(tiles)
    print(f"tile kernel (not used by the df strip): {tx}×{ty} px = {tx*step:.1f}×{ty*step:.1f} Å, "
          f"mesh window max {mem['nxL']}×{mem['nyL']}×{mem['nzL']} nodes", flush=True)
    print(f"  cores {mem['atoms']/1024:.2f} KB + coeffs {mem['coeffs']/1024:.2f} KB "
          f"(nloc_max={mem['nloc_max']}) + mesh {mem['mesh']/1024:.2f} KB = {mem['total']/1024:.2f} KB", flush=True)
    dz = float(h_scan[1] - h_scan[0])
    scan_p0 = np.array([float(scan_xs[0]), float(scan_ys[0]), z0], np.float32)
    da = np.array([step, 0, 0], np.float32)
    db = np.array([0, step, 0], np.float32)
    dfs = {}
    for mode in ('fire', 'sph'):
        afm.run_scan_contact_pme(nxy=(nx_s, ny_s), nz=nz_s, dtip=-abs(dz),
                                 scan_p0=scan_p0, scan_da=da, scan_db=db,
                                 core_backend='local', relax_mode=mode)
        t0 = time.perf_counter()
        FEs, _ = afm.run_scan_contact_pme(nxy=(nx_s, ny_s), nz=nz_s, dtip=-abs(dz),
                                          scan_p0=scan_p0, scan_da=da, scan_db=db,
                                          core_backend='local', relax_mode=mode)
        ms = (time.perf_counter() - t0) * 1e3
        r = shared_postprocess(FEs[:, :, ::-1, :], spec, backend_name='contact_pme')
        dfs[mode] = r.df
        it = getattr(afm, 'last_relax_iters', None)
        extra = f"  iters mean={np.mean(np.abs(it)):.2f}" if it is not None else ""
        print(f"  {mode:4s}  {ms:.1f} ms{extra}", flush=True)
    d = dfs['sph'] - dfs['fire']
    print(f"  max|df sph−fire|={np.max(np.abs(d)):.3e}   "
          f"df fire [{dfs['fire'].min():.3e},{dfs['fire'].max():.3e}]", flush=True)
    variants = {'fire': {'df': dfs['fire']}, 'sph': {'df': dfs['sph']}}
    rows = [('df', 'fire', 'FIRE', 'gray'), ('df', 'sph', 'sphere', 'gray')]
    plot_afm_variant_height_strip(
        variants, rows, h_df, os.path.join(OUT, 'heights_df_sph.png'),
        scale='per_image', title='df  FIRE vs sphere-constrained Newton   K_LAT=0.5 N/m',
        extent=scan_extent(scan_xs, scan_ys), amp=amp, amp_align=True,
        long_axis_vertical=True, tight=True)


def bench_solvers(sides=(16, 24, 32, 48, 64), modes=('fire', 'qn', 'sph')):
    """CLI PP scan (local kernel) versus sheet size, one curve per relaxer."""
    from spammm.SPM.AFM import AFMulator, stiffness_Nm_to_eVA2
    from spammm.SPM.AFM_utils import ScanSpec, afm_df_height_stacks, shared_postprocess
    step, scan_margin = 0.1, 2.0
    K_LAT = stiffness_Nm_to_eVA2(0.5)
    K_RAD, bond = 20.0, 3.0
    afm = AFMulator(use_morse=True, nloc=32, use_fire=True)
    rows = []
    hdr = f"{'side':>5} {'na':>6} {'scan':>12} |" + "".join(f" {m:>8}" for m in modes) + " |" + "".join(f" ddf_{m:>4}" for m in modes if m != 'fire')
    print(hdr, flush=True)
    print("PP scan K_LAT=0.5 N/m  K_RAD=20  L=3 Å  df 3.7–4.7 amp=1  second run", flush=True)
    print(f"{'':>5} {'':>6} {'':>12} | {'FIRE':>8} {'QN':>8} {'sph':>8} {'tile':>8} {'bucket':>8}", flush=True)
    for side in sides:
        print(f"  building {side:.0f} Å ...", flush=True)
        pos, el = square_sheet(side)
        mol = os.path.join(OUT, f"graphene_square{side:.0f}_2L.xyz")
        write_xyz(mol, pos, el)
        afm.load_molecule(mol)
        afm.assign_params(params_path='data/ElementTypes.dat')
        afm.tipQs[:] = 0.0
        afm.dpos0 = np.array([0., 0., -bond, bond], np.float32)
        afm.stiffness = np.array([-K_LAT, -K_LAT, -K_LAT, -K_RAD], np.float32)
        ztop = float(pos[:, 2].max())
        h_df, h_Fz, h_scan = afm_df_height_stacks(3.7, 4.7, 0.1, amp=1.0, amp_align=True)
        scan_xs = np.arange(float(pos[:, 0].min() - scan_margin), float(pos[:, 0].max() + scan_margin), step, dtype=np.float32)
        scan_ys = np.arange(float(pos[:, 1].min() - scan_margin), float(pos[:, 1].max() + scan_margin), step, dtype=np.float32)
        spec = ScanSpec(scan_xs=scan_xs, scan_ys=scan_ys, h_df=h_df, h_Fz=h_Fz, h_scan=h_scan,
                        amplitude=1.0, osc_dir=(0., 0., 1.), K_LAT=K_LAT, K_RAD=K_RAD,
                        bond_length=bond, scan_margin=scan_margin)
        z_pad = 1.5
        qbounds = np.array([[float(scan_xs.min()), float(scan_xs.max())],
                            [float(scan_ys.min()), float(scan_ys.max())],
                            [ztop + float(h_scan.min()) - z_pad, ztop + float(h_scan.max()) + z_pad]])
        afm.fit_contact_pme(query_bounds=qbounds, h_mesh=1.0, halo_nodes=6, bPrint=False)
        nx_s, ny_s, nz_s = len(scan_xs), len(scan_ys), len(h_scan)
        dz = float(h_scan[1] - h_scan[0])
        p0 = np.array([float(scan_xs[0]), float(scan_ys[0]), ztop + float(h_scan.max()) + bond], np.float32)
        base = dict(nxy=(nx_s, ny_s), nz=nz_s, dtip=-abs(dz), scan_p0=p0,
                    scan_da=np.array([step, 0, 0], np.float32), scan_db=np.array([0, step, 0], np.float32))
        jobs = [(m, 'local', {}) for m in modes] + [('sph', 'tile', dict(tile_xy=(16, 16), nzL=5, rho_max=1.0))]
        timed, dfs = {}, {}
        for mode, backend, extra in jobs:
            key = f'{backend}:{mode}'
            print(f"    {key} {nx_s}×{ny_s}×{nz_s} ...", flush=True)
            try:
                kw = dict(base, core_backend=backend, relax_mode=mode, **extra)
                afm.queue.finish()
                afm.run_scan_contact_pme(**kw)
                afm.queue.finish()
                t0 = time.perf_counter()
                FEs, _ = afm.run_scan_contact_pme(**kw, bAlloc=False)
                afm.queue.finish()
                timed[key] = (time.perf_counter() - t0) * 1e3
                dfs[key] = shared_postprocess(FEs[:, :, ::-1, :], spec, backend_name='contact_pme').df
            except (MemoryError, ValueError) as e:
                timed[key] = np.nan
                print(f"    {key} skipped: {e}", flush=True)
        if not np.isfinite(timed.get('local:fire', np.nan)):
            print(f"    bucket:fire {nx_s}×{ny_s}×{nz_s} ...", flush=True)
            kw = dict(base, core_backend='bucket', relax_mode='fire')
            afm.queue.finish()
            afm.run_scan_contact_pme(**kw)
            afm.queue.finish()
            t0 = time.perf_counter()
            FEs, _ = afm.run_scan_contact_pme(**kw, bAlloc=False)
            afm.queue.finish()
            timed['bucket:fire'] = (time.perf_counter() - t0) * 1e3
            dfs['bucket:fire'] = shared_postprocess(FEs[:, :, ::-1, :], spec, backend_name='contact_pme').df
        ref = dfs.get('local:fire', dfs.get('bucket:fire'))
        ddf = {}
        for k, d in dfs.items():
            if k.endswith('fire') or ref is None:
                continue
            ddf[k] = float(np.max(np.abs(d - ref)))
        row = dict(side=float(side), na=int(afm.cpm.na), nxy=nx_s, ny=ny_s, nz=nz_s, ms=timed, ddf=ddf)
        rows.append(row)
        def _ms(k):
            v = timed.get(k, np.nan)
            return f"{v:8.1f}" if np.isfinite(v) else f"{'—':>8}"
        line = (f"{side:5.0f} {row['na']:6d} {nx_s:4d}×{ny_s:<3d}×{nz_s:<2d} |"
                f" {_ms('local:fire')} {_ms('local:qn')} {_ms('local:sph')} {_ms('tile:sph')} {_ms('bucket:fire')}")
        if ddf:
            line += " |" + "".join(f" {k} {v:.2e}" for k, v in ddf.items())
        print(line, flush=True)
    na = np.array([r['na'] for r in rows])
    npx = np.array([r['nxy'] * r['ny'] * r['nz'] for r in rows], np.float64)
    fig, ax = plt.subplots(1, 2, figsize=(12.2, 5.2))
    series = [('local:fire', 'o-', 'local FIRE'), ('local:qn', 's-', 'local quasi-Newton'),
              ('local:sph', '^-', 'local sphere'), ('tile:sph', 'v--', 'tile sphere'),
              ('bucket:fire', 'D--', 'bucket FIRE')]
    for key, fmt, lab in series:
        y = np.array([r['ms'].get(key, np.nan) for r in rows], np.float64)
        if not np.isfinite(y).any():
            continue
        ax[0].semilogy(na, y, fmt, label=lab)
        ax[1].semilogy(na, y * 1e3 / npx, fmt, label=lab)
    ax[0].set_xlabel('atoms (bilayer)')
    ax[0].set_ylabel('scan wall [ms]')
    ax[0].set_title('second run, K_LAT=0.5 N/m\ndf window 3.7–4.7 Å, amp=1')
    ax[0].legend()
    ax[1].set_xlabel('atoms (bilayer)')
    ax[1].set_ylabel('µs / pixel-slice')
    ax[1].set_title('local kernel stops when cores exceed 48 KB')
    ax[1].legend()
    fig.tight_layout()
    fn = os.path.join(OUT, 'solvers_scale.png')
    fig.savefig(fn, dpi=140)
    plt.close(fig)
    print(f'REVIEW: {fn}', flush=True)
    return rows


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--partition', action='store_true', help='WG in-box/halo counts and local-memory table')
    ap.add_argument('--pic', action='store_true', help='PIC 3x3 partition the core kernel walks')
    ap.add_argument('--compact', action='store_true', help='square vs elongated scan tiles, local memory')
    ap.add_argument('--pitch', type=float, default=None, help='AFM pixel pitch in Å (0.1 → 16x16 = 1.6 Å)')
    ap.add_argument('--bench', action='store_true', help='WG-size timing at 0.1 Å/pixel on a square sheet')
    ap.add_argument('--cells', action='store_true', help='PIC cells and mesh nodes loaded by one 16x16 tile')
    ap.add_argument('--scale', action='store_true', help='fit + scan time versus graphene sheet size')
    ap.add_argument('--gridff', action='store_true', help='GridFF voxel build versus PME, plus supersampled spline fit')
    ap.add_argument('--heights', action='store_true', help='Fz height strips: GridFF vs 1x, 2x, 3x spline')
    ap.add_argument('--sph', action='store_true', help='FIRE vs sphere-constrained Newton, plus local-memory split')
    ap.add_argument('--solvers', action='store_true', help='FIRE / quasi-Newton / sphere scan time versus sheet size')
    ap.add_argument('--nacl', action='store_true', help='NaCl(001) square lattice instead of graphene')
    ap.add_argument('--mol', default='data/xyz/PTCDA.xyz')
    ap.add_argument('--replica', type=int, nargs=2, default=None)
    ap.add_argument('--graphene', type=int, nargs=2, default=None, metavar=('NX', 'NY'))
    ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--defect', default='N')
    ap.add_argument('--rho', type=float, default=1.0)
    ap.add_argument('--scan', action='store_true')
    args = ap.parse_args()

    if args.scale:
        bench_scale()
        return
    if args.gridff:
        bench_gridff()
        return
    if args.heights:
        plot_supersample_heights()
        return
    if args.sph:
        bench_sph()
        return
    if args.solvers:
        bench_solvers()
        return
    from spammm.SPM.AFM import AFMulator
    afm = AFMulator(use_morse=True, use_fire=False)
    mol = args.mol
    if args.replica:
        from tests.SPM.bench_contact_pme import make_replica_xyz
        mol = os.path.join(OUT, f"replica_{args.replica[0]}x{args.replica[1]}.xyz")
        if not os.path.exists(mol):
            make_replica_xyz(args.mol, *args.replica, mol)
    if args.bench and args.nacl:
        mol = os.path.join(OUT, "nacl_12x12_2L.xyz")
        if not os.path.exists(mol):
            pos, el = make_nacl()
            write_xyz(mol, pos, el)
    elif args.bench or args.cells:
        mol = os.path.join(OUT, "graphene_square24_2L.xyz")
        if not os.path.exists(mol):
            pos, el = square_graphene(24.0, layers=2)
            write_xyz(mol, pos, el)
    if args.graphene:
        mol = os.path.join(OUT, f"graphene_{args.graphene[0]}x{args.graphene[1]}x{args.layers}L_{args.defect}.xyz")
        if not os.path.exists(mol):
            make_graphene_xyz(*args.graphene, layers=args.layers, defect=args.defect, out_path=mol)
    afm.load_molecule(mol)
    afm.assign_params(params_path='data/ElementTypes.dat')
    afm.tipQs[:] = 0.0
    if args.pic:
        plot_pic_kernel(afm, mol, delta_b=1.2)
        return
    if args.compact:
        sweep_compact_tiles(afm, mol)
        return
    if args.pitch is not None:
        plot_pitch_tile(afm, mol, pitch=args.pitch)
        return
    if args.cells:
        plot_loaded_cells(afm, mol)
        return
    if args.bench:
        bench_wg(afm, mol)
        return
    if args.partition:
        run_box_partition(afm, mol, rho=args.rho)
        return

    # ---------- fit benchmark ----------
    t0 = time.perf_counter(); afm.fit_contact_pme(h_mesh=1.0, bPrint=False); t_fit = time.perf_counter() - t0
    p = afm.cpm
    print(f"== {os.path.basename(mol)}: na={p.na} mesh={p.mesh_shape} fit={t_fit:.3f}s", flush=True)
    print(f"   bucket cell={p.bucket_cell_size:.2f}A grid={p.bucket_nbx}x{p.bucket_nby} "
          f"core d_span={p.core_d_span:.3f} r_b_max={float(np.max(np.asarray(p.core_fit.r_lo))+p.core_d_span):.2f}", flush=True)

    # ---------- scan ----------
    if args.scan:
        t0 = time.perf_counter()
        FEs, pts = afm.run_scan_contact_pme(nxy=(48, 48), nz=60, dtip=-0.1, relax_mode='sph',
                                          core_backend='tile', rho_max=args.rho, nzL=6)
        afm.queue.finish()
        st = afm.last_scan_status
        print(f"   tile scan rho_cap={args.rho}: {time.perf_counter() - t0:.3f}s "
              f"esc={int(afm.last_pme_esc.sum())} invalid={((st & 8) != 0).mean():.3f}", flush=True)

    # ---------- tiles + PIC stats ----------
    from spammm.SPM.AFM import build_scan_xy_points_vectorized
    scan_p0, scan_da, scan_db = afm._scan_grid_auto((48, 48))
    pts = build_scan_xy_points_vectorized(scan_p0, scan_da, scan_db, 48, 48)
    tiles = afm._pme_build_wg_tiles(p, pts, 48, 48, 60, -0.1, tx=8, ty=4, rho_max=args.rho, nzL=6)
    S = pic_stats(afm, tiles)
    nloc, inside, occ = S['nloc'], S['inside'], S['occ']
    print(f"   tiles {tiles['ntx']}x{tiles['nty']} (tile_xy={tiles['tx']}x{tiles['ty']})  "
          f"nloc: min={nloc.min()} mean={nloc.mean():.0f} max={nloc.max()} of na={p.na}", flush=True)
    print(f"   atoms inside footprint: mean={inside.mean():.1f}  halo (r_b reach): mean={(nloc - inside).mean():.1f}", flush=True)
    print(f"   bucket occ: mean={occ.mean():.1f} max={occ.max()} cell={S['cell']:.2f}A  "
          f"~candidates/eval(3x3 cells)~{occ.mean() * 9:.0f}", flush=True)

    # ---------- PIC cell detail: inside / halo / excluded + cutoff spheres ----------
    apos_all = np.asarray(p.atom_pos)
    r_b = np.asarray(p.core_fit.r_lo) + p.core_d_span        # exact basis cutoff (phi=0 at r_b)
    x0b, y0b = float(p.bucket_bounds[0]), float(p.bucket_bounds[1])
    nbx, nby, cs = int(p.bucket_nbx), int(p.bucket_nby), float(p.bucket_cell_size)
    # cell containing the atom nearest the flake center (or densest cell)
    occ = np.diff(np.asarray(p.bucket_offsets)[:nbx * nby])
    # cell containing the flake center (symmetric surroundings)
    bx = min(max(int((0.0 - x0b) / cs), 0), nbx - 1)
    by = min(max(int((0.0 - y0b) / cs), 0), nby - 1)
    c_lo = np.array([x0b + bx * cs, y0b + by * cs])
    c_hi = c_lo + cs
    batoms = np.asarray(p.bucket_atoms)
    boffs = np.asarray(p.bucket_offsets)
    inside_ids = batoms[boffs[by * nbx + bx]:boffs[by * nbx + bx + 1]]
    # PP z band for the proper scan (h_scan above top atom; morse CLI defaults)
    ztop_at = float(apos_all[:, 2].max())
    ppz_lo, ppz_hi = ztop_at + 2.7, ztop_at + 5.7       # h_scan=[2.7,5.7] + amp margin
    # halo: atoms whose r_b sphere reaches the cell box x PP z-band (3D)
    cc = np.array([[(c_lo[0] + c_hi[0]) / 2, (c_lo[1] + c_hi[1]) / 2, (ppz_lo + ppz_hi) / 2]])
    hw3 = np.array([[cs / 2, cs / 2, (ppz_hi - ppz_lo) / 2]])
    d3 = np.maximum(np.abs(apos_all - cc) - hw3, 0.0)
    dist3 = np.sqrt((d3 * d3).sum(axis=1))
    inarr = np.arange(len(apos_all))
    halo_ids = np.flatnonzero((dist3 < r_b) & ~np.isin(inarr, inside_ids))
    incell = np.zeros(len(apos_all), bool); incell[inside_ids] = True
    inhalo = np.zeros(len(apos_all), bool); inhalo[halo_ids] = True
    deep = (apos_all[:, 2] < ztop_at - 0.5)   # below-surface layer: mesh-only, never core
    sel = incell | inhalo | deep

    fig2, ax2 = plt.subplots(1, 2, figsize=(15, 7.2))
    for ax, j1, l1 in [(ax2[0], 1, 'y'), (ax2[1], 2, 'z')]:
        ax.scatter(apos_all[~sel, 0], apos_all[~sel, j1], s=16, c='0.8',
                   label='excluded (mesh only)')
        ax.scatter(apos_all[deep, 0], apos_all[deep, j1], s=24, c='royalblue', marker='v',
                   label=f'below-surface layer n={int(deep.sum())} (mesh only)')
        ax.scatter(apos_all[inhalo & ~deep, 0], apos_all[inhalo & ~deep, j1], s=26, c='orange',
                   label=f'halo: r_b reaches cell+PPband n={int((inhalo & ~deep).sum())}')
        ax.scatter(apos_all[incell & ~deep, 0], apos_all[incell & ~deep, j1], s=34, c='seagreen',
                   label=f'inside cell (core-evaluated) n={int((incell & ~deep).sum())}')
        ax.set_aspect('equal'); ax.set_xlabel('x'); ax.set_ylabel(l1)
    th = np.linspace(0, 2 * np.pi, 72)
    for ia in inside_ids:                       # cutoff spheres only for inside atoms
        ax2[0].plot(apos_all[ia, 0] + r_b[ia] * np.cos(th), apos_all[ia, 1] + r_b[ia] * np.sin(th),
                    color='tab:blue', lw=0.4, alpha=0.55)
        ax2[1].plot(apos_all[ia, 0] + r_b[ia] * np.cos(th), apos_all[ia, 2] + r_b[ia] * np.sin(th),
                    color='tab:blue', lw=0.4, alpha=0.45)
    ax2[0].plot([], [], color='tab:blue', lw=0.6, label=f'r_b={r_b.max():.2f}A cutoff (phi=0)')
    ax2[0].add_patch(plt.Rectangle(c_lo, cs, cs, fill=False, ec='crimson', lw=1.8, label='PIC cell'))
    ax2[0].add_patch(plt.Rectangle(c_lo - r_b.max(), cs + 2 * r_b.max(), cs + 2 * r_b.max(),
                                   fill=False, ec='orange', lw=0.9, ls=':', label='cell + r_b'))
    ax2[1].add_patch(plt.Rectangle((c_lo[0], ppz_lo), cs, ppz_hi - ppz_lo, fill=False,
                                   ec='crimson', lw=1.8, label='PP z sweep above cell'))
    ax2[1].axhline(ztop_at, color='crimson', lw=0.8, ls='--', label='top atom')
    ax2[0].legend(loc='upper right', fontsize=8)
    ax2[1].legend(loc='lower right', fontsize=8)
    ax2[0].set_title(f'PIC cell ({bx},{by}) cell={cs:.2f}A — xy view, r_b spheres')
    ax2[1].set_title(f'side view (XZ) — PP sweeps z=[{ppz_lo:.1f},{ppz_hi:.1f}]; bottom layer z-screened')
    fig2.tight_layout()
    fn2 = os.path.join(OUT, f'pic_cell_detail_{os.path.basename(mol)[:-4]}.png')
    fig2.savefig(fn2, dpi=140); plt.close(fig2)
    print(f"REVIEW: {fn2}", flush=True)

    # ---------- atom -> cell/mesh projection (uniform square cells) ----------
    figp, axp = plt.subplots(1, 2, figsize=(15, 6.5))
    # left: atom -> PIC cell scatter map (each atom owned by exactly one square cell)
    cid = np.zeros(len(apos_all), dtype=np.int64)
    for b in range(nbx * nby):
        cid[batoms[boffs[b]:boffs[b + 1]]] = b
    sc = axp[0].scatter(apos_all[:, 0], apos_all[:, 1], c=cid % 20, cmap='tab20', s=22)
    for gx in range(nbx + 1):
        axp[0].axvline(x0b + gx * cs, color='k', lw=0.4, alpha=0.6)
    for gy in range(nby + 1):
        axp[0].axhline(y0b + gy * cs, color='k', lw=0.4, alpha=0.6)
    axp[0].set_aspect('equal'); axp[0].set_xlabel('x'); axp[0].set_ylabel('y')
    axp[0].set_title(f'atom -> PIC cell projection (square cells {cs:.2f}A)\ncolor = owning cell (mod 16)')
    # right: atom -> coarse mesh projection is DENSE (long-range v_L at every node):
    # show mesh grid + one atom's |v_L| contribution map on the top-layer plane
    nxm, nym, nzm = p.mesh_shape; h = float(p.mesh_h)
    oxm, oym = float(p.mesh_origin[0]), float(p.mesh_origin[1])
    coeff = np.asarray(p.mesh_coeffs).reshape(nxm, nym, nzm)
    izm = int(np.argmin(np.abs(p.mesh_origin[2] + h * np.arange(nzm) - (ztop_at))))
    field = coeff[:, :, izm]
    im = axp[1].imshow(field.T, origin='lower', cmap='seismic',
                       extent=(oxm, oxm + (nxm - 1) * h, oym, oym + (nym - 1) * h),
                       vmin=-np.abs(field).max(), vmax=np.abs(field).max())
    for gx in range(0, nxm, 5):
        axp[1].axvline(oxm + gx * h, color='k', lw=0.3, alpha=0.5)
    for gy in range(0, nym, 5):
        axp[1].axhline(oym + gy * h, color='k', lw=0.3, alpha=0.5)
    axp[1].scatter(apos_all[:, 0], apos_all[:, 1], s=10, c='k', alpha=0.7)
    axp[1].set_aspect('equal'); axp[1].set_xlabel('x'); axp[1].set_ylabel('y')
    axp[1].set_title(f'atom -> coarse mesh (dense, all atoms -> all nodes)\nV_mesh plane z={p.mesh_origin[2]+izm*h:.1f}A, h={h}A, nodes={nxm}x{nym}')
    figp.colorbar(im, ax=axp[1], shrink=0.8)
    figp.tight_layout()
    fnp = os.path.join(OUT, f'projection_{os.path.basename(mol)[:-4]}.png')
    figp.savefig(fnp, dpi=140); plt.close(figp)
    print(f"REVIEW: {fnp}", flush=True)

    # ---------- field decomposition on the fitted system (mesh vs core) ----------
    from spammm.surfaces.CoarseMesh import CoarseMesh, eval_mesh
    from spammm.surfaces.PICCore import eval_core
    mesh = CoarseMesh(h=float(p.mesh_h), origin=np.asarray(p.mesh_origin),
                      coeffs=np.asarray(p.mesh_coeffs).reshape(p.mesh_shape),
                      halo=6, query_interior=(np.asarray(p.query_interior[0]), np.asarray(p.query_interior[1])))
    # x-z plane through the N defect row (y = cN y ~ 0.71)
    yq = 0.71
    xs_q = np.arange(-9.0, 9.01, 0.12)
    zs_q = np.arange(4.6, 9.01, 0.12)
    gx, gz = np.meshgrid(xs_q, zs_q, indexing='xy')
    qq = np.stack([gx.ravel(), np.full(gx.size, yq), gz.ravel()], axis=1)
    print(f'  field decomp: {len(qq)} queries eval_mesh+eval_core ...', flush=True)
    Em, _ = eval_mesh(mesh, qq)
    Ec, _ = eval_core(qq, apos_all, p.core_fit)
    Em = Em.reshape(gx.shape); Ec = Ec.reshape(gx.shape)
    figd, axd = plt.subplots(1, 4, figsize=(20, 4.8))
    vmm = np.percentile(np.abs(Em), 99)
    axd[0].pcolormesh(gx, gz, Em, cmap='seismic', vmin=-vmm, vmax=vmm, shading='auto')
    axd[0].set_title(f'V_mesh (smooth, dense — all atoms)')
    vmc = np.percentile(np.abs(Ec), 99)
    axd[1].pcolormesh(gx, gz, Ec, cmap='seismic', vmin=-vmc, vmax=vmc, shading='auto')
    axd[1].set_title(f'V_core (compact, only atoms within r_b)')
    vmt = np.percentile(np.abs(Em + Ec), 99)
    axd[2].pcolormesh(gx, gz, Em + Ec, cmap='seismic', vmin=-vmt, vmax=vmt, shading='auto')
    axd[2].set_title('V_mesh + V_core (total)')
    for a in axd[:3]:
        sel_row = np.abs(apos_all[:, 1] - yq) < 3.0
        a.scatter(apos_all[sel_row, 0], apos_all[sel_row, 2], s=12, c='k')
        a.set_xlabel('x'); a.set_ylabel('z'); a.set_aspect('equal')
    # 1D profile at PP height z=6.1 (h=2.7 above top atom 3.4)
    izq = int(np.argmin(np.abs(zs_q - 6.1)))
    axd[3].plot(xs_q, Em[izq], 'b-', label='$V_{mesh}$')
    axd[3].plot(xs_q, Ec[izq], 'r--', label='$V_{core}$')
    axd[3].plot(xs_q, Em[izq] + Ec[izq], 'k-', lw=1.5, label='total')
    for xa in apos_all[np.abs(apos_all[:, 2] - ztop_at) < 0.1, 0]:
        if -9 < xa < 9:
            axd[3].axvline(xa, color='gray', lw=0.3)
    axd[3].set_xlabel('x'); axd[3].set_title(f'profile z={zs_q[izq]:.1f}A (h=2.7)')
    axd[3].legend(fontsize=8)
    figd.tight_layout()
    fnd = os.path.join(OUT, f'field_decomp_{os.path.basename(mol)[:-4]}.png')
    figd.savefig(fnd, dpi=140); plt.close(figd)
    print(f"REVIEW: {fnd}", flush=True)

    # ---------- 1D core basis + fitted residual ----------
    from spammm.surfaces.PICCore import core_basis, CORE_POWERS
    from spammm.surfaces.PMESplit import soft_core_split, precompute_split_cache
    ia = int(inside_ids[0]) if len(inside_ids) else 0
    pi_at = afm.cpm.split_params.with_atom(ia)
    cache_at = precompute_split_cache(pi_at)
    r = np.linspace(0.2, float(r_b[ia]) + 0.5, 400)
    ph, dph = core_basis(r, float(p.core_fit.r_lo[ia]), float(r_b[ia]))
    s = soft_core_split(np.clip(r, 0.3, r_b[ia] - 1e-3), pi_at, cache=cache_at)
    fig3, ax3 = plt.subplots(1, 4, figsize=(21, 4.6))
    # v = v_L + v_S decomposition (PAW even-poly inside r_b)
    ax3[0].plot(r, s['v'], 'k-', lw=1.5, label='$v$ total (Morse+Q)')
    ax3[0].plot(r, s['v_L'], 'b-', lw=1.5, label='$v_L$ -> B-spline mesh')
    ax3[0].plot(r, s['v_S'], 'r--', lw=1.2, label='$v_S=v-v_L$ -> compact core')
    ax3[0].axvline(float(p.core_fit.r_lo[ia]), color='k', ls=':', lw=1)
    ax3[0].axvline(float(r_b[ia]), color='crimson', ls='--', lw=1.2)
    ax3[0].set_ylim(-0.4, 0.4)
    ax3[0].set_xlabel('r [A]'); ax3[0].set_title('split decomposition')
    ax3[0].legend(fontsize=8)
    for m, pn in enumerate(CORE_POWERS):
        ax3[1].plot(r, ph[:, m], label=f'$t^{{{pn}}}$')
    ax3[1].axvline(float(p.core_fit.r_lo[ia]), color='k', ls=':', lw=1, label=f'r_lo={p.core_fit.r_lo[ia]:.2f}')
    ax3[1].axvline(float(r_b[ia]), color='crimson', ls='--', lw=1.2, label=f'r_b={r_b[ia]:.2f} (cutoff)')
    ax3[1].set_xlabel('r [A]'); ax3[1].set_title('core basis  $\\phi_m=t^{2^n}$,  $t=(r_b-r)/(r_b-r_{lo})$')
    ax3[1].legend(fontsize=8)
    mfit = r >= p.core_fit.r_lo[ia]
    ax3[2].plot(r[mfit], s['v_S'][mfit], 'k-', lw=1.5, label='$v_S$ residual (fit target)')
    ax3[2].plot(r, ph @ np.asarray(p.core_fit.coeffs)[ia], 'r--', lw=1, label='$\\Sigma c_m\\phi_m$ fitted')
    ax3[2].axvline(float(r_b[ia]), color='crimson', ls='--', lw=1.2)
    ax3[2].set_xlim(p.core_fit.r_lo[ia] - 0.1, r_b[ia] + 0.3)
    ax3[2].set_xlabel('r [A]'); ax3[2].set_title(f'atom {ia} compact core on [r_lo, r_b]')
    ax3[2].legend(fontsize=8)
    resid = ph[mfit] @ np.asarray(p.core_fit.coeffs)[ia] - s['v_S'][mfit]
    ax3[3].plot(r[mfit], resid, 'b-', lw=1)
    ax3[3].axhline(0, color='k', lw=0.5)
    ax3[3].axvline(float(r_b[ia]), color='crimson', ls='--', lw=1.2)
    ax3[3].set_xlabel('r [A]'); ax3[3].set_title(f'fit residual (max {np.abs(resid).max():.2e} eV)')
    ax3[3].set_xlim(p.core_fit.r_lo[ia] - 0.1, r_b[ia] + 0.3)
    fig3.tight_layout()
    fn3 = os.path.join(OUT, f'core_basis_1d_{os.path.basename(mol)[:-4]}.png')
    fig3.savefig(fn3, dpi=140); plt.close(fig3)
    print(f"REVIEW: {fn3}", flush=True)

    # ---------- plots ----------
    apos = np.asarray(p.atom_pos)
    fig, axs = plt.subplots(1, 3, figsize=(17, 5.2))
    ax = axs[0]
    ax.scatter(apos[:, 0], apos[:, 1], s=6, c='0.3')
    x0b, y0b = p.bucket_bounds[0], p.bucket_bounds[1]
    for gx in range(int(p.bucket_nbx) + 1):
        ax.axvline(x0b + gx * S['cell'], color='steelblue', lw=0.4, alpha=0.5)
    for gy in range(int(p.bucket_nby) + 1):
        ax.axhline(y0b + gy * S['cell'], color='steelblue', lw=0.4, alpha=0.5)
    ox, oy = p.mesh_origin[0], p.mesh_origin[1]
    for t in range(len(tiles['tile_desc'])):
        d = tiles['tile_desc'][t]
        r = plt.Rectangle((ox + d[0], oy + d[1]), d[2], d[3], fill=False, ec='crimson', lw=0.8)
        ax.add_patch(r)
    ax.set_aspect('equal'); ax.set_title(f"{os.path.basename(mol)}\nna={p.na}  buckets(blue) vs mesh-window tiles(red)")
    ax = axs[1]
    ax.hist(occ, bins=range(0, int(occ.max()) + 2), color='steelblue', alpha=0.8)
    ax.set_title(f"PIC bucket occupancy (cell={S['cell']:.1f}A)\nmean={occ.mean():.1f} max={occ.max()}")
    ax.set_xlabel('atoms per bucket cell')
    ax = axs[2]
    w = 0.4
    ax.hist([inside, nloc - inside], bins=np.arange(0, nloc.max() + 5, 5), stacked=True,
            label=['inside footprint', 'halo (r_b reach)'], color=['seagreen', 'orange'])
    ax.set_title(f"tile atoms rho_cap={args.rho}A\nmean in={inside.mean():.0f} halo={(nloc - inside).mean():.0f}")
    ax.legend(); ax.set_xlabel('atoms per WG tile')
    fig.tight_layout()
    fn = os.path.join(OUT, f"pic_map_{os.path.basename(mol)[:-4]}_rho{args.rho}.png")
    fig.savefig(fn, dpi=140); plt.close(fig)
    print(f"REVIEW: {fn}", flush=True)

if __name__ == '__main__':
    main()
