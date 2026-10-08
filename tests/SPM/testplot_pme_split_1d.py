"""1D illustration of the ContactPME PAW split on a model Morse+Coulomb potential.

v(r)  = Morse + damped Coulomb (combined_atom_potential) — the sharp true wall
v_L   = even soft poly P(r)=a0+a2r²+a4r⁴+a6r⁶ inside r_b, = v outside — MESH part
v_S   = v − v_L, compact on [r_lo, r_b] — CORE part (fitted by t^p, p=2,4,8,16,32)

Panels: (a) symlog overview v / v_L / v_S / fit;
        (b) linear zoom on the core window — the actual split;
        (c) radial force dv/dr components;
        (d) the 5 u*=u basis functions and c_m·t^p vs v_S on the basis domain.
Output: debug/testplot_pme_split_1d/pme_split_1d.png  (REVIEW)
"""

import os
import numpy as np
import matplotlib.pyplot as plt

OUT = 'debug/testplot_pme_split_1d'


def main():
    from spammm.surfaces.PMESplit import SplitParams, soft_core_split
    from spammm.surfaces.PICCore import fit_core_paw_grid, core_basis, CORE_POWERS

    # model atom: carbon vs CO tip (typical cLJ numbers)
    R0, E0, alpha, q_tip, r_damp = 3.30, 0.0200, 1.8, 0.0, 0.1
    delta_in, delta_b = 1.0, 2.0
    p = SplitParams(R0=np.array([R0]), E0=np.array([E0]), q=np.array([0.0]),
                    alpha=alpha, q_tip=q_tip, r_damp=r_damp,
                    split_mode='paw', delta_in=delta_in, delta_b=delta_b)
    r_lo, r_b = float(p.r_lo[0]), float(p.r_b[0])

    coeffs = fit_core_paw_grid(p)                     # production per-atom 5×5 fit
    print(f'r_lo={r_lo:.2f} r_b={r_b:.2f}  coeffs={coeffs[0]}')

    r = np.linspace(0.2, r_b + 2.5, 4000)
    s = soft_core_split(r, p)
    v, v_L, v_S = s['v'], s['v_L'], s['v_S']
    dv, dv_L, dv_S = s['dvdr'], s['dv_L_dr'], s['dv_S_dr']

    phi, dphi = core_basis(r, r_lo, r_b)
    v_S_fit = phi @ coeffs[0]

    fig, ax = plt.subplots(4, 1, figsize=(9, 14), sharex=True,
                           gridspec_kw=dict(hspace=0.14))

    for a in ax:
        a.axvspan(r_lo, r_b, color='0.93', zorder=0)
        for x, lab in ((R0, 'R0'), (r_lo, 'r_lo'), (r_b, 'r_b')):
            a.axvline(x, color='0.5', ls=':', lw=1)

    # (a) symlog overview
    a = ax[0]
    a.plot(r, v, 'k-', lw=1.6, label='v (true, sharp) — full potential')
    a.plot(r, v_L, 'b-', lw=1.6, label='$v_L$ = soft poly → MESH')
    a.plot(r, v_S, 'r-', lw=1.6, label='$v_S$ = v − $v_L$ → CORE')
    a.plot(r, v_S_fit, 'm--', lw=1.1, label='fitted core $\\Sigma c_m t^{p_m}$')
    a.set_yscale('symlog', linthresh=1e-2)
    a.set_ylabel('E [eV]')
    a.set_title(f'PAW split  (R0={R0}, E0={E0}, α={alpha};  r_lo=R0−{delta_in}={r_lo:.2f}, r_b=R0+{delta_b}={r_b:.2f})')
    a.legend(loc='lower left', fontsize=9)
    a.text(R0 + .05, 8.0, 'R0'); a.text(r_lo - .15, 8.0, 'r_lo'); a.text(r_b + .1, 8.0, 'r_b')

    # (b) linear zoom — the actual split in the AFM-relevant window
    a = ax[1]
    a.plot(r, v, 'k-', lw=2.0, label='v (true)')
    a.plot(r, v_L, 'b-', lw=2.0, label='$v_L$ → mesh')
    a.plot(r, v_S, 'r-', lw=2.0, label='$v_S$ → core')
    a.plot(r, v_S_fit, 'm--', lw=1.2, label='core fit')
    for hm, mk, cc in ((0.5, 'o', 'tab:cyan'), (1.0, 's', 'tab:purple')):
        rm = np.arange(0.2, r_b + 2.5, hm)
        sm = soft_core_split(rm, p)
        a.plot(rm, sm['v_L'], mk, ms=6, mfc='none', color=cc,
               label=f'mesh nodes h={hm}', zorder=5)
    a.set_xlim(1.6, r_b + 1.2)
    a.set_ylim(-0.05, 1.2)
    a.axhline(0, color='0.7', lw=0.5)
    a.set_ylabel('E [eV]  (linear zoom)')
    a.set_title('core window: mesh carries the smooth bowl (dots = mesh nodes), core carries the wall')
    a.legend(loc='upper right', fontsize=9)

    # (c) radial force
    a = ax[2]
    a.plot(r, dv, 'k-', lw=1.4, label='dv/dr (wall force)')
    a.plot(r, dv_L, 'b-', lw=1.4, label='d$v_L$/dr — mesh slope')
    a.plot(r, dv_S, 'r-', lw=1.4, label='d$v_S$/dr — core slope')
    a.axhline(0, color='0.7', lw=0.5)
    a.set_xlim(1.6, r_b + 1.2)
    a.set_yscale('symlog', linthresh=1e-1)
    a.set_ylabel('dv/dr [eV/Å]')
    a.legend(loc='upper left', fontsize=9)

    # (d) u*=u basis functions on [r_lo, r_b]
    a = ax[3]
    for m in range(len(CORE_POWERS)):
        a.plot(r, phi[:, m], lw=1.2, label=f't$^{{{int(CORE_POWERS[m])}}}$')
    a.axhline(0, color='0.7', lw=0.5)
    a.set_xlim(1.6, r_b + 1.2)
    a.set_xlabel('r [Å]')
    a.set_ylabel('basis value')
    a.set_title('core basis  t=(r_b−r)/(r_b−r_lo),  φ=t^p by squaring; flat-clamped below r_lo')
    a.legend(loc='upper left', fontsize=9, ncol=5)

    m_ok = (r >= r_lo) & (r <= r_b)
    err = np.abs(v_S - v_S_fit)[m_ok]
    print(f'core fit err on [r_lo,r_b]: rms={np.sqrt(np.mean(err**2)):.3e} eV  max={err.max():.3e} eV')

    os.makedirs(OUT, exist_ok=True)
    fig_path = os.path.join(OUT, 'pme_split_1d.png')
    fig.savefig(fig_path, dpi=160, bbox_inches='tight')
    print(f'REVIEW: {fig_path}')


if __name__ == '__main__':
    main()
