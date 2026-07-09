// lcao_grid.cl - LCAO density and orbital projection onto real-space grids
// ====================================================================
//
// LCAO BASIS PROJECTION FOR GPU ELECTRONIC STRUCTURE
// ===================================================
//
// Projects atomic orbital basis functions (LCAO: Linear Combination of Atomic
// Orbitals) and density matrices onto 3D real-space grids. Supports any LCAO
// basis set (DFTB, Fireball, Siesta, etc.) with Slater-type or numerical
// orbitals. Used for electron density visualization, charge analysis, and
// as input for STM/Dyson calculations (lcao_stm.cl).
//
// --- Physics Overview ---
//
// In LCAO methods, the electron density is:
//   rho(r) = Σ_{i,j} Σ_{mu,nu} D_{mu_i, nu_j} * phi_mu(r - R_i) * phi_nu(r - R_j)
// where D is the density matrix, phi are atomic orbital basis functions,
// and R_i, R_j are atomic positions.
//
// Each basis function is a product of radial and angular parts:
//   phi_{l,m}(r) = R_l(r) * Y_l^m(r_hat)
// where Y_l^m are real spherical harmonics (tesseral harmonics).
//
// Supported angular momenta:
//   l=0 (s): 1 orbital,  Y_00 = 1/sqrt(4*pi)
//   l=1 (p): 3 orbitals, Y_1m ~ (x,y,z)/r * sqrt(3/(4*pi))
//   l=2 (d): 5 orbitals, Y_2m ~ xy, yz, 3z^2-r^2, xz, x^2-y^2
//
// Radial functions:
//   - Spline-based: Cubic B-spline interpolation from tabulated radial data
//     (Fireball/DFTB numerical basis). Stored as (wf, d2wf) pairs for
//     second-derivative interpolation.
//   - Exponential: f(r) = exp(-beta*(r - r0)) for vacuum/STM region.
//     Simpler, no tabulated basis needed, used for STM tip orbitals.
//
// --- Orbital Ordering Convention ---
//
// Two conventions are used in this codebase:
//   - Fortran (Fireball): [s, py, pz, px]  (used in density matrix rho)
//   - OpenCL:             [px, py, pz, s]  (used in MO coefficients)
// The swizzle .wyzx converts between them. See DFT/utils.py convCoefs().
//
// --- GPU Parallelization: Task-Based Block Decomposition ---
//
// The 3D grid is divided into 8x8x8 voxel blocks (512 voxels per block).
// Each block becomes a "task" handled by one workgroup:
//   1. count_atoms_per_block: Count which atoms overlap each block (sphere-AABB test)
//   2. fill_task_atoms: Fill atom indices into per-block lists (atomic_inc for slot)
//   3. compact_tasks: Remove empty blocks (prefix-sum compaction)
//   4. project_density_sparse[_tiled]: Evaluate density for each voxel in the block
//
// This approach avoids iterating over all atoms for all voxels — only atoms
// whose orbital cutoff overlaps the block are considered.
//
// --- Sparse vs Dense Density Matrix ---
//
// Sparse variant: rho[iatom][ineigh][nu][mu] — only nonzero neighbor pairs stored.
//   Requires neighbor list (neigh_j) to find the ineigh index for each (i,j) pair.
//   Used with Fireball/DFTB sparse density matrices.
//
// Dense variant: dm[mu_total][nu_total] — full norb_total × norb_total matrix.
//   No neighbor list needed. Used for small systems or when full matrix is available.
//
// --- Key Caveats ---
//
//   CAVEAT 1: The sparse density projection uses a linear search over neigh_max
//   to find the neighbor index ineigh_ij for each (i,j) pair. This is O(neigh_max)
//   per pair and can be a bottleneck for systems with many neighbors.
//
//   CAVEAT 2: The tiled variant (project_density_sparse_tiled) uses atomic_add
//   to accumulate into out_grid, which requires the grid to be zeroed first.
//   The non-tiled variant writes directly (=, not +=).
//
//   CAVEAT 3: The orbital ordering swizzle .wyzx in the tiled kernel converts
//   from OpenCL [px,py,pz,s] to Fortran [s,py,pz,px] for the rho matrix dot
//   product. Getting this wrong produces silently incorrect densities.
//
//   CAVEAT 4: The exponential radial variants (project_orbital_*_exp) have NO
//   cutoff — they evaluate at all distances. This is intentional for STM
//   where the vacuum decay is the physics of interest, but it means the
//   kernel is O(natoms) per point with no early exit.
//
// Kernels:
//   - project_density_sparse: Sparse density -> grid. 1 workgroup = 1 task block.
//   - count_atoms_per_block: Count atoms overlapping each block. 1 thread = 1 atom.
//   - fill_task_atoms: Fill atom indices into block lists. 1 thread = 1 atom.
//   - compact_tasks: Remove empty blocks via prefix sum. 1 thread = 1 block.
//   - project_density_sparse_tiled: Tiled sparse density with __local atom cache.
//   - project_orbital: Single MO -> grid. 1 workgroup = 1 task block.
//   - project_orbital_points_exp: MO at arbitrary points with exp radial. 1 thread = 1 point.
//   - mo_overlap_points_exp_sk: MO overlap with SK angular. 1 thread = 1 scan point.
//   - mo_overlap_points_exp_sk_2mol: Same, explicit two-molecule interface.
//   - project_orbital_points: MO at points with spline radial. 1 thread = 1 point.
//   - project_orbital_dense_points: Dense MO at points (s,p,d support). 1 thread = 1 point.
//   - project_orbital_dense_points_exp: Dense MO at points, exp radial. 1 thread = 1 point.
//   - project_density_dense_points: Dense density at points. 1 thread = 1 point.
//   - project_orbital_dense: Dense MO -> grid. 1 workgroup = 1 task block.
//   - project_density_dense: Dense density -> grid. 1 workgroup = 1 task block.
//
// Helper functions: evaluate_radial (cubic B-spline radial interpolation),
// eval_angular_dense (real spherical harmonics Y_lm for l=0,1,2),
// eval_atom_orbitals (full orbital evaluation at a point, s/p/d),
// sk_contract_sp (Slater-Koster overlap contraction for s-p basis),
// quat_rotate3 (quaternion rotation for tip orbital orientation).
// Self-contained (own types: GridSpec, AtomData, TaskData).

// ======================================================================
//                          Data Types
// ======================================================================
//
// GridSpec: 3D grid definition (origin, 3 lattice vectors, dimensions).
// AtomData: Per-atom info (position, cutoff radius, type index, orbital range).
// TaskData: Spatial block descriptor (block indices, atom count, j-split).
//
// Memory layout (originally Fortran column-major):
// rho(imu, inu, ineigh, iatom) -> rho[iatom][ineigh][inu][imu] in C-order
//
// OpenCL kernel for projecting sparse density matrix to a real-space grid.

#ifndef DEBUG_EARLY_EXIT
#define DEBUG_EARLY_EXIT 0
#endif

#ifndef DEBUG_CLEAR_ONLY
#define DEBUG_CLEAR_ONLY 0
#endif

#ifndef DEBUG_RETURN0
#define DEBUG_RETURN0 0
#endif

#ifndef DEBUG_READ_TASK
#define DEBUG_READ_TASK 0
#endif

#ifndef DEBUG_READ_GRID
#define DEBUG_READ_GRID 0
#endif

typedef struct {
    float4 origin;
    float4 dA;
    float4 dB;
    float4 dC;
    int4 ngrid;
} GridSpec;

typedef struct {
    float4 pos_rcut; // x, y, z, Rcut
    int type;        // index into basis data
    int i0orb;       // start index in global orbital list
    int norb;        // number of orbitals
    int pad;
} AtomData;

typedef struct {
    int x, y, z, w;  // block_idx. x,y,z are coordinates, w is padding
    int na;          // number of overlapping atoms
    int nj;          // start of jatom block ( if off-diagonal block )
    int pad1;
    int pad2;
} TaskData;

// ======================================================================
//  Real Spherical Harmonic (Tesseral Harmonic) Prefactors
// ======================================================================
//
//  Y_00 = pref_s = 1/sqrt(4*pi)
//  Y_1m = pref_p * (x,y,z)/r  = sqrt(3/(4*pi)) * r_hat
//  Y_2m = pref_d * (xy, yz, xz)  or  pref_d_z2 * (3z^2-r^2)  or  pref_d_x2y2 * (x^2-y^2)
//
//  These are the normalization constants for real (tesseral) spherical
//  harmonics, which are the angular part of atomic orbitals.
//
#define PREF_S 0.28209479f   // 1/sqrt(4*pi)
#define PREF_P 0.48860251f   // sqrt(3/(4*pi))
#define PREF_D 1.09254843f   // sqrt(15/(4*pi))
#define PREF_D_Z2 0.31539157f // sqrt(5/(16*pi)) for 3z^2-r^2
#define PREF_D_X2Y2 0.54627422f // sqrt(15/(16*pi)) for x^2-y^2

// ======================================================================
//  evaluate_radial() — Cubic B-spline interpolation of tabulated radial functions
// ======================================================================
//
//  Interpolates the radial part R_l(r) of atomic orbitals from tabulated
//  numerical basis data (e.g. Fireball pseudoatomic orbitals).
//
//  The basis_data array stores (wf, d2wf) pairs per radial node, where
//  d2wf is the second derivative precomputed for cubic spline interpolation.
//
//  Formula (matching Fortran getpsi()):
//    psi = a*ylo + b*yhi + ((a^3-a)*d2lo + (b^3-b)*d2hi) * h^2/6
//  where a = 1-t, b = t, t = fractional position in [i, i+1].
//
//  Returns 0.0 if r is beyond the tabulated range or indices are invalid.
//
float evaluate_radial(
    float r, 
    int ityp, int ish, 
    __global const float* basis_data,
    int n_nodes, float dr, int max_shells
) {
    if (ityp < 0) return 0.0f;
    if (ish < 0) return 0.0f;
    if (ish >= max_shells) return 0.0f;
    if (r >= (n_nodes - 1) * dr) return 0.0f;
    // NOTE: basis_data is packed as float2 per node: (wf, wf_spline_second_derivative)
    const __global float2* basis2_ptr = (const __global float2*)basis_data;

    const float x = r / dr;
    int i = (int)floor(x);
    if (i < 0) i = 0;
    if (i > (n_nodes - 2)) i = (n_nodes - 2);
    const float t = x - (float)i;

    // interval [i, i+1]
    const int base = (ityp * max_shells + ish) * n_nodes;
    const float2 lo = basis2_ptr[base + i];
    const float2 hi = basis2_ptr[base + i + 1];

    const float a = 1.0f - t;
    const float b = t;
    // Fortran getpsi(): psi = a*ylo + b*yhi + ((a^3-a)*d2lo + (b^3-b)*d2hi)*(h^2)/6
    const float h2_6 = (dr * dr) * (1.0f/6.0f);
    const float corr = ((a*a*a - a) * lo.y + (b*b*b - b) * hi.y) * h2_6;
    return a * lo.x + b * hi.x + corr;
}

// ============================================================================
// Dense-matrix orbital/density projection helpers with d-orbital support
// ============================================================================
//
// These helpers support arbitrary angular momenta (s, p, d) via the
// shell index = angular momentum convention (ish = l for STO basis).
//

// ======================================================================
//  eval_angular_dense() — Real spherical harmonic Y_l^m(r_hat)
// ======================================================================
//
//  Evaluates the angular part of atomic orbitals for l=0,1,2.
//  mm ranges from -l to +l, following Fortran realTessY ordering.
//
//  l=0 (s):  Y_00 = pref_s
//  l=1 (p):  Y_1,-1 = py = pref_p * y_hat,  Y_1,0 = pz = pref_p * z_hat,
//            Y_1,+1 = px = pref_p * x_hat
//  l=2 (d):  Y_2,-2 = d_xy,  Y_2,-1 = d_yz,  Y_2,0 = d_z2,
//            Y_2,+1 = d_xz,  Y_2,+2 = d_x2-y2
//
inline float eval_angular_dense(int l, int mm, float3 rhat) {
    switch(l) {
    case 0:
        return PREF_S;
    case 1:
        switch(mm) {
        case -1: return rhat.y * PREF_P;  // py
        case 0:  return rhat.z * PREF_P;  // pz
        case 1:  return rhat.x * PREF_P;  // px
        }
        break;
    case 2:
        switch(mm) {
        case -2: return rhat.x * rhat.y * PREF_D;                                    // d_xy
        case -1: return rhat.y * rhat.z * PREF_D;                                    // d_yz
        case 0:  return (2.0f*rhat.z*rhat.z - rhat.x*rhat.x - rhat.y*rhat.y) * PREF_D_Z2; // d_z2
        case 1:  return rhat.x * rhat.z * PREF_D;                                    // d_xz
        case 2:  return (rhat.x*rhat.x - rhat.y*rhat.y) * PREF_D_X2Y2;                // d_x2y2
        }
        break;
    }
    return 0.0f;
}

// ======================================================================
//  eval_atom_orbitals() — Evaluate all basis functions for one atom at a point
// ======================================================================
//
//  Computes phi_mu(r) = R_l(r) * Y_l^m(r_hat) for all orbitals of one atom.
//  Iterates over shells (ish = l = 0,1,2,...) and m = -l..+l.
//  Returns the number of orbitals written into out_buf (0 if outside cutoff).
//
//  Supports s, p, d orbitals via the shell-index = angular-momentum convention.
//  out_buf must have at least ad.norb slots (max 9 for spd).
//
inline int eval_atom_orbitals(
    float3 r_vox,
    AtomData ad,
    __global const float* basis_data,
    int n_nodes, float dr_basis, int max_shells,
    float* out_buf
) {
    float3 d = r_vox - ad.pos_rcut.xyz;
    float r2 = dot(d, d);
    float rcut = ad.pos_rcut.w;
    if (r2 > rcut*rcut) return 0;
    float r = sqrt(r2);
    float3 rhat = d / (r + 1e-12f);

    int norb = ad.norb;
    int i0orb = ad.i0orb;
    int iorb = 0;
    for (int ish = 0; ish < max_shells && iorb < norb; ++ish) {
        float R = evaluate_radial(r, ad.type, ish, basis_data, n_nodes, dr_basis, max_shells);
        int l = ish;  // shell index = angular momentum for STO basis
        int n_m = 2*l + 1;
        for (int mm = -l; mm <= l && iorb < norb; ++mm, ++iorb) {
            out_buf[iorb] = R * eval_angular_dense(l, mm, rhat);
        }
    }
    return iorb;
}

// ======================================================================
//                          project_density_sparse()
// ======================================================================
//
//  Projects sparse density matrix onto 3D grid. 1 workgroup = 1 task block
//  (8x8x8 = 512 voxels). Each thread processes multiple voxels via striding.
//
//  Physics:
//    rho(r) = Σ_{i,j} Σ_{mu,nu} D_{mu_i, nu_j} * phi_mu(r - R_i) * phi_nu(r - R_j)
//
//  The sparse density matrix is indexed as:
//    rho[iatom][ineigh][nu][imu]  (Fortran column-major -> C-order)
//  where ineigh is the neighbor index of atom j in atom i's neighbor list.
//  The neighbor list neigh_j[iatom*neigh_max + k] = j_atom + 1 (1-based).
//
//  For each voxel, the kernel:
//    1. Computes real-space position r_vox from grid indices
//    2. For each atom pair (i, j) in the task block:
//       a. Evaluates 4 orbitals (s, px, py, pz) at r_vox for each atom
//       b. Finds the neighbor index ineigh_ij via linear search
//       c. Loads the 4x4 density sub-block rho_ij
//       d. Computes den += dot(psi_i, rho_ij · psi_j) (4x4 matrix-vector)
//    3. Writes den to out_grid[g_idx]
//
//  Diagonal blocks (nj < 0): i interacts with j >= i (symmetric, pairsym=2 for off-diag).
//  Off-diagonal blocks (nj >= 0): i in [0, nj), j in [nj, na).
//
//  CAVEAT: Only s and p orbitals are supported (4 per atom). The 4x4 block
//  is hardcoded. For d-orbital support, use project_density_dense instead.
//
//  CAVEAT: The neighbor search is O(neigh_max) per pair — a linear scan.
//  For systems with large neighbor lists this is a bottleneck.
//
__kernel void project_density_sparse(
    __global const GridSpec* grid,
    const int n_tasks,
    __global const TaskData* tasks,
    __global const AtomData* atoms,
    __global const int* task_atoms,   // [n_tasks][nMaxAtom]
    __global const float* rho,        // [natoms][neigh_max][numorb_max][numorb_max]
    __global const int* neigh_j,      // [natoms][neigh_max]
    __global const float* basis_data, // [n_species][max_shells][n_nodes]
    __global const int4* species_info, // [n_species] -> (nssh, l0, l1, l2)
    const int n_nodes, 
    const float dr_basis,
    const int max_shells,
    const int neigh_max,
    const int numorb_max,
    const int nMaxAtom,
    __global float* out_grid          // [nx][ny][nz]
) {
    const int gid = get_global_id(0);
    const int threads_per_task = get_local_size(0);
    // DEBUG: emit one-line params to verify kernel entry
    // if (0 && get_global_id(0) == 0) {
    //     printf("GPU: kernel entry: n_tasks=%d vox_per_task=%d n_nodes=%d max_shells=%d neigh_max=%d numorb_max=%d ngrid=(%d,%d,%d)\n",
    //            n_tasks, 512, n_nodes, max_shells, neigh_max, numorb_max,
    //            grid->ngrid.x, grid->ngrid.y, grid->ngrid.z);
    // }
    const int i_task = get_group_id(0);
    const int t_idx  = get_local_id(0);

    //{ // sanitize local memory space
    if (i_task >= n_tasks) return;
    if (DEBUG_RETURN0) return;
    __global const int* my_atoms = task_atoms + i_task * nMaxAtom;
    const TaskData task = tasks[i_task];
    const int  na    = task.na;
    const int  nj    = task.nj;
    if (DEBUG_READ_TASK) return;
    if (DEBUG_READ_GRID) { (void)grid->ngrid.x; return; }
    //const int3 b_idx = task.block_idx.xyz;
    
    //}

    // Each thread processes 32 voxels
    for (int v = t_idx; v < 512; v += threads_per_task) {
        float3 r_vox;
        int    g_idx;
        const int lx =  v       & 7;    // v     % 8
        const int ly = (v >> 3) & 7;   // (v/8 ) % 8
        const int lz = (v >> 6) & 7;   // (v/64) % 8
        { 
            //const int lx =  v       & 7;    // v     % 8
            //const int ly = (v >> 3) & 7;   // (v/8 ) % 8
            //const int lz = (v >> 6) & 7;   // (v/64) % 8
            const int gx = task.x * 8 + lx;
            const int gy = task.y * 8 + ly;
            const int gz = task.z * 8 + lz;
            const int3 ngrid_dim = grid->ngrid.xyz;
            if (gx >= ngrid_dim.x || gy >= ngrid_dim.y || gz >= ngrid_dim.z) continue;
            g_idx = (gx * ngrid_dim.y + gy) * ngrid_dim.z + gz;
            r_vox = grid->origin.xyz + (float)gx * grid->dA.xyz + (float)gy * grid->dB.xyz + (float)gz * grid->dC.xyz;
            if(v==0) { 
            //    printf("GPU task[%3i] b_idx=(%i,%i,%i) na=%i nj=%i g_idx=%i <? nxyz=%i \n", i_task, task.x, task.y, task.z, na, nj, g_idx, ngrid_dim.x*ngrid_dim.y*ngrid_dim.z ); 
            }
        }

        // if( t_idx==0 ){ 
        //     printf("GPU task[%3i] b_idx=(%i,%i,%i) na=%i nj=%i \n", i_task, task.block_idx.x, task.block_idx.y, task.block_idx.z, na, nj); 
        // }

        if (DEBUG_CLEAR_ONLY) {
            out_grid[g_idx] = 0.0f;
            continue;
        }
        float den = 0.0f;
        if (DEBUG_EARLY_EXIT) {
            out_grid[g_idx] = 0.0f;
            continue;
        }
        // Loop over active pairs in this block
        for (int i = 0; i < na; i++) {
            
            int j_start, j_end;
            if (nj < 0) { // Diagonal block: i interacting with j >= i
                j_start = i;  j_end= na;
            } else {      // Off-diagonal block: i in [0, nj) interacting with j in [nj, na)
                if (i>=nj) break; 
                j_start=nj; j_end=na;
            }

            const int i_atom = my_atoms[i]; // <-- GLOBAL READ
            AtomData ad_i    = atoms[i_atom];  // <-- GLOBAL READ
            float    rcut_i2 = ad_i.pos_rcut.w; rcut_i2*=rcut_i2;
            float4 dri;
            dri.xyz       = r_vox - ad_i.pos_rcut.xyz;
            const float ri2 = dot(dri.xyz, dri.xyz);
            if (ri2 > rcut_i2) continue;
            const float ri = sqrt(ri2);
            dri.xyz /= (ri + 1e-12f);
            dri.w =  evaluate_radial(ri, ad_i.type, 0, basis_data, n_nodes, dr_basis, max_shells) * PREF_S;
            dri.xyz *= evaluate_radial(ri, ad_i.type, 1, basis_data, n_nodes, dr_basis, max_shells) * PREF_P;
            
            for (int j = j_start; j < j_end; j++) {
                const int j_atom = my_atoms[j]; // <-- GLOBAL READ
                // Find neighbor index ineigh_ij such that neigh_j[i_atom * neigh_max + k] == j_atom + 1
                int ineigh_ij = -1;
                for (int k = 0; k < neigh_max; k++) {
                    if (neigh_j[i_atom * neigh_max + k] == j_atom + 1) {
                        ineigh_ij = k;
                        break;
                    }
                }
                if (ineigh_ij < 0) continue;

                AtomData ad_j    = atoms[j_atom];  // <-- GLOBAL READ
                float    rcut_j2 = ad_j.pos_rcut.w; rcut_j2*=rcut_j2;
                float4 drj;
                drj.xyz = r_vox - ad_j.pos_rcut.xyz;
                const float rj2 = dot(drj.xyz, drj.xyz);

                // if( t_idx==0 ){
                //     float3 rij = ad_j.pos_rcut.xyz - ad_i.pos_rcut.xyz;
                //     int rho_base = i_atom * neigh_max * numorb_max * numorb_max + ineigh_ij * numorb_max * numorb_max;
                //     if( dot(rij,rij)<(2*rcut_i2) ) printf("GPU task[%3i] rho[%i,%i] %f \n", i_task, i, j, rho[rho_base+0] ); 
                // } 
                
                if (rj2 <= rcut_j2) {
                    //int4 sp_i = species_info[ad_i.type];
                    //int4 sp_j = species_info[ad_j.type];

                    int rho_base = i_atom * neigh_max * numorb_max * numorb_max + ineigh_ij * numorb_max * numorb_max;
                    const __global float4* rho_ij_ptr = (const __global float4*)(rho + rho_base); // <-- GLOBAL READ
                    float4 rho_ij_0 = rho_ij_ptr[0];
                    float4 rho_ij_1 = rho_ij_ptr[1];
                    float4 rho_ij_2 = rho_ij_ptr[2];
                    float4 rho_ij_3 = rho_ij_ptr[3];
                    const float rj = sqrt(rj2);
                    drj.xyz /= (rj + 1e-12f);
                    drj.w =  evaluate_radial(rj, ad_j.type, 0, basis_data, n_nodes, dr_basis, max_shells) * PREF_S;
                    drj.xyz *= evaluate_radial(rj, ad_j.type, 1, basis_data, n_nodes, dr_basis, max_shells) * PREF_P;
                    // 4x4 block (px,py,pz,s)_i * (px,py,pz,s)_j 
                    // den += dot( dri,  (
                    // rho_ij[0]  * drj.x +     // <-- GLOBAL READ
                    // rho_ij[1]  * drj.y + 
                    // rho_ij[2]  * drj.z + 
                    // rho_ij[3]  * drj.w   ) );  
                    

                    // Correct formula: Σ_αβ ρ_ij[α,β] φ_i[α] φ_j[β]
                    // Compute full 4x4 matrix multiplication
                    float4 rho_i0 = rho_ij_0;  // [ρ_sx, ρ_sy, ρ_sz, ρ_ss] or similar
                    float4 rho_i1 = rho_ij_1;
                    float4 rho_i2 = rho_ij_2;
                    float4 rho_i3 = rho_ij_3;
                    
                    // den = dri · (ρ_ij · drj)
                    // where ρ_ij is 4x4 block, dri and drj are 4-vectors
                    // den = Σ_α (Σ_β ρ_ij[α,β] * drj[β]) * dri[α]
                    
                    float4 rho_dot_drj;
                    rho_dot_drj.x = rho_i0.x * drj.x + rho_i0.y * drj.y + rho_i0.z * drj.z + rho_i0.w * drj.w;
                    rho_dot_drj.y = rho_i1.x * drj.x + rho_i1.y * drj.y + rho_i1.z * drj.z + rho_i1.w * drj.w;
                    rho_dot_drj.z = rho_i2.x * drj.x + rho_i2.y * drj.y + rho_i2.z * drj.z + rho_i2.w * drj.w;
                    rho_dot_drj.w = rho_i3.x * drj.x + rho_i3.y * drj.y + rho_i3.z * drj.z + rho_i3.w * drj.w;
                    
                    den += dot(dri.wxyz, rho_dot_drj);  
                }
            }
        }
        //if( v==0 ){  // <---works
        //if( lx == 0 ){ // <---works
        // if( ly == 0 ){ // <---works
        // //if( lz == 0 ){ // <--- crashs pyopencl._cl.LogicError: clFinish failed: INVALID_COMMAND_QUEUE
        // //if( 0  == 0 ){ // <--- crashs pyopencl._cl.LogicError: clFinish failed: INVALID_COMMAND_QUEUE
            out_grid[g_idx] = den;
        //}
    }
}

typedef struct {
    int x, y, z, w;
    int na;
    int nj;
    int pad1, pad2;
} TaskData_local;

// ======================================================================
//                          count_atoms_per_block()
// ======================================================================
//
//  Counts how many atoms overlap each spatial block. 1 thread = 1 atom.
//  Uses sphere-AABB intersection test: for each block in the atom's cutoff
//  radius, check if the atom's sphere intersects the block's bounding box.
//  Uses atomic_inc to increment the per-block counter.
//
//  This is the first pass of the task-based block decomposition:
//    count_atoms_per_block -> fill_task_atoms -> compact_tasks -> project_*
//
__kernel void count_atoms_per_block(
    __global const GridSpec* grid,
    const int natoms,
    __global const AtomData* atoms,
    const int block_res,
    const int n_blocks_x,
    const int n_blocks_y,
    const int n_blocks_z,
    __global int* block_counts
) {
    const int ia = get_global_id(0);
    if (ia >= natoms) return;

    AtomData ad = atoms[ia];
    float3 pos = ad.pos_rcut.xyz;
    float rcut = ad.pos_rcut.w;

    // Find range of blocks this atom can touch
    float3 r_min = (pos - rcut - grid->origin.xyz);
    float3 r_max = (pos + rcut - grid->origin.xyz);
    
    // Convert to block indices using floor (since origin is grid zero)
    // NOTE: dCell is dA.x, dB.y, dC.z assuming orthogonal grid for simplicity in indexing
    float3 block_size = (float)block_res * (float3)(grid->dA.x, grid->dB.y, grid->dC.z);
    int3 b0 = convert_int3(floor(r_min / block_size));
    int3 b1 = convert_int3(floor(r_max / block_size));

    b0 = clamp(b0, (int3)0, (int3)(n_blocks_x-1, n_blocks_y-1, n_blocks_z-1));
    b1 = clamp(b1, (int3)0, (int3)(n_blocks_x-1, n_blocks_y-1, n_blocks_z-1));

    for (int ix = b0.x; ix <= b1.x; ix++) {
        for (int iy = b0.y; iy <= b1.y; iy++) {
            for (int iz = b0.z; iz <= b1.z; iz++) {
                // Sphere-AABB check for each candidate block
                float3 b_min = grid->origin.xyz + (float)ix * block_res * grid->dA.xyz + (float)iy * block_res * grid->dB.xyz + (float)iz * block_res * grid->dC.xyz;
                float3 b_max = b_min + (float)block_res * (grid->dA.xyz + grid->dB.xyz + grid->dC.xyz);
                
                float3 closest_p = clamp(pos, b_min, b_max);
                float3 diff = pos - closest_p;
                if (dot(diff, diff) < rcut * rcut) {
                    int b_idx = (ix * n_blocks_y + iy) * n_blocks_z + iz;
                    atomic_inc(&block_counts[b_idx]);
                }
            }
        }
    }
}

// ======================================================================
//                          fill_task_atoms()
// ======================================================================
//
//  Fills atom indices into per-block lists. 1 thread = 1 atom.
//  Same sphere-AABB test as count_atoms_per_block, but writes the atom
//  index into task_atoms[block * nMaxAtom + slot] using atomic_inc for slot.
//  If slot >= nMaxAtom, the atom is silently dropped (overflow).
//
//  Requires block_offsets to be initialized from block_counts (prefix sum).
//
__kernel void fill_task_atoms(
    __global const GridSpec* grid,
    const int natoms,
    __global const AtomData* atoms,
    const int block_res,
    const int n_blocks_x,
    const int n_blocks_y,
    const int n_blocks_z,
    __global int* block_offsets, // used for atomic fetch-add to write atom ids
    __global int* task_atoms,    // [n_blocks][nMaxAtom]
    const int nMaxAtom
) {
    const int ia = get_global_id(0);
    if (ia >= natoms) return;

    AtomData ad = atoms[ia];
    float3 pos = ad.pos_rcut.xyz;
    float rcut = ad.pos_rcut.w;

    float3 r_min = (pos - rcut - grid->origin.xyz);
    float3 r_max = (pos + rcut - grid->origin.xyz);
    float3 block_size = (float)block_res * (float3)(grid->dA.x, grid->dB.y, grid->dC.z);
    int3 b0 = convert_int3(floor(r_min / block_size));
    int3 b1 = convert_int3(floor(r_max / block_size));
    b0 = clamp(b0, (int3)0, (int3)(n_blocks_x-1, n_blocks_y-1, n_blocks_z-1));
    b1 = clamp(b1, (int3)0, (int3)(n_blocks_x-1, n_blocks_y-1, n_blocks_z-1));

    for (int ix = b0.x; ix <= b1.x; ix++) {
        for (int iy = b0.y; iy <= b1.y; iy++) {
            for (int iz = b0.z; iz <= b1.z; iz++) {
                float3 b_min = grid->origin.xyz + (float)ix * block_res * grid->dA.xyz + (float)iy * block_res * grid->dB.xyz + (float)iz * block_res * grid->dC.xyz;
                float3 b_max = b_min + (float)block_res * (grid->dA.xyz + grid->dB.xyz + grid->dC.xyz);
                float3 closest_p = clamp(pos, b_min, b_max);
                float3 diff = pos - closest_p;
                if (dot(diff, diff) < rcut * rcut) {
                    int b_idx = (ix * n_blocks_y + iy) * n_blocks_z + iz;
                    int slot = atomic_inc(&block_offsets[b_idx]);
                    if (slot < nMaxAtom) {
                        task_atoms[b_idx * nMaxAtom + slot] = ia;
                    }
                }
            }
        }
    }
}

// ======================================================================
//                          compact_tasks()
// ======================================================================
//
//  Compacts the sparse task list by removing empty blocks. 1 thread = 1 block.
//  Uses task_offsets (prefix sum of non-empty blocks) to write compacted
//  TaskData and task_atoms arrays. Only blocks with na > 0 are kept.
//
__kernel void compact_tasks(
    const int n_blocks_x,
    const int n_blocks_y,
    const int n_blocks_z,
    __global const int* block_counts,
    __global const int* task_offsets, // prefix sum of (block_counts > 0)
    __global const int* task_atoms_raw, // [n_blocks][nMaxAtom]
    __global TaskData_local* tasks_out,
    __global int* task_atoms_out,
    const int nMaxAtom
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    const int iz = get_global_id(2);

    if (ix >= n_blocks_x || iy >= n_blocks_y || iz >= n_blocks_z) return;

    int b_idx = (ix * n_blocks_y + iy) * n_blocks_z + iz;
    int na = block_counts[b_idx];
    if (na > 0) {
        int t_idx = task_offsets[b_idx];
        TaskData_local task;
        task.x = ix; task.y = iy; task.z = iz; task.w = 0;
        task.na = na;
        task.nj = -1;
        task.pad1 = 0; task.pad2 = 0;
        tasks_out[t_idx] = task;
        
        for (int k = 0; k < nMaxAtom; k++) {
            task_atoms_out[t_idx * nMaxAtom + k] = task_atoms_raw[b_idx * nMaxAtom + k];
        }
    }
}

// ======================================================================
//                          project_density_sparse_tiled()
// ======================================================================
//
//  Tiled variant of project_density_sparse with __local memory caching.
//  1 workgroup = 1 task block. Atoms and rho blocks are loaded into
//  __local memory in TILE_ATOMS × TILE_ATOMS tiles to reduce global reads.
//
//  Key difference from non-tiled:
//    - AtomData and rho sub-blocks cached in __local (l_atom_i, l_atom_j, l_rho)
//    - Uses atomic_add to accumulate into out_grid (requires zeroing first)
//    - Ortega orbital convention swizzle: .wyzx converts [px,py,pz,s] -> [s,py,pz,px]
//    - pairsym = 2.0 for off-diagonal pairs (i != j), 1.0 for diagonal (i == j)
//
//  CAVEAT: The output grid is zeroed at the start of the kernel, then
//  accumulated with += across tiles. This is correct but means the kernel
//  cannot be run concurrently on overlapping blocks.
//
//  CAVEAT: The orbital swizzle .wyzx is critical — it converts from OpenCL
//  [px,py,pz,s] to Fortran [s,py,pz,px] order for the rho matrix. Getting
//  this wrong produces silently incorrect densities.
//
#ifndef TILE_ATOMS
#define TILE_ATOMS 8
#endif
__kernel void project_density_sparse_tiled(
    __global const GridSpec* grid,
    const int n_tasks,
    __global const TaskData* tasks,
    __global const AtomData* atoms,
    __global const int* task_atoms,   // [n_tasks][nMaxAtom]
    __global const float* rho,        // [natoms][neigh_max][numorb_max][numorb_max]
    __global const int* neigh_j,      // [natoms][neigh_max]
    __global const float* basis_data, // [n_species][max_shells][n_nodes]
    __global const int4* species_info, // [n_species] -> (nssh, l0, l1, l2)
    const int n_nodes, 
    const float dr_basis,
    const int max_shells,
    const int neigh_max,
    const int numorb_max,
    const int nMaxAtom,
    __global float* out_grid          // [nx][ny][nz]
) {
    const int i_task = get_group_id(0);
    const int t_idx  = get_local_id(0);
    const int threads_per_task = get_local_size(0);
    
    if (i_task >= n_tasks) return;
    if (DEBUG_RETURN0) return;

    TaskData task = tasks[i_task];
    if (DEBUG_READ_TASK) return;
    if (DEBUG_READ_GRID) { (void)grid->ngrid.x; return; }

    // Clean up output grid so we can accumulate into it   --- to avoid need for  l_den[512];
    for (int v = t_idx; v < 512; v += threads_per_task) {
        const int lx =  v       & 7;
        const int ly = (v >> 3) & 7;
        const int lz = (v >> 6) & 7;
        const int gx = task.x * 8 + lx;
        const int gy = task.y * 8 + ly;
        const int gz = task.z * 8 + lz;
        const int3 ngrid_dim = grid->ngrid.xyz;
        if (gx < ngrid_dim.x && gy < ngrid_dim.y && gz < ngrid_dim.z) {
            int g_idx = (gx * ngrid_dim.y + gy) * ngrid_dim.z + gz;
            out_grid[g_idx] = 0.0f;
        }
    }

    const int na = task.na;
    __local AtomData l_atom_i[TILE_ATOMS];
    __local AtomData l_atom_j[TILE_ATOMS];
    __local float4   l_rho[TILE_ATOMS*TILE_ATOMS*4]; // TILE_ATOMS x TILE_ATOMS atoms * 4 orbitals
    //__local float  l_den[512];  // Store accumulated density for each voxel in the block

    // Initialize local density buffer
    //for (int v = t_idx; v < 512; v += threads_per_task) {    l_den[v] = 0.0f;}

    barrier(CLK_LOCAL_MEM_FENCE);

    if (DEBUG_CLEAR_ONLY) return;

    if (DEBUG_EARLY_EXIT) return;

    // Tiled interaction loop: TILE_ATOMS x TILE_ATOMS atoms at a time
    for (int it = 0; it < na; it += TILE_ATOMS) {
        // Load atom_i block to local memory
        if (t_idx < TILE_ATOMS && (it + t_idx) < na) {
            int i_atom = task_atoms[i_task * nMaxAtom + it + t_idx];
            l_atom_i[t_idx] = atoms[i_atom];
        }
        
        for (int jt = 0; jt < na; jt += TILE_ATOMS) {
            if (task.nj < 0 && jt < it) continue; 

            // Load atom_j block to local memory
            if (t_idx < TILE_ATOMS && (jt + t_idx) < na) {
                int j_atom = task_atoms[i_task * nMaxAtom + jt + t_idx];
                l_atom_j[t_idx] = atoms[j_atom];
            }

            // Load rho_ij blocks for the tile
            // Each thread can help load TILE_ATOMS*TILE_ATOMS*4 float4s
            for (int k = t_idx; k < TILE_ATOMS*TILE_ATOMS*4; k += threads_per_task) {
                int pair_idx = k / 4;
                int orb_idx  = k % 4;
                int i_in_tile = pair_idx / TILE_ATOMS;
                int j_in_tile = pair_idx % TILE_ATOMS;
                
                int i = it + i_in_tile;
                int j = jt + j_in_tile;
                
                bool active = (i < na && j < na);
                if (active && task.nj < 0 && j < i) active = false;
                if (active && task.nj >= 0 && (i >= task.nj || j < task.nj)) active = false;

                if (active) {
                    int i_atom = task_atoms[i_task * nMaxAtom + i];
                    int j_atom = task_atoms[i_task * nMaxAtom + j];
                    
                    int ineigh_ij = -1;
                    for (int n = 0; n < neigh_max; n++) {
                        if (neigh_j[i_atom * neigh_max + n] == j_atom + 1) {
                            ineigh_ij = n;
                            break;
                        }
                    }
                    
                    if (ineigh_ij >= 0) {
                        int rho_base = i_atom * neigh_max * numorb_max * numorb_max + ineigh_ij * numorb_max * numorb_max;
                        float4 rho_val = ((__global float4*)(rho + rho_base))[orb_idx];
                        l_rho[k] = rho_val;
                    } else {
                        l_rho[k] = (float4)(0.0f);
                    }
                } else {
                    l_rho[k] = (float4)(0.0f);
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            // Loop over voxels in the 8x8x8 block and accumulate density from this tile
            for (int v = t_idx; v < 512; v += threads_per_task) {
                const int lx =  v       & 7;
                const int ly = (v >> 3) & 7;
                const int lz = (v >> 6) & 7;
                const int gx = task.x * 8 + lx;
                const int gy = task.y * 8 + ly;
                const int gz = task.z * 8 + lz;
                const int3 ngrid_dim = grid->ngrid.xyz;
                
                if (gx >= ngrid_dim.x || gy >= ngrid_dim.y || gz >= ngrid_dim.z) continue;
                
                float3 r_vox = grid->origin.xyz + (float)gx * grid->dA.xyz + (float)gy * grid->dB.xyz + (float)gz * grid->dC.xyz;
                float den_tile = 0.0f;

                for (int i_in_tile = 0; i_in_tile < TILE_ATOMS && (it + i_in_tile) < na; i_in_tile++) {
                    const AtomData ad_i = l_atom_i[i_in_tile];
                    const float3 dpos_i = r_vox - ad_i.pos_rcut.xyz;
                    const float  r2_i   = dot(dpos_i, dpos_i);
                    const float  rcut_i = ad_i.pos_rcut.w;
                    if (r2_i > rcut_i*rcut_i) continue;
                    const float r_i      = sqrt(r2_i);
                    const float si = evaluate_radial(r_i, ad_i.type, 0, basis_data, n_nodes, dr_basis, max_shells) * PREF_S;
                    const float pi = evaluate_radial(r_i, ad_i.type, 1, basis_data, n_nodes, dr_basis, max_shells) * PREF_P;
                    const float sc_i = (pi / (r_i + 1e-12f));
                    const float4 psi_i = (float4)(dpos_i.x*sc_i, dpos_i.y*sc_i, dpos_i.z*sc_i, si);

                    const int j_start_tile = (task.nj < 0 && jt == it) ? i_in_tile : 0;
                    for (int j_in_tile = j_start_tile; j_in_tile < TILE_ATOMS && (jt + j_in_tile) < na; j_in_tile++) {
                        const int i = it + i_in_tile;
                        const int j = jt + j_in_tile;
                        if (task.nj >= 0 && (i >= task.nj || j < task.nj)) continue;

                        const AtomData ad_j = l_atom_j[j_in_tile];
                        const float3 dpos_j = r_vox - ad_j.pos_rcut.xyz;
                        const float r2_j    = dot(dpos_j, dpos_j);
                        const float rcut_j  = ad_j.pos_rcut.w;
                        if (r2_j > rcut_j*rcut_j) continue;
                        const float r_j  = sqrt(r2_j);
                        const float sj = evaluate_radial(r_j, ad_j.type, 0, basis_data, n_nodes, dr_basis, max_shells) * PREF_S;
                        const float pj = evaluate_radial(r_j, ad_j.type, 1, basis_data, n_nodes, dr_basis, max_shells) * PREF_P;
                        const float sc_j = (pj / (r_j + 1e-12f));
                        const float4 psi_j = (float4)(dpos_j.x*sc_j, dpos_j.y*sc_j, dpos_j.z*sc_j, sj);

                        const int tile_rho_base = (i_in_tile * TILE_ATOMS + j_in_tile) * 4;
                        const float pairsym = (task_atoms[i_task * nMaxAtom + i] == task_atoms[i_task * nMaxAtom + j]) ? 1.0f : 2.0f;
                        
                        // In Ortega convention (s, py, pz, px): s-s is [0,0], s-pz is [0,2], pz-pz is [2,2]
                        den_tile += pairsym * dot( psi_i.wyzx, (
                            l_rho[tile_rho_base + 0] * psi_j.w +
                            l_rho[tile_rho_base + 1] * psi_j.y +
                            l_rho[tile_rho_base + 2] * psi_j.z +
                            l_rho[tile_rho_base + 3] * psi_j.x ) );
                    }
                }
                //l_den[v] += den_tile;

                const int g_idx = (gx * ngrid_dim.y + gy) * ngrid_dim.z + gz;
                out_grid[g_idx] += den_tile;

            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

/*
    // Final write back to global memory
    for (int v = t_idx; v < 512; v += threads_per_task) {
        const int lx =  v       & 7;
        const int ly = (v >> 3) & 7;
        const int lz = (v >> 6) & 7;
        const int gx = task.x * 8 + lx;
        const int gy = task.y * 8 + ly;
        const int gz = task.z * 8 + lz;
        const int3 ngrid_dim = grid->ngrid.xyz;

        if (gx < ngrid_dim.x && gy < ngrid_dim.y && gz < ngrid_dim.z) {
            int g_idx = (gx * ngrid_dim.y + gy) * ngrid_dim.z + gz;
            out_grid[g_idx] = l_den[v];
        }
    }
*/
}

// ======================================================================
//                          project_orbital()
// ======================================================================
//
//  Projects a single molecular orbital onto 3D grid. 1 workgroup = 1 task.
//  Computes psi(r) = Σ_i C_i · phi_i(r) where C_i are MO coefficients.
//
//  Simpler than density projection: no pairwise sum, just a linear
//  combination of basis functions evaluated at each voxel.
//
//  IMPORTANT: Coeffs must be in OpenCL order [px, py, pz, s] (remapped
//  from Fortran [s, py, pz, px] by convCoefs() in DFT/utils.py).
//
//  Only s and p orbitals supported (4 per atom, hardcoded as float4).
//  For d-orbital support, use project_orbital_dense instead.
//
__kernel void project_orbital(
    __global const GridSpec* grid,
    const int n_tasks,
    __global const TaskData* tasks,
    __global const AtomData* atoms,
    __global const int* task_atoms,
    __global const float* coeffs,     // [natoms][numorb_max] - MO coefficients in OpenCL order [s, px, py, pz]
    __global const float* basis_data,
    const int n_nodes,
    const float dr_basis,
    const int max_shells,
    const int numorb_max,
    const int nMaxAtom,
    __global float* out_grid
) {
    const int gid = get_global_id(0);
    const int threads_per_task = get_local_size(0);
    const int i_task = get_group_id(0);
    const int t_idx = get_local_id(0);
    if (i_task >= n_tasks) return;

    const TaskData task = tasks[i_task];
    const int na = task.na;

    for (int v = t_idx; v < 512; v += threads_per_task) {
        float3 r_vox;
        int g_idx;
        const int lx = v & 7;
        const int ly = (v >> 3) & 7;
        const int lz = (v >> 6) & 7;
        {
            const int gx = task.x * 8 + lx;
            const int gy = task.y * 8 + ly;
            const int gz = task.z * 8 + lz;
            const int3 ngrid_dim = grid->ngrid.xyz;
            if (gx >= ngrid_dim.x || gy >= ngrid_dim.y || gz >= ngrid_dim.z) continue;
            g_idx = (gx * ngrid_dim.y + gy) * ngrid_dim.z + gz;
            r_vox = grid->origin.xyz + (float)gx * grid->dA.xyz + (float)gy * grid->dB.xyz + (float)gz * grid->dC.xyz;
        }

        float psi = 0.0f;

        for (int i = 0; i < na; i++) {
            const int i_atom = task_atoms[i_task * nMaxAtom + i];
            AtomData ad_i = atoms[i_atom];
            float rcut_i2 = ad_i.pos_rcut.w;
            rcut_i2 *= rcut_i2;

            float3 dri;
            dri = r_vox - ad_i.pos_rcut.xyz;
            const float ri2 = dot(dri, dri);
            if (ri2 > rcut_i2) continue;
            const float ri = sqrt(ri2);

            // Evaluate radial parts
            const float rs = evaluate_radial(ri, ad_i.type, 0, basis_data, n_nodes, dr_basis, max_shells);
            const float rp = evaluate_radial(ri, ad_i.type, 1, basis_data, n_nodes, dr_basis, max_shells);
            
            // Angular unit vector (x/r, y/r, z/r) for p-orbital angular dependence
            const float3 rhat = dri / (ri + 1e-12f);

            // Sum over orbitals: ψ += C_i * φ_i(r)
            // Coeffs are in order [px, py, pz, s] matching existing FireballOCL convention
            // See DFT/utils.py convCoefs() which produces [px, py, pz, s] order
            const int coeff_base = i_atom * numorb_max;
            const int norb = ad_i.norb;

            // Compute basis function values: [px*rhat.x, py*rhat.y, pz*rhat.z, s]
            // This matches sp3_tex() in myprog.cl: return (float4)(dp*ir*pref_p, pref_s)*fr
            const float4 basis_val = (float4)(
                rp * rhat.x * PREF_P,   // px component
                rp * rhat.y * PREF_P,   // py component
                rp * rhat.z * PREF_P,   // pz component
                rs * PREF_S             // s component
            );

            // Dot product with coefficients [px, py, pz, s]
            const __global float4* coeffs_ptr = (const __global float4*)(coeffs);
            float4 coeff_val = coeffs_ptr[coeff_base / 4];
            psi += dot(coeff_val, basis_val);
    }

        out_grid[g_idx] = psi;
    }
}

// ======================================================================
//                          project_orbital_points_exp()
// ======================================================================
//
//  Projects a single MO at arbitrary points with exponential radial decay.
//  1 thread = 1 point. No grid structure needed — evaluates at any 3D point.
//
//  Radial part: f(r) = exp(-beta*(r - r0)) instead of spline basis tables.
//  This is used for STM vacuum region where orbitals decay exponentially.
//
//  CAVEAT: No distance cutoff — all atoms within rcut are evaluated.
//  The cutoff is from AtomData.pos_rcut.w, not from the exponential itself.
//
__kernel void project_orbital_points_exp(
    const int n_points,
    __global const float4* points,    // [n_points] xyz
    __global const AtomData* atoms,   // [natoms]
    const int natoms,
    __global const float* coeffs,     // [natoms*4] packed as float4
    const float beta,
    const float r0,
    __global float* out_psi           // [n_points]
) {
    const int ip = get_global_id(0);
    if (ip >= n_points) return;

    const float3 p = points[ip].xyz;
    float psi = 0.0f;

    for (int ia = 0; ia < natoms; ia++) {
        const AtomData ad = atoms[ia];
        float3 d = p - ad.pos_rcut.xyz;
        const float r2 = dot(d, d);
        const float rcut2 = ad.pos_rcut.w * ad.pos_rcut.w;
        if (r2 > rcut2) continue;
        const float r = sqrt(r2);

        const float invr = 1.0f / (r + 1e-12f);
        const float3 rhat = d * invr;

        const float f = exp(-beta * (r - r0));
        const float rs = f;
        const float rp = f;

        const float4 basis_val = (float4)(
            rp * rhat.x * PREF_P,
            rp * rhat.y * PREF_P,
            rp * rhat.z * PREF_P,
            rs * PREF_S
        );

        const int coeff_base = ad.i0orb;
        const __global float4* coeffs_ptr = (const __global float4*)(coeffs);
        float4 coeff_val = coeffs_ptr[coeff_base / 4];
        psi += dot(coeff_val, basis_val);
    }

    out_psi[ip] = psi;
}

// ======================================================================
//  sk_contract_sp() — Slater-Koster overlap contraction for s-p basis
// ======================================================================
//
//  Computes the 4×4 overlap matrix element between two atoms' orbitals
//  (s, px, py, pz) using the Slater-Koster two-center approximation:
//
//    t = cT^T · S(l,m,n) · cS
//
//  where (l,m,n) is the unit vector along the interatomic axis, and
//  S contains the SK hopping integrals:
//    t_ss = sT * Vss * sS
//    t_sp = sT * Vsp * (l*pxS + m*pyS + n*pzS)
//    t_ps = (l*pxT + m*pyT + n*pzT) * Vps * sS
//    t_pp = Vpp_pi * (pT·pS) + (Vpp_sig - Vpp_pi) * (pT·u) * (pS·u)
//
//  The last term decomposes the p-p interaction into sigma (along axis)
//  and pi (perpendicular) components via the projection (p·u).
//
//  Coeff convention: float4 [px, py, pz, s] per atom (OpenCL order).
//
// ============================================================================
//  MO overlap for molecular tip vs molecular sample
// ============================================================================
//
//  Each work-item computes the overlap amplitude for one tip-center position:
//    t(R_tip) = Σ_{ia∈tip} Σ_{ja∈smp} cT(ia)^T · S_ij(R_ij) · cS(ja)
//
//  The tip can be rotated per-pixel via a quaternion (tip_quat), which is
//  applied to both the tip atom positions and the p-orbital coefficients.
//
//  Output: out_t = signed amplitude, out_I = |t|^2 (intensity).
//
inline float sk_contract_sp(
    const float4 cT_pxpy_pz_s,
    const float4 cS_pxpy_pz_s,
    const float l, const float m, const float n,
    const float Vss, const float Vsp, const float Vps,
    const float Vpp_sig, const float Vpp_pi
){
    // reorder [px,py,pz,s] -> (s,px,py,pz)
    const float sT  = cT_pxpy_pz_s.w;
    const float pxT = cT_pxpy_pz_s.x;
    const float pyT = cT_pxpy_pz_s.y;
    const float pzT = cT_pxpy_pz_s.z;
    const float sS  = cS_pxpy_pz_s.w;
    const float pxS = cS_pxpy_pz_s.x;
    const float pyS = cS_pxpy_pz_s.y;
    const float pzS = cS_pxpy_pz_s.z;

    const float pT_dot_pS = pxT*pxS + pyT*pyS + pzT*pzS;
    const float pT_dot_u  = pxT*l  + pyT*m  + pzT*n;
    const float pS_dot_u  = pxS*l  + pyS*m  + pzS*n;

    const float t_ss = sT * Vss * sS;
    const float t_sp = sT * Vsp * (l*pxS + m*pyS + n*pzS);
    const float t_ps = (l*pxT + m*pyT + n*pzT) * Vps * sS;
    const float d = Vpp_sig - Vpp_pi;
    const float t_pp = Vpp_pi * pT_dot_pS + d * pT_dot_u * pS_dot_u;
    return t_ss + t_sp + t_ps + t_pp;
}

// ======================================================================
//  quat_rotate3() — Rotate vector v by unit quaternion q = (x,y,z,w)
// ======================================================================
//
//  Uses the optimized quaternion rotation formula:
//    v' = v + 2*w*(qv × v) + 2*qv × (qv × v)
//  where qv = (q.x, q.y, q.z). This is equivalent to the standard
//  rotation matrix but requires fewer operations (no trig functions).
//
//  Used to rotate tip atom positions and p-orbital coefficients for
//  arbitrary tip orientations in STM scan kernels.
//
inline float3 quat_rotate3( const float4 q, const float3 v ){
    // q = (x,y,z,w), unit quaternion; rotate v by q
    const float3 qv = (float3)(q.x,q.y,q.z);
    const float3 t  = 2.0f*cross(qv, v);
    return v + q.w*t + cross(qv, t);
}

// ======================================================================
//                          mo_overlap_points_exp_sk()
// ======================================================================
//
//  Computes MO overlap amplitude between a molecular tip and sample at each
//  scan point. 1 thread = 1 scan point (pixel).
//
//  Physics:
//    t(R_tip) = Σ_{ia∈tip} Σ_{ja∈smp} cT(ia)^T · S_SK(R_ij) · cS(ja)
//    I(R_tip) = |t|^2
//
//  The tip is positioned at tip_centers[ip] with optional rotation tip_quat[ip].
//  Tip atom positions are relative to the tip center and rotated by the quaternion.
//  The p-orbital coefficients are also rotated to match the tip orientation.
//
//  SK parameters use a simplified exponential model:
//    Vss = Vsp = Vps = Vpp_sig = -f,  Vpp_pi = +f
//  where f = exp(-beta*(r - r0)). The sign convention ensures correct
//  bonding/antibonding phase for sigma and pi interactions.
//
__kernel void mo_overlap_points_exp_sk(
    // Scan grid: each work-item computes overlap for one tip-center position
    const int n_points,                       // number of scan points (pixels)
    __global const float4* tip_centers,      // [n_points] tip-center positions (x,y,z,NA) in Å; : These are the lateral shifts of the rigid tip relative to sample
    __global const float4* tip_quat,         // [n_points] tip rotation quaternion (x,y,z,w), unit; : Applied per pixel/work-item (can vary across the scan)
    __global const float4* tip_pos_rel,      // [ntip_atoms] tip atom positions relative to tip center (x,y,z,NA) in Å; :  This geometry is already rotated on the host side (if rotation is desired)
    __global const float4* smp_pos,          // [nsmp_atoms] sample atom absolute positions (x,y,z, NA) in Å
    const int ntip_atoms,                    // number of tip atoms
    const int nsmp_atoms,                    // number of sample atoms
    __global const float4* coeffs_tip,       // [ntip_atoms] MO coefficients for tip, per atom as float4 [px,py,pz,s]: Order: .x=px, .y=py, .z=pz, .w=s (cartesian p orbitals)
    __global const float4* coeffs_smp,       // [nsmp_atoms] MO coefficients for sample, per atom as float4 [px,py,pz,s]
    // Exponential radial decay parameters: f(r) = exp(-beta*(r - r0))
    const float beta,                        // decay constant (Å^-1)
    const float r0,                          // reference distance (Å) where f=1
    const float rcut,                        // distance cutoff (Å); atom pairs beyond rcut are skipped
    __global float* out_t,                   // [n_points] output signed overlap amplitude t = c_tip^T S_ts c_sample
    __global float* out_I                    // [n_points] output intensity |t|^2
){
    const int ip = get_global_id(0);
    if(ip >= n_points) return;
    const float3 cen = tip_centers[ip].xyz;
    const float4 q   = tip_quat[ip];
    const float rcut2 = rcut*rcut;
    float t = 0.0f;

    for(int ia=0; ia<ntip_atoms; ia++){
        const float3 pT = cen + quat_rotate3(q, tip_pos_rel[ia].xyz);
        const float4 cT0 = coeffs_tip[ia];
        const float3 pcoefT = quat_rotate3(q, (float3)(cT0.x,cT0.y,cT0.z));
        const float4 cT = (float4)(pcoefT.x,pcoefT.y,pcoefT.z,cT0.w);
        for(int ja=0; ja<nsmp_atoms; ja++){
            const float3 d = pT - smp_pos[ja].xyz;
            const float r2 = dot(d,d);
            if(r2 > rcut2) continue;
            const float r = sqrt(r2);
            const float invr = 1.0f/(r + 1e-12f);
            const float l = d.x*invr;
            const float m = d.y*invr;
            const float n = d.z*invr;
            const float f = exp(-beta*(r - r0));
            // Pure exp radial + SK angular (no extra per-channel amplitudes):
            // Use fixed sign convention to keep p-p sigma/pi anisotropy.
            const float Vss = -f;
            const float Vsp = -f;
            const float Vps = -f;
            const float Vpp_sig = -f;
            const float Vpp_pi  = +f;
            t += sk_contract_sp(cT, coeffs_smp[ja], l,m,n, Vss, Vsp, Vps, Vpp_sig, Vpp_pi);
        }
    }
    out_t[ip] = t;
    out_I[ip] = t*t;
}

// ======================================================================
//                          mo_overlap_points_exp_sk_2mol()
// ======================================================================
//
//  Same as mo_overlap_points_exp_sk but with explicit two-molecule interface.
//  Implementation is identical — separate entry point for self-documenting
//  call sites when tip and sample are different molecules.
//
__kernel void mo_overlap_points_exp_sk_2mol(
    // Explicit two-molecule entrypoint (tip and sample may be different molecules)
    // NOTE: Implementation is identical to mo_overlap_points_exp_sk; we keep it separate
    //       to avoid breaking existing workflows and to make call sites self-documenting.
    const int n_points,
    __global const float4* tip_centers,
    __global const float4* tip_quat,
    __global const float4* tip_pos_rel,
    __global const float4* smp_pos,
    const int ntip_atoms,
    const int nsmp_atoms,
    __global const float4* coeffs_tip,
    __global const float4* coeffs_smp,
    const float beta,
    const float r0,
    const float rcut,
    __global float* out_t,
    __global float* out_I
){
    const int ip = get_global_id(0);
    if(ip >= n_points) return;
    const float3 cen = tip_centers[ip].xyz;
    const float4 q   = tip_quat[ip];
    const float rcut2 = rcut*rcut;
    float t = 0.0f;

    for(int ia=0; ia<ntip_atoms; ia++){
        const float3 pT = cen + quat_rotate3(q, tip_pos_rel[ia].xyz);
        const float4 cT0 = coeffs_tip[ia];
        const float3 pcoefT = quat_rotate3(q, (float3)(cT0.x,cT0.y,cT0.z));
        const float4 cT = (float4)(pcoefT.x,pcoefT.y,pcoefT.z,cT0.w);
        for(int ja=0; ja<nsmp_atoms; ja++){
            const float3 d = pT - smp_pos[ja].xyz;
            const float r2 = dot(d,d);
            if(r2 > rcut2) continue;
            const float r = sqrt(r2);
            const float invr = 1.0f/(r + 1e-12f);
            const float l = d.x*invr;
            const float m = d.y*invr;
            const float n = d.z*invr;
            const float f = exp(-beta*(r - r0));
            const float Vss = -f;
            const float Vsp = -f;
            const float Vps = -f;
            const float Vpp_sig = -f;
            const float Vpp_pi  = +f;
            t += sk_contract_sp(cT, coeffs_smp[ja], l,m,n, Vss, Vsp, Vps, Vpp_sig, Vpp_pi);
        }
    }
    out_t[ip] = t;
    out_I[ip] = t*t;
}
// ======================================================================
//                          project_orbital_points()
// ======================================================================
//
//  Projects a single MO at arbitrary points using spline-based radial functions.
//  1 thread = 1 point. Same as project_orbital but for arbitrary points
//  (not on a regular grid). Uses evaluate_radial() for the radial part.
//
//  Coeff convention: [px, py, pz, s] per atom (float4).
//  basis_data: packed as float2 per node (wf, wf_spline second derivative).
//
__kernel void project_orbital_points(
    const int n_points,
    __global const float4* points,    // [n_points] xyz
    __global const AtomData* atoms,   // [natoms]
    const int natoms,
    __global const float* coeffs,     // [natoms*4] packed as float4
    __global const float* basis_data,
    const int n_nodes,
    const float dr_basis,
    const int max_shells,
    __global float* out_psi           // [n_points]
) {
    const int ip = get_global_id(0);
    if (ip >= n_points) return;

    const float3 p = points[ip].xyz;
    float psi = 0.0f;

    for (int ia = 0; ia < natoms; ia++) {
        const AtomData ad = atoms[ia];
        float3 d = p - ad.pos_rcut.xyz;
        const float r2 = dot(d, d);
        const float rcut2 = ad.pos_rcut.w * ad.pos_rcut.w;
        if (r2 > rcut2) continue;
        const float r = sqrt(r2);

        const float rs = evaluate_radial(r, ad.type, 0, basis_data, n_nodes, dr_basis, max_shells);
        const float rp = evaluate_radial(r, ad.type, 1, basis_data, n_nodes, dr_basis, max_shells);
        const float invr = 1.0f / (r + 1e-12f);
        const float3 rhat = d * invr;

        const float4 basis_val = (float4)(
            rp * rhat.x * PREF_P,
            rp * rhat.y * PREF_P,
            rp * rhat.z * PREF_P,
            rs * PREF_S
        );

        const int coeff_base = ia * 4;
        const __global float4* coeffs_ptr = (const __global float4*)(coeffs);
        const float4 c = coeffs_ptr[coeff_base / 4];
        psi += dot(c, basis_val);
    }

    out_psi[ip] = psi;
}



// ======================================================================
//                          project_orbital_dense_points()
// ======================================================================
//
//  Projects a single MO at arbitrary points using dense coefficient vector.
//  1 thread = 1 point. Supports s, p, d orbitals via eval_atom_orbitals().
//
//  Unlike project_orbital_points, this uses a dense coefficient vector
//  indexed by i0orb + orbital_index, allowing arbitrary numbers of orbitals
//  per atom (not limited to 4). Uses spline-based radial functions.
//
__kernel void project_orbital_dense_points(
    const int n_points,
    __global const float4* points,
    __global const AtomData* atoms,
    const int natoms,
    __global const float* coeffs,       // [norb_total] dense coefficient vector
    __global const float* basis_data,
    const int n_nodes,
    const float dr_basis,
    const int max_shells,
    __global float* out_psi
) {
    const int ip = get_global_id(0);
    if (ip >= n_points) return;

    const float3 p = points[ip].xyz;
    float psi = 0.0f;

    for (int ia = 0; ia < natoms; ++ia) {
        AtomData ad = atoms[ia];
        float3 d = p - ad.pos_rcut.xyz;
        float r2 = dot(d, d);
        float rcut2 = ad.pos_rcut.w * ad.pos_rcut.w;
        if (r2 > rcut2) continue;

        float r = sqrt(r2);
        float3 rhat = d / (r + 1e-12f);

        int i0 = ad.i0orb;
        int norb = ad.norb;
        int iorb = 0;
        for (int ish = 0; ish < max_shells && iorb < norb; ++ish) {
            float R = evaluate_radial(r, ad.type, ish, basis_data, n_nodes, dr_basis, max_shells);
            int l = ish;
            for (int mm = -l; mm <= l && iorb < norb; ++mm, ++iorb) {
                float ang = eval_angular_dense(l, mm, rhat);
                psi += coeffs[i0 + iorb] * R * ang;
            }
        }
    }

    out_psi[ip] = psi;
}

// ======================================================================
//                          project_orbital_dense_points_exp()
// ======================================================================
//
//  Dense MO projection at points with exponential radial decay.
//  1 thread = 1 point. Supports s, p, d orbitals.
//
//  Uses f(r) = exp(-beta*(r - r0)) for the radial part — no tabulated
//  basis needed. NO CUTOFF: all atoms are evaluated regardless of distance.
//  This is intentional for STM simulation where vacuum decay is the physics.
//
//  CAVEAT: O(natoms) per point with no early exit. For large systems,
//  call with smaller point batches or use a cutoff-based variant.
//
__kernel void project_orbital_dense_points_exp(
    const int n_points,
    __global const float4* points,
    __global const AtomData* atoms,
    const int natoms,
    __global const float* coeffs,       // [norb_total] dense coefficient vector
    const float beta,                   // exponential decay constant (Å^-1), must be > 0
    const float r0,                     // reference distance (Å) where f=1
    const int max_shells,
    __global float* out_psi
) {
    const int ip = get_global_id(0);
    if (ip >= n_points) return;

    const float3 p = points[ip].xyz;
    float psi = 0.0f;

    for (int ia = 0; ia < natoms; ++ia) {
        AtomData ad = atoms[ia];
        float3 d = p - ad.pos_rcut.xyz;
        float r = sqrt(dot(d, d));
        float3 rhat = d / (r + 1e-12f);

        // Exponential radial decay: f(r) = exp(-beta*(r - r0))
        // beta > 0 ensures decaying function
        float R = exp(-beta * (r - r0));

        int i0 = ad.i0orb;
        int norb = ad.norb;
        int iorb = 0;
        for (int ish = 0; ish < max_shells && iorb < norb; ++ish) {
            int l = ish;
            for (int mm = -l; mm <= l && iorb < norb; ++mm, ++iorb) {
                float ang = eval_angular_dense(l, mm, rhat);
                psi += coeffs[i0 + iorb] * R * ang;
            }
        }
    }

    out_psi[ip] = psi;
}

// ======================================================================
//                          project_density_dense_points()
// ======================================================================
//
//  Projects dense density matrix at arbitrary points. 1 thread = 1 point.
//  Computes rho(r) = Σ_{i,j} Σ_{mu,nu} D_{mu_i,nu_j} * phi_mu(r) * phi_nu(r)
//
//  Supports full density matrix (dm is norb_total × norb_total) with
//  s, p, d orbitals via eval_atom_orbitals(). Uses pairsym=2 for i!=j
//  to account for the symmetric density matrix.
//
//  CAVEAT: O(natoms^2) per point. The inner loop evaluates all atom pairs.
//  For large systems, use the sparse variant or the grid-based approach.
//
//  CAVEAT: Private arrays phi_i[9], phi_j[9] limit to 9 orbitals per atom
//  (1 s + 3 p + 5 d = 9). For f-orbitals (l=3), this would overflow.
//
__kernel void project_density_dense_points(
    const int n_points,
    __global const float4* points,
    __global const AtomData* atoms,
    const int natoms,
    __global const float* dm,          // dense density matrix [norb_total*norb_total]
    const int norb_total,
    __global const float* basis_data,
    const int n_nodes,
    const float dr_basis,
    const int max_shells,
    __global float* out_rho
) {
    const int ip = get_global_id(0);
    if (ip >= n_points) return;

    const float3 p = points[ip].xyz;

    // Evaluate all orbitals at this point into private arrays
    // Max orbitals per atom = 9 (spd), max atoms in a typical task = 64
    // We use a fixed-size buffer for simplicity; for very large systems,
    // this kernel should be called with smaller point batches.
    float phi_i[9];  // max orbitals per atom (s=1, p=3, d=5)
    float phi_j[9];

    float density = 0.0f;

    for (int ia = 0; ia < natoms; ++ia) {
        AtomData ad_i = atoms[ia];
        int i0_i = ad_i.i0orb;
        int norb_i = eval_atom_orbitals(p, ad_i, basis_data, n_nodes, dr_basis, max_shells, phi_i);
        if (norb_i == 0) continue;

        for (int ja = ia; ja < natoms; ++ja) {
            AtomData ad_j = atoms[ja];
            int i0_j = ad_j.i0orb;
            int norb_j = eval_atom_orbitals(p, ad_j, basis_data, n_nodes, dr_basis, max_shells, phi_j);
            if (norb_j == 0) continue;

            float pairsym = (ia == ja) ? 1.0f : 2.0f;

            for (int mu = 0; mu < norb_i; ++mu) {
                for (int nu = 0; nu < norb_j; ++nu) {
                    float rho_munu = dm[(i0_i + mu) * norb_total + (i0_j + nu)];
                    density += pairsym * rho_munu * phi_i[mu] * phi_j[nu];
                }
            }
        }
    }

    out_rho[ip] = density;
}

// ======================================================================
//                          project_orbital_dense()
// ======================================================================
//
//  Projects a single MO onto 3D grid using dense coefficient vector.
//  1 workgroup = 1 task block (8x8x8 voxels). Supports s, p, d orbitals.
//  Same execution model as project_orbital but with dense coefficients
//  and full angular momentum support via eval_angular_dense().
//
__kernel void project_orbital_dense(
    __global const GridSpec* grid,
    const int n_tasks,
    __global const TaskData* tasks,
    __global const AtomData* atoms,
    __global const int* task_atoms,
    __global const float* coeffs,       // [norb_total] dense coefficient vector
    __global const float* basis_data,
    const int n_nodes,
    const float dr_basis,
    const int max_shells,
    const int nMaxAtom,
    __global float* out_grid
) {
    const int i_task = get_group_id(0);
    const int t_idx  = get_local_id(0);
    const int threads_per_task = get_local_size(0);

    if (i_task >= n_tasks) return;

    const TaskData task = tasks[i_task];
    const int na = task.na;

    for (int v = t_idx; v < 512; v += threads_per_task) {
        const int lx = v & 7;
        const int ly = (v >> 3) & 7;
        const int lz = (v >> 6) & 7;
        const int gx = task.x * 8 + lx;
        const int gy = task.y * 8 + ly;
        const int gz = task.z * 8 + lz;
        const int3 ngrid_dim = grid->ngrid.xyz;
        if (gx >= ngrid_dim.x || gy >= ngrid_dim.y || gz >= ngrid_dim.z) continue;
        const int g_idx = (gx * ngrid_dim.y + gy) * ngrid_dim.z + gz;
        float3 r_vox = grid->origin.xyz + (float)gx * grid->dA.xyz + (float)gy * grid->dB.xyz + (float)gz * grid->dC.xyz;

        float psi = 0.0f;
        for (int i = 0; i < na; ++i) {
            const int i_atom = task_atoms[i_task * nMaxAtom + i];
            AtomData ad = atoms[i_atom];
            float3 d = r_vox - ad.pos_rcut.xyz;
            float r2 = dot(d, d);
            float rcut2 = ad.pos_rcut.w * ad.pos_rcut.w;
            if (r2 > rcut2) continue;
            float r = sqrt(r2);
            float3 rhat = d / (r + 1e-12f);

            int i0 = ad.i0orb;
            int norb = ad.norb;
            int iorb = 0;
            for (int ish = 0; ish < max_shells && iorb < norb; ++ish) {
                float R = evaluate_radial(r, ad.type, ish, basis_data, n_nodes, dr_basis, max_shells);
                int l = ish;
                for (int mm = -l; mm <= l && iorb < norb; ++mm, ++iorb) {
                    float ang = eval_angular_dense(l, mm, rhat);
                    psi += coeffs[i0 + iorb] * R * ang;
                }
            }
        }
        out_grid[g_idx] = psi;
    }
}

// ======================================================================
//                          project_density_dense()
// ======================================================================
//
//  Projects dense density matrix onto 3D grid. 1 workgroup = 1 task block.
//  Computes rho(r) = Σ_{i,j} Σ_{mu,nu} D_{mu_i,nu_j} * phi_mu(r) * phi_nu(r)
//  with full s, p, d support via eval_atom_orbitals().
//
//  Same execution model as project_density_sparse but with dense dm matrix.
//  No neighbor list needed — all pairs within the task block are evaluated.
//  Uses pairsym=2 for off-diagonal atom pairs.
//
__kernel void project_density_dense(
    __global const GridSpec* grid,
    const int n_tasks,
    __global const TaskData* tasks,
    __global const AtomData* atoms,
    __global const int* task_atoms,
    __global const float* dm,          // [norb_total*norb_total] dense density matrix
    const int norb_total,
    __global const float* basis_data,
    const int n_nodes,
    const float dr_basis,
    const int max_shells,
    const int nMaxAtom,
    __global float* out_grid
) {
    const int i_task = get_group_id(0);
    const int t_idx  = get_local_id(0);
    const int threads_per_task = get_local_size(0);

    if (i_task >= n_tasks) return;

    const TaskData task = tasks[i_task];
    const int na = task.na;

    for (int v = t_idx; v < 512; v += threads_per_task) {
        const int lx = v & 7;
        const int ly = (v >> 3) & 7;
        const int lz = (v >> 6) & 7;
        const int gx = task.x * 8 + lx;
        const int gy = task.y * 8 + ly;
        const int gz = task.z * 8 + lz;
        const int3 ngrid_dim = grid->ngrid.xyz;
        if (gx >= ngrid_dim.x || gy >= ngrid_dim.y || gz >= ngrid_dim.z) continue;
        const int g_idx = (gx * ngrid_dim.y + gy) * ngrid_dim.z + gz;
        float3 r_vox = grid->origin.xyz + (float)gx * grid->dA.xyz + (float)gy * grid->dB.xyz + (float)gz * grid->dC.xyz;

        float phi_i[9];  // max orbitals per atom (s=1, p=3, d=5)
        float phi_j[9];
        float density = 0.0f;

        for (int i = 0; i < na; ++i) {
            const int i_atom = task_atoms[i_task * nMaxAtom + i];
            AtomData ad_i = atoms[i_atom];
            int i0_i = ad_i.i0orb;
            int norb_i = eval_atom_orbitals(r_vox, ad_i, basis_data, n_nodes, dr_basis, max_shells, phi_i);
            if (norb_i == 0) continue;

            for (int j = i; j < na; ++j) {
                const int j_atom = task_atoms[i_task * nMaxAtom + j];
                AtomData ad_j = atoms[j_atom];
                int i0_j = ad_j.i0orb;
                int norb_j = eval_atom_orbitals(r_vox, ad_j, basis_data, n_nodes, dr_basis, max_shells, phi_j);
                if (norb_j == 0) continue;

                float pairsym = (i_atom == j_atom) ? 1.0f : 2.0f;
                for (int mu = 0; mu < norb_i; ++mu) {
                    for (int nu = 0; nu < norb_j; ++nu) {
                        float rho_munu = dm[(i0_i + mu) * norb_total + (i0_j + nu)];
                        density += pairsym * rho_munu * phi_i[mu] * phi_j[nu];
                    }
                }
            }
        }
        out_grid[g_idx] = density;
    }
}






// ============================================================================
// STM Response Amplitude — exponential-basis coupling, single s-tip orbital
// ============================================================================
// Precompute on CPU:  G0 = inv((E+iη)S_s - H_s),  v = C^T G0
// GPU kernel builds a_st = (E+iη)S_ts - H_ts per grid point and computes:
//   resp = |v·a_st^H|^2 / |(E+iη-E_tip) - a_st·G0·a_st^H|^2
//
// Buffers:
//   points      [n_points]   float4 xyz (tip positions)
//   atoms_s     [natoms_s]   AtomData for sample atoms
//   starts_s    [natoms_s+1] int   orbital offsets
//   v_re, v_im  [ns]         float  precomputed v = C^T G0
//   G0_re, G0_im[ns*ns]      float  precomputed sample Green's function
//   params: E_re, E_im, E_tip, beta, r0, A_ss, A_sp, rcut
//
// out_resp      [n_points]   float  response amplitude
// ----------------------------------------------------------------------------
