// surface.cl - Surface electrostatics and molecule-substrate interaction kernels
//
// Provides multiple methods for evaluating molecule-surface interactions:
//   1. Brute-force pairwise (getSurfMorse, getSurfFlat) — sum over substrate
//      replicas with PBC, accurate but slow for large surfaces.
//   2. Folded basis expansion (getSurfFolded, getSurfFolded_workgroup,
//      getSurfFolded_harmonics) — analytic Fourier-type expansion of the
//      surface potential in a periodic basis.
//   3. Ewald 2D summation (compute_ewald_coefficients, eval_potential_vacuum,
//      eval_potential_full, eval_potential_brute) — GPU implementation of
//      2D Ewald electrostatics for charged surfaces.
//   4. Isosurface-based (getSurfaceIsoSurfMorse, getSurfaceIsoGridFF) —
//      evaluate forces on atoms near a precomputed isosurface.
//   5. Macro dipole (addDipoleField, macro_phi_rect_dipole/charge) —
//      analytic potential of polarized rectangular sheets.
//
// Kernels:
//   - getSurfMorse: Brute-force Morse + Coulomb between molecule atoms and
//     substrate atoms with PBC replicas.
//   - getSurfFolded: Folded-basis surface potential evaluation (per-atom).
//   - getSurfFolded_workgroup: Same with workgroup-shared harmonic coefficients.
//   - getSurfFolded_harmonics: Precompute folded-basis harmonic coefficients.
//   - compute_ewald_coefficients: Project charge density onto G-vectors (Ewald2D).
//   - eval_potential_vacuum: Evaluate Ewald2D potential in vacuum region.
//   - eval_potential_full: Evaluate Ewald2D potential at any z (with screening).
//   - eval_potential_brute: Brute-force Coulomb sum for validation.
//   - getSurfFlat: Simple flat-surface Morse interaction (no PBC replicas).
//   - getSurfaceIsoSurfMorse: Isosurface-based Morse force evaluation.
//   - getSurfaceIsoGridFF: Isosurface-based GridFF force evaluation.
//   - addDipoleField: Add macroscopic dipole sheet field to force grid.
//
// Helper functions: macro_phi_rect_dipole/charge (analytic rectangle potential),
// folded_eval_basis/grad (folded basis evaluation), getR4repulsion, limnitForce.
// Requires: common.cl + Forces.cl to be concatenated before this file.

inline float macro_phi_rect_dipole(float3 p, float4 Pz, float4 AB) {
    float Ax = AB.x;
    float Bx = AB.y;
    float x = p.x;
    float y = p.y;
    float z = p.z;
    float sumOmega = 0.0f;
    float sumLogY  = 0.0f;
    float sumLogX  = 0.0f;
    float xs[2] = {-Ax, Ax};
    float ys[2] = {-Bx, Bx};
    for (int ix=0; ix<2; ix++) {
        for (int iy=0; iy<2; iy++) {
            float X = x - xs[ix];
            float Y = y - ys[iy];
            float R = sqrt(X*X + Y*Y + z*z);
            float s = ((ix==0)?-1.0f:1.0f) * ((iy==0)?-1.0f:1.0f);
            sumOmega += s * atan2( X*Y, z * R + 1e-12f );
            sumLogY  += s * log( Y + R + 1e-12f );
            sumLogX  += s * log( X + R + 1e-12f );
        }
    }
    return (Pz.z * sumOmega) - (Pz.x * sumLogY) - (Pz.y * sumLogX);
}

inline float rect_sheet_F(float X, float Y, float Z){
    float R = sqrt(X*X + Y*Y + Z*Z);
    return X*log(Y + R + 1e-12f) + Y*log(X + R + 1e-12f) - Z*atan2(X*Y, Z*R + 1e-12f);
}

inline float macro_phi_rect_charge(float3 p, float4 AB){
    float Ax = AB.x;
    float By = AB.y;
    float x0 = p.x + Ax;
    float x1 = p.x - Ax;
    float y0 = p.y + By;
    float y1 = p.y - By;
    return rect_sheet_F(x0,y0,p.z) - rect_sheet_F(x1,y0,p.z) - rect_sheet_F(x0,y1,p.z) + rect_sheet_F(x1,y1,p.z);
}

inline float4 getMacroRectLayers( float3 pos, float q, float4 bounds, float4 L0, float4 L1, float4 L2, float4 S0, float4 Q0, float4 Q1, float4 Q2, int nlayer ){
    float Ax = 0.5f*(bounds.y - bounds.x);
    float By = 0.5f*(bounds.w - bounds.z);
    float cx = 0.5f*(bounds.y + bounds.x);
    float cy = 0.5f*(bounds.w + bounds.z);
    float3 p = pos - (float3)(cx,cy,0.0f);
    float phi = 0.0f;
    float4 ls[3] = {L0,L1,L2};
    float sigmas[3] = {S0.x,S0.y,S0.z};
    float4 qs[3] = {Q0,Q1,Q2};
    for(int i=0; i<nlayer; i++){
        float4 Li = ls[i];
        float3 pp = (float3)(p.x,p.y,p.z-Li.w);
        float4 AB = (float4)(Ax,By,0.0f,0.0f);
        phi += sigmas[i] * macro_phi_rect_charge( pp, AB );
        // dipole contribution
        float4 Pz = (float4)(qs[i].x, qs[i].y, qs[i].z, 0.0f);
        phi += q * macro_phi_rect_dipole( pp, Pz, AB );
    }
    // potential gradient (force) - TODO: implement gradient
    return (float4){0.0f, 0.0f, 0.0f, phi};
}

inline float folded_eval_basis(float u, float v, float z, float4 prm){
    float bx = cos( (2.0f*M_PI_F) * prm.x * u );
    float by = cos( (2.0f*M_PI_F) * prm.y * v );
    float dz = fmax(0.0f, z - prm.w);
    float bz = exp( -prm.z * dz );
    return bx * by * bz;
}

inline float3 folded_eval_grad(float u, float v, float z, float4 prm, float4 invLvec2d){
    float phix = (2.0f*M_PI_F) * prm.x;
    float phiy = (2.0f*M_PI_F) * prm.y;
    float cu = cos(phix*u);
    float su = sin(phix*u);
    float cv = cos(phiy*v);
    float sv = sin(phiy*v);
    float dz = fmax(0.0f, z - prm.w);
    float bz = exp(-prm.z * dz);
    float dEdu = -phix * su * cv * bz;
    float dEdv = -phiy * cu * sv * bz;
    float dEdz = (z > prm.w) ? (-prm.z * cu * cv * bz) : 0.0f;
    float dudx = invLvec2d.x;
    float dudy = invLvec2d.z;
    float dvdx = invLvec2d.y;
    float dvdy = invLvec2d.w;
    return (float3)( dEdu*dudx + dEdv*dvdx, dEdu*dudy + dEdv*dvdy, dEdz );
}

// limit force magnitude to fmax
float3 limnitForce( float3 f, float fmax ){
    float fr2 = dot(f,f);                         // force magnitude squared
    if( fr2>(fmax*fmax) ){ f*=(fmax/sqrt(fr2)); } // if force magnitude is larger than fmax we scale it down to fmax
    return f;
}

float4 getR4repulsion( float3 d, float R, float Rcut, float A ){
    // we use R4blob(r) = A * (1-r^2)^2
    // such that at distance r=R we have force f = fmax
    // f = -dR4blob/dr = 4*A*r*(1-r^2) = fmax
    // A = fmax/(4*R*(1-R^2))
    float R2    = R*R;
    float R2cut = Rcut*Rcut;
    float r2 = dot(d,d);
    if( r2>R2cut ){
        return (float4){0.0f,0.0f,0.0f,0.0f};
    }else if( r2>R2 ){
        float mr2 = R2cut-r2;
        float fr = A*mr2;
        return (float4){ d*(-4*fr), fr*mr2 };
    }else{
        float mr2 = R2cut-R2;
        float fr = A*mr2;
        return (float4){ d*(-4*fr), fr*mr2 };
    }
}

#ifndef MAKE_INDS_PBC_DEF
#define MAKE_INDS_PBC_DEF
inline int4 make_inds_pbc(const int n, const int iG) {
    // Generate PBC index patterns for B-spline interpolation
    // Returns 4 indices: (i0, i1, i2, i3) for 4-point B-spline
    // Handles wrapping at boundaries
    int4 inds;
    int i = iG % n;
    inds.x = (i - 1 + n) % n;
    inds.y = i;
    inds.z = (i + 1) % n;
    inds.w = (i + 2) % n;
    return inds;
}
#endif

// ============================================================
//  Brute Force Surface Interaction (getSurfMorse)
// ============================================================
// This is brute-force alternative to GridFF - describes interaction
// of molecule with substrate by pairwise interactions with multiple replicas

__kernel void getSurfMorse(
    const int4 ns,                // 1
    __global float4*  atoms,      // 2
    __global float4*  REQs,       // 3
    __global float4*  forces,     // 4
    __global float4*  atoms_s,    // 5
    __global float4*  REQ_s,      // 6
    __global float4*  surf_mpos,  // 7  (xmin,xmax,ymin,ymax)
    __global float4*  surf_mdip,  // 8  (mx,my,mz,0)
    __global float4*  surf_mQa,   // 9  Q row a
    __global float4*  surf_mQb,   // 10 Q row b
    __global float4*  surf_mQc,   // 11 (sigma0,sigma1,sigma2,Qtot)
    __global float4*  surf_qQa,   // 12 layer quadrupole (Qxx,Qxy,Qyy,z0)
    __global float4*  surf_qQb,   // 13 layer quadrupole (Qxx,Qxy,Qyy,z1)
    __global float4*  surf_qQc,   // 14 layer quadrupole (Qxx,Qxy,Qyy,z2)
    const int4     nPBC,          // 15
    const cl_Mat3  lvec,          // 16
    const float4   pos0,          // 17
    const float4   GFFParams,     // 18
    const float4   PLQH           // 19   (Pauli, London, Coulomb, HBond)
){

    __local float4 LATOMS[32];
    __local float4 LCLJS [32];

    const int nAtoms  = ns.x;

    const int iG = get_global_id  (0); // index of atom in the system
    const int iS = get_global_id  (1); // index of system
    const int iL = get_local_id   (0); // index of atom in the local memory chunk
    const int nG = get_global_size(0); // total number of atoms in the system
    const int nS = get_global_size(1); // total number of systems
    const int nL = get_local_size (0); // number of atoms in the local memory chunk

    const int natoms  = ns.x;         // number of atoms in the system
    const int nnode   = ns.y;         // number of nodes in the system
    const int nvec    = natoms+nnode; // number of vectos (atoms and pi-orbitals) in the system
    const int na_surf = ns.z;         //

    const int i0a = iS*natoms;     // index of the first atom in the system
    const int i0v = iS*nvec;       // index of the first vector (atom or pi-orbital) in the system
    const int iaa = iG + i0a;      // index of the atom in the system
    const int iav = iG + i0v;      // index of the vector (atom or pi-orbital) in the system

    float4 fe   = (float4){0.0f,0.0f,0.0f,0.0f};

    if(iG>=nAtoms) return;

    const float  K          = -GFFParams.y;
    const float  R2damp     =  GFFParams.x*GFFParams.x;
    const float3 shift_b = lvec.b.xyz + lvec.a.xyz*(nPBC.x*-2.f-1.f);      //  shift in scan(iy)
    const float3 shift_c = lvec.c.xyz + lvec.b.xyz*(nPBC.y*-2.f-1.f);      //  shift in scan(iz)
    const int bMacro      = (int)(GFFParams.z>0.5f);

    const float3 pos  = atoms[iav].xyz - pos0.xyz +  lvec.a.xyz*-nPBC.x + lvec .b.xyz*-nPBC.y + lvec.c.xyz*-nPBC.z;  // most negative PBC-cell
    const float4 REQi = REQs [iaa];

    for (int j0=0; j0<na_surf; j0+= nL ){
        const int i = j0 + iL;
        LATOMS[iL] = atoms_s[i];
        LCLJS [iL] = REQ_s  [i];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int jl=0; jl<nL; jl++){
            const int ja=jl+j0;
            if( ja<na_surf ){
                float4 REQH =       LCLJS [jl];
                float3 dp   = pos - LATOMS[jl].xyz;
                REQH.x   += REQi.x;
                REQH.yzw *= REQi.yzw;
                for(int iz=-nPBC.z; iz<=nPBC.z; iz++){
                    for(int iy=-nPBC.y; iy<=nPBC.y; iy++){
                        for(int ix=-nPBC.x; ix<=nPBC.x; ix++){
                            float4 fej = getMorsePLQH( dp, REQH, PLQH, K, R2damp );
                            fe -= fej;
                            dp   +=lvec.a.xyz;
                        }
                        dp   +=shift_b;
                    }
                    dp   +=shift_c;
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if( bMacro && (fabs(PLQH.z) > 1e-12f) && (fabs(REQi.z) > 1e-12f) ){
        int nlayer = (int)(GFFParams.w + 0.5f);
        float4 fm = getMacroRectLayers( atoms[iav].xyz, REQi.z, surf_mpos[iS], surf_mdip[iS], surf_mQa[iS], surf_mQb[iS], surf_mQc[iS], surf_qQa[iS], surf_qQb[iS], surf_qQc[iS], nlayer );
        fe.xyz += fm.xyz;
        fe.w   += fm.w;
    }

    forces[iav] += fe;
}

// ============================================================
//  Folded Basis Evaluation (getSurfFolded)
// ============================================================

__kernel void getSurfFolded(
    const int4 ns,                     // 1
    __global float4*  atoms,           // 2
    __global float4*  REQs,            // 3
    __global float4*  forces,          // 4
    __global float*   folded_coeffs,   // 5  [ntypeMax*nbasisMax]
    __global float4*  folded_kxyz,     // 6  [nbasisMax]
    __global int*     folded_atom_type,// 7  [natoms]
    const int4        folded_meta,     // 8  (nbasis, ntypes, 0, 0)
    const float4      folded_lvec2d    // 9  (ax,bx,ay,by)
){
    __local float4 LBASIS[64];
    __local float  LCOEFFS[8*64];

    const int iG = get_global_id(0);
    const int iS = get_global_id(1);
    const int iL = get_local_id(0);
    const int nL = get_local_size(0);

    const int natoms = ns.x;
    const int nnode  = ns.y;
    const int nvec   = natoms + nnode;
    const int i0a    = iS*natoms;
    const int i0v    = iS*nvec;
    const int iaa    = iG + i0a;
    const int iav    = iG + i0v;
    if(iG>=natoms) return;

    const int nbasis = folded_meta.x;
    const int ntypes = folded_meta.y;
    if(nbasis<=0) return;
    if(nbasis>64){ return; }
    if(ntypes>8 ){ return; }

    for(int j=iL; j<nbasis; j+=nL){
        LBASIS[j] = folded_kxyz[j];
    }
    for(int j=iL; j<nbasis*ntypes; j+=nL){
        LCOEFFS[j] = folded_coeffs[j];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    float ax = folded_lvec2d.x;
    float bx = folded_lvec2d.y;
    float ay = folded_lvec2d.z;
    float by = folded_lvec2d.w;
    float det = ax*by - bx*ay;
    if(fabs(det) < 1e-12f) return;
    float4 invLvec2d = (float4)( by/det, -bx/det, -ay/det, ax/det );

    float3 pos = atoms[iav].xyz;
    float u = invLvec2d.x*pos.x + invLvec2d.y*pos.y;
    float v = invLvec2d.z*pos.x + invLvec2d.w*pos.y;
    u = u - floor(u);
    v = v - floor(v);
    int ityp = folded_atom_type[iG];
    if(ityp < 0 || ityp >= ntypes) return;

    float E = 0.0f;
    float3 F = (float3)(0.0f,0.0f,0.0f);
    int ioff = ityp*nbasis;
    for(int ib=0; ib<nbasis; ib++){
        float c = LCOEFFS[ioff + ib];
        float4 prm = LBASIS[ib];
        float  b = folded_eval_basis(u, v, pos.z, prm);
        float3 g = folded_eval_grad (u, v, pos.z, prm, invLvec2d);
        E += c * b;
        F -= c * g;
    }
    forces[iav] += (float4)(F.x, F.y, F.z, -E);
}

// ============================================================
//  Folded Basis Workgroup-Optimized (getSurfFolded_workgroup)
// ============================================================

#define MAX_ATOMS 64
#define MAX_XY 4
#define MAX_Z  8

__kernel void getSurfFolded_workgroup(
    const int4 ns,                     // (natoms, nnode, 0, 0)
    __global float4*  atoms,           
    __global float4*  REQs,            
    __global float4*  forces,          
    __global float*   folded_coeffs,   
    __global float4*  folded_kxyz,     // [Nxy params, Nz params]
    __global int*     folded_atom_type,
    const int4        folded_meta,     // (N_xy, N_z, ntypes, 0) 
    const float4      folded_lvec2d    
){
    const int iG = get_global_id(0);
    const int iS = get_global_id(1);
    const int iL = get_local_id(0);    // Thread ID (0 to 63) maps to Atom index within batch
    const int nL = get_local_size(0);  // 64

    const int natoms = ns.x;
    const int Nxy = folded_meta.x; 
    const int Nz  = folded_meta.y;
    const int ntypes = folded_meta.z;
    const int nbasis_total = Nxy * Nxy * Nz;

    // ==================================================================
    // 1. ALLOCATE __LOCAL MEMORY FOR EXPLICIT PRECALCULATION STORAGE
    // ==================================================================
    // Coefficients and parameters
    __local float  LCOEFFS[MAX_XY * MAX_XY * MAX_Z * 8]; 
    __local float4 LPARAMS_XY[MAX_XY]; 
    __local float4 LPARAMS_Z[MAX_Z];

    // Evaluated 1D Basis Arrays [Atom_Index][Basis_Index]
    __local float L_BX [MAX_ATOMS][MAX_XY];
    __local float L_dBX[MAX_ATOMS][MAX_XY];
    __local float L_BY [MAX_ATOMS][MAX_XY];
    __local float L_dBY[MAX_ATOMS][MAX_XY];
    __local float L_BZ [MAX_ATOMS][MAX_Z];
    __local float L_dBZ[MAX_ATOMS][MAX_Z];

    // Cooperative parameter loading
    for(int j = iL; j < Nxy; j += nL) LPARAMS_XY[j] = folded_kxyz[j];
    for(int j = iL; j < Nz;  j += nL) LPARAMS_Z[j]  = folded_kxyz[Nxy + j];
    for(int j = iL; j < nbasis_total * ntypes; j += nL) LCOEFFS[j] = folded_coeffs[j];

    barrier(CLK_LOCAL_MEM_FENCE);

    int active = (iG < natoms);
    int ityp = active ? folded_atom_type[iG] : -1;
    active = active && (ityp >= 0) && (ityp < ntypes);

    // Geometry transforms
    float det = folded_lvec2d.x * folded_lvec2d.w - folded_lvec2d.y * folded_lvec2d.z;
    float4 invLvec = (float4)(folded_lvec2d.w/det, -folded_lvec2d.y/det, -folded_lvec2d.z/det, folded_lvec2d.x/det);

    int iav = iG + iS * (natoms + ns.y);
    float3 pos = (float3)(0.0f, 0.0f, 0.0f);
    if(active){ pos = atoms[iav].xyz; }
    
    float u = invLvec.x * pos.x + invLvec.y * pos.y;
    float v = invLvec.z * pos.x + invLvec.w * pos.y;
    u -= floor(u);
    v -= floor(v);

    // ==================================================================
    // 2. PARALLEL PRECALCULATION -> SAVE TO LOCAL MEMORY
    // Every thread calculates its own atom's basis and explicitly saves 
    // it to its dedicated row in the Local Memory array.
    // ==================================================================
    for(int i = 0; i < Nxy; i++){
        float k = LPARAMS_XY[i].x; 
        float phi = 2.0f * M_PI_F * k;
        
        float phix_u = phi * u;
        L_BX[iL][i]  = active ? native_cos(phix_u) : 0.0f;
        L_dBX[iL][i] = active ? (-phi * native_sin(phix_u)) : 0.0f;
        
        float phiy_v = phi * v;
        L_BY[iL][i]  = active ? native_cos(phiy_v) : 0.0f;
        L_dBY[iL][i] = active ? (-phi * native_sin(phiy_v)) : 0.0f;
    }

    for(int i = 0; i < Nz; i++){
        float kz = LPARAMS_Z[i].z;
        float z0 = LPARAMS_Z[i].w;
        float dz = fmax(0.0f, pos.z - z0);
        float bz = active ? native_exp(-kz * dz) : 0.0f;
        L_BZ[iL][i]  = bz;
        L_dBZ[iL][i] = active && (pos.z > z0) ? (-kz * bz) : 0.0f;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // ==================================================================
    // 3. THE TRIPLE LOOP
    // Thread streams its precalculated 1D factors from Local Memory,
    // avoiding the risk of register spilling entirely.
    // ==================================================================
    float E_tot = 0.0f;
    float dEdu_tot = 0.0f;
    float dEdv_tot = 0.0f;
    float dEdz_tot = 0.0f;

    int ic = active ? (ityp * nbasis_total) : 0; // Pointer to coefficients

    for(int iz = 0; iz < Nz; iz++){
        float bz  = L_BZ[iL][iz];
        float dbz = L_dBZ[iL][iz];

        for(int iy = 0; iy < Nxy; iy++){
            float by  = L_BY[iL][iy];
            float dby = L_dBY[iL][iy];
            
            // Outer loop multipliers
            float bz_by  = bz * by;
            float dbz_by = dbz * by;
            float bz_dby = bz * dby;

            for(int ix = 0; ix < Nxy; ix++){
                float bx  = L_BX[iL][ix];
                float dbx = L_dBX[iL][ix];

                float c = LCOEFFS[ic++]; 

                // Dynamic 3D Basis Construction
                E_tot    += c * (bx * bz_by);
                dEdu_tot += c * (dbx * bz_by);
                dEdv_tot += c * (bx * bz_dby);
                dEdz_tot += c * (bx * dbz_by);
            }
        }
    }

    // Map gradients back to forces
    float3 F_tot;
    F_tot.x = -(dEdu_tot * invLvec.x + dEdv_tot * invLvec.z);
    F_tot.y = -(dEdu_tot * invLvec.y + dEdv_tot * invLvec.w);
    F_tot.z = -dEdz_tot;

    if(active){ forces[iav] += (float4)(F_tot.x, F_tot.y, F_tot.z, -E_tot); }
}

// ============================================================
//  Folded Basis Harmonics (getSurfFolded_harmonics)
// ============================================================

__kernel void getSurfFolded_harmonics(
    const int4 ns,                     
    __global float4*  atoms,           
    __global float4*  REQs,            
    __global float4*  forces,          
    __global float*   folded_coeffs,   
    __global float4*  folded_kxyz,     // Now stores 1D params: [Nx params, Ny params, Nz params]
    __global int*     folded_atom_type,
    const int4        folded_meta,     // (Nx, Ny, Nz, ntypes)
    const float4      folded_lvec2d    
){    
    // Local memory for coefficients and 1D parameters
    __local float  LCOEFFS[MAX_XY * MAX_XY * MAX_Z * 8];
    __local float4 LBASIS[(2 * MAX_XY) + MAX_Z];

    const int iG = get_global_id(0);
    const int iS = get_global_id(1);
    const int iL = get_local_id(0);
    const int nL = get_local_size(0);
    const int natoms = ns.x;
    
    if(iG >= natoms) return;

    // Tensor product dimensions
    const int Nx = folded_meta.x;
    const int Ny = folded_meta.y;
    const int Nz = folded_meta.z;
    const int ntypes = folded_meta.w;
    const int nbasis_total = Nx * Ny * Nz;
    const int nparams_1d = Nx + Ny + Nz;

    // TODO: Complete harmonics kernel implementation
}

// ============================================================
//  Folded Basis Tensor Product Kernels — exp & poly variants
// ============================================================
//
//  One thread per atom. No private arrays.
//  folded_coeffs preloaded into local memory for performance.
//  Full-accuracy sincos/exp used once per atom, amortized by cmul.
//
//  float4 coefficients per basis function: (cPauli, cLondon, cCoulomb, cH)
//    E = B * (cCoulomb + B*(cLondon + B*cPauli))
//      = cCoulomb*B + cLondon*B^2 + cPauli*B^3
//    Coulomb decays as t^n (slowest), London as t^(2n), Pauli as t^(3n)
//    dE/dB = cCoulomb + B*(2*cLondon + B*3*cPauli)
//    c.w (H) omitted for now
//
//  Two specialized kernels:
//    getSurfFolded_tensor_exp:  iz→iy→ix (exp expensive, outermost)
//      needs folded_kxyz for per-basis alpha and z0
//    getSurfFolded_tensor_poly: ix→iy→iz (cheap tpow*=t innermost)
//      no folded_kxyz — uses scalar zmin, zcut, m_start
//      powers = m_start, m_start+1, ..., m_start+Nz-1

#ifndef FOLDED_TYPES_MAX
#define FOLDED_TYPES_MAX 8
#endif
#ifndef FOLDED_BASIS_MAX
#define FOLDED_BASIS_MAX 128
#endif

// Complex multiply helper (needed by tensor kernels)
inline float2 cmul(float2 a, float2 b) {
    return (float2)(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}

// --- Exponential variant ---
__kernel void getSurfFolded_tensor_exp(
    const int4 ns,                     // (natoms, nnode, 0, 0)
    __global float4*  atoms,
    __global float4*  REQs,
    __global float4*  forces,
    __global float4*  folded_coeffs,   // [ntypes * Nxy * Nxy * Nz] float4
    __global float4*  folded_kxyz,
    __global int*     folded_atom_type,
    const int4        folded_meta,     // (Nxy, Nz, ntypes, 0)
    const float4      folded_lvec2d,
    const float       poly_R           // unused
){
    const int iG = get_global_id(0);
    const int iS = get_global_id(1);
    if(iG >= ns.x) return;

    const int Nxy = folded_meta.x;
    const int Nz  = folded_meta.y;
    const int ntypes = folded_meta.z;
    const int nbasis_total = Nxy * Nxy * Nz;

    // Preload coefficients into local memory
    __local float4 L_coeffs[FOLDED_TYPES_MAX * FOLDED_BASIS_MAX];
    int total_coeffs = ntypes * nbasis_total;
    int lid = get_local_linear_id();
    int lsize = get_local_size(0) * get_local_size(1);
    for(int i = lid; i < total_coeffs; i += lsize){  L_coeffs[i] = folded_coeffs[i]; }
    barrier(CLK_LOCAL_MEM_FENCE);

    int ityp = folded_atom_type[iG];
    if(ityp < 0 || ityp >= ntypes) return;

    float det = folded_lvec2d.x * folded_lvec2d.w - folded_lvec2d.y * folded_lvec2d.z;
    float4 invLvec = (float4)(folded_lvec2d.w/det, -folded_lvec2d.y/det,
                              -folded_lvec2d.z/det,  folded_lvec2d.x/det);
    int iav = iG + iS * (ns.x + ns.y);
    float3 pos = atoms[iav].xyz;
    float u = invLvec.x * pos.x + invLvec.y * pos.y;
    float v = invLvec.z * pos.x + invLvec.w * pos.y;
    u -= floor(u);
    v -= floor(v);

    float cu, su = sincos(2.0f * M_PI_F * u, &cu);
    float cv, sv = sincos(2.0f * M_PI_F * v, &cv);
    float2 z1_u = (float2)(cu, su);
    float2 z1_v = (float2)(cv, sv);

    float E_tot = 0.0f, dEdu_tot = 0.0f, dEdv_tot = 0.0f, dEdz_tot = 0.0f;
    int ic = ityp * nbasis_total;

    for(int iz = 0; iz < Nz; iz++){
        float alpha = folded_kxyz[2*Nxy + iz].z;
        float z0    = folded_kxyz[2*Nxy + iz].w;
        float dz = fmax(0.0f, pos.z - z0);
        float bz = exp(-alpha * dz);
        float dbz = (pos.z > z0) ? (-alpha * bz) : 0.0f;

        float2 z_v = (float2)(1.0f, 0.0f);
        for(int iy = 0; iy < Nxy; iy++){
            float by = z_v.x;
            float dby = -2.0f * M_PI_F * (float)iy * z_v.y;
            float bz_by = bz * by, dbz_by = dbz * by, bz_dby = bz * dby;
            float2 z_u = (float2)(1.0f, 0.0f);
            for(int ix = 0; ix < Nxy; ix++){
                float bx = z_u.x;
                float dbx = -2.0f * M_PI_F * (float)ix * z_u.y;
                float B = bx * bz_by;
                float4 c = L_coeffs[ic++];
                E_tot    += B * (c.z + B*(c.y + B*c.x));
                float dE_fac = c.z + B*(2.0f*c.y + B*3.0f*c.x);
                dEdu_tot += dE_fac * (dbx * bz_by);
                dEdv_tot += dE_fac * (bx * bz_dby);
                dEdz_tot += dE_fac * (bx * dbz_by);
                z_u = cmul(z_u, z1_u);
            }
            z_v = cmul(z_v, z1_v);
        }
    }

    float3 F_tot;
    F_tot.x = -(dEdu_tot * invLvec.x + dEdv_tot * invLvec.z);
    F_tot.y = -(dEdu_tot * invLvec.y + dEdv_tot * invLvec.w);
    F_tot.z = -dEdz_tot;
    forces[iav] += (float4)(F_tot.x, F_tot.y, F_tot.z, -E_tot);
}

// --- Polynomial variant ---
// Loop order: ix→iy→iz (cheap tpow*=t innermost, expensive cmul outermost)
// Coefficient layout: coeffs[ntype][ix][iy][iz] (natural order, no transpose)
__kernel void getSurfFolded_tensor_poly(
    const int4 ns,                     // (natoms, nnode, 0, 0)
    __global float4*  atoms,
    __global float4*  REQs,
    __global float4*  forces,
    __global float4*  folded_coeffs,   // [ntypes * Nxy * Nxy * Nz] float4
    __global int*     folded_atom_type,
    const int4        folded_meta,     // (Nxy, Nz, ntypes, m_start)
    const float4      folded_lvec2d,
    const float       zmin,
    const float       zcut
){
    const int iG = get_global_id(0);
    const int iS = get_global_id(1);
    if(iG >= ns.x) return;

    const int Nxy = folded_meta.x;
    const int Nz  = folded_meta.y;
    const int ntypes = folded_meta.z;
    const int m_start = folded_meta.w;
    const int nbasis_total = Nxy * Nxy * Nz;

    // Preload coefficients into local memory
    __local float4 L_coeffs[FOLDED_TYPES_MAX * FOLDED_BASIS_MAX];
    int total_coeffs = ntypes * nbasis_total;
    int lid = get_local_linear_id();
    int lsize = get_local_size(0) * get_local_size(1);
    for(int i = lid; i < total_coeffs; i += lsize)
        L_coeffs[i] = folded_coeffs[i];
    barrier(CLK_LOCAL_MEM_FENCE);

    int ityp = folded_atom_type[iG];
    if(ityp < 0 || ityp >= ntypes) return;

    float det = folded_lvec2d.x * folded_lvec2d.w - folded_lvec2d.y * folded_lvec2d.z;
    float4 invLvec = (float4)(folded_lvec2d.w/det, -folded_lvec2d.y/det,
                              -folded_lvec2d.z/det,  folded_lvec2d.x/det);
    int iav = iG + iS * (ns.x + ns.y);
    float3 pos = atoms[iav].xyz;
    float u = invLvec.x * pos.x + invLvec.y * pos.y;
    float v = invLvec.z * pos.x + invLvec.w * pos.y;
    u -= floor(u);
    v -= floor(v);

    float cu, su = sincos(2.0f * M_PI_F * u, &cu);
    float cv, sv = sincos(2.0f * M_PI_F * v, &cv);
    float2 z1_u = (float2)(cu, su);
    float2 z1_v = (float2)(cv, sv);

    // Poly z-basis: t = 1 - min(dz/zcut, 1), powers = m_start..m_start+Nz-1
    float dz = fmax(0.0f, pos.z - zmin);
    float invR = 1.0f / zcut;
    float x = fmin(dz * invR, 1.0f);
    float t = 1.0f - x;
    bool active_z = (pos.z > zmin) && (x < 1.0f);

    // Precompute t^m_start and t^(m_start-1) for reset inside loop
    float t_m_start = 1.0f, t_m_start_prev = 1.0f;
    for(int i = 0; i < m_start; i++){ t_m_start_prev = t_m_start; t_m_start *= t; }

    float E_tot = 0.0f, dEdu_tot = 0.0f, dEdv_tot = 0.0f, dEdz_tot = 0.0f;
    int ic = ityp * nbasis_total;

    float2 z_u = (float2)(1.0f, 0.0f);
    for(int ix = 0; ix < Nxy; ix++){
        float bx = z_u.x;
        float dbx = -2.0f * M_PI_F * (float)ix * z_u.y;

        float2 z_v = (float2)(1.0f, 0.0f);
        for(int iy = 0; iy < Nxy; iy++){
            float by = z_v.x;
            float dby = -2.0f * M_PI_F * (float)iy * z_v.y;

            float tpow = t_m_start, tprev = t_m_start_prev;
            for(int iz = 0; iz < Nz; iz++){
                float n = (float)(m_start + iz);
                float bz = tpow;
                float dbz = active_z ? (-n * invR * tprev) : 0.0f;

                float B = bx * by * bz;
                float4 c = L_coeffs[ic++];
                E_tot    += B * (c.z + B*(c.y + B*c.x));
                float dE_fac = c.z + B*(2.0f*c.y + B*3.0f*c.x);
                dEdu_tot += dE_fac * (dbx * by * bz);
                dEdv_tot += dE_fac * (bx * dby * bz);
                dEdz_tot += dE_fac * (bx * by * dbz);

                tprev = tpow;
                tpow *= t;
            }
            z_v = cmul(z_v, z1_v);
        }
        z_u = cmul(z_u, z1_u);
    }

    float3 F_tot;
    F_tot.x = -(dEdu_tot * invLvec.x + dEdv_tot * invLvec.z);
    F_tot.y = -(dEdu_tot * invLvec.y + dEdv_tot * invLvec.w);
    F_tot.z = -dEdz_tot;
    forces[iav] += (float4)(F_tot.x, F_tot.y, F_tot.z, -E_tot);
}

// ============================================================
//  OpenCL Ewald2D Kernels (GPU-accelerated surface electrostatics)
// ============================================================
// Reference: pyBall/Ewald2D.py (Python implementation)
// Key optimization: Use complex multiplication to compute e^{iG·ρ}
//
// For G = h*b1 + k*b2:
//   e^{iG·ρ} = e^{i(h*b1 + k*b2)·ρ} = e^{ih*b1·ρ} * e^{ik*b2·ρ}
//
// Precompute z1_b1 = e^{i*b1·ρ}, z1_b2 = e^{i*b2·ρ}
// Then:
//   e^{ih*b1·ρ} = z1_b1^h (by repeated multiplication)
//   e^{ik*b2·ρ} = z1_b2^k
//
// This reduces N_G cos/sin evaluations to just 2 per point!

// (cmul defined earlier, before tensor kernels)

// ------------------------------------------------------------------
// Kernel 1: Compute C_G coefficients (vacuum) and w[g,i] (full)
// ------------------------------------------------------------------
// Each work item computes coefficients for one G-vector
// Work size: N_G (number of G-vectors)
__kernel void compute_ewald_coefficients(
    __global const float4* ion_data,
    __global const float4* G_data,
    __global const float2* b_vectors,
    const float area,
    const int N_ions,
    const int N_G,
    __global float2* C_G_out,
    __global float2* w_out
){
    const int ig = get_global_id(0);
    if(ig >= N_G) return;

    float4 G = G_data[ig];
    int h = (int)G.x;
    int k = (int)G.y;
    float Gn = G.z;

    float2 b1 = b_vectors[0];
    float2 b2 = b_vectors[1];

    float Gx = h * b1.x + k * b2.x;
    float Gy = h * b1.y + k * b2.y;

    float prefactor = (2.0f * M_PI_F) / (area * Gn);

    float2 C_G = (float2)(0.0f, 0.0f);

    for(int i = 0; i < N_ions; i++){
        float4 ion = ion_data[i];
        float rx = ion.x;
        float ry = ion.y;
        float rz = ion.z;
        float q = ion.w;

        float Gdotr = Gx * rx + Gy * ry;
        float cos_gr = cos(Gdotr);
        float sin_gr = sin(Gdotr);
        float2 phase = (float2)(cos_gr, -sin_gr);

        float decay_ion = exp(Gn * rz);
        float2 contrib = (float2)(q * decay_ion * phase.x, q * decay_ion * phase.y);
        C_G += contrib;

        if(w_out != NULL){
            float2 w_gi = (float2)(q * phase.x * prefactor, q * phase.y * prefactor);
            w_out[ig * N_ions + i] = w_gi;
        }
    }

    C_G_out[ig] = (float2)(C_G.x * prefactor, C_G.y * prefactor);
}

// ------------------------------------------------------------------
// Kernel 2: Vacuum potential evaluation
// ------------------------------------------------------------------
__kernel void eval_potential_vacuum(
    __global const float4* eval_points,
    __global const float2* C_G,
    __global const float4* G_data,
    __global const float2* b_vectors,
    const int N_points,
    const int N_G,
    const int n_harm,
    __global float* phi_out
){
    const int ip = get_global_id(0);
    if(ip >= N_points) return;

    float4 p = eval_points[ip];
    float x = p.x;
    float y = p.y;
    float z = p.z;

    float2 b1 = b_vectors[0];
    float2 b2 = b_vectors[1];

    float b1dotr = b1.x * x + b1.y * y;
    float b2dotr = b2.x * x + b2.y * y;
    float2 z1_b1 = (float2)(cos(b1dotr), sin(b1dotr));
    float2 z1_b2 = (float2)(cos(b2dotr), sin(b2dotr));

    float phi = 0.0f;

    for(int ig = 0; ig < N_G; ig++){
        float4 G = G_data[ig];
        int h = (int)G.x;
        int k = (int)G.y;
        float Gn = G.z;

        float2 zh_b1 = (float2)(1.0f, 0.0f);
        int h_abs = abs(h);
        for(int i = 0; i < h_abs; i++){
            zh_b1 = cmul(zh_b1, z1_b1);
        }
        if(h < 0) zh_b1.y = -zh_b1.y;

        float2 zk_b2 = (float2)(1.0f, 0.0f);
        int k_abs = abs(k);
        for(int i = 0; i < k_abs; i++){
            zk_b2 = cmul(zk_b2, z1_b2);
        }
        if(k < 0) zk_b2.y = -zk_b2.y;

        float2 phase = cmul(zh_b1, zk_b2);
        float decay = exp(-Gn * z);
        float2 C = C_G[ig];
        float2 contrib = cmul(C, phase);

        phi += contrib.x * decay;
    }

    phi_out[ip] = phi * COULOMB_CONST;
}

// ------------------------------------------------------------------
// Kernel 3: Full potential evaluation (any z)
// ------------------------------------------------------------------
__kernel void eval_potential_full(
    __global const float4* eval_points,
    __global const float2* w,
    __global const float4* ion_data,
    __global const float4* G_data,
    __global const float2* b_vectors,
    const float area,
    const int N_points,
    const int N_ions,
    const int N_G,
    __global float* phi_out
){
    const int ip = get_global_id(0);
    if(ip >= N_points) return;

    float4 p = eval_points[ip];
    float x = p.x;
    float y = p.y;
    float z = p.z;

    float2 b1 = b_vectors[0];
    float2 b2 = b_vectors[1];

    float b1dotr = b1.x * x + b1.y * y;
    float b2dotr = b2.x * x + b2.y * y;
    float2 z1_b1 = (float2)(cos(b1dotr), sin(b1dotr));
    float2 z1_b2 = (float2)(cos(b2dotr), sin(b2dotr));

    float phi0 = 0.0f;
    for(int i = 0; i < N_ions; i++){
        float4 ion = ion_data[i];
        float q = ion.w;
        float rz = ion.z;
        phi0 -= q * fabs(z - rz);
    }
    phi0 *= (2.0f * M_PI_F / area);

    float phi_G = 0.0f;

    for(int ig = 0; ig < N_G; ig++){
        float4 G = G_data[ig];
        int h = (int)G.x;
        int k = (int)G.y;
        float Gn = G.z;

        float2 zh_b1 = (float2)(1.0f, 0.0f);
        int h_abs = abs(h);
        for(int i = 0; i < h_abs; i++){
            zh_b1 = cmul(zh_b1, z1_b1);
        }
        if(h < 0) zh_b1.y = -zh_b1.y;

        float2 zk_b2 = (float2)(1.0f, 0.0f);
        int k_abs = abs(k);
        for(int i = 0; i < k_abs; i++){
            zk_b2 = cmul(zk_b2, z1_b2);
        }
        if(k < 0) zk_b2.y = -zk_b2.y;

        float2 phase = cmul(zh_b1, zk_b2);

        for(int i = 0; i < N_ions; i++){
            float4 ion = ion_data[i];
            float rz = ion.z;
            float decay = exp(-Gn * fabs(z - rz));
            float2 w_gi = w[ig * N_ions + i];
            float2 contrib = cmul(w_gi, phase);
            phi_G += contrib.x * decay;
        }
    }

    phi_out[ip] = (phi0 + phi_G) * COULOMB_CONST;
}

// ------------------------------------------------------------------
// Kernel 4: Brute force Coulomb sum (reference/validation)
// ------------------------------------------------------------------
__kernel void eval_potential_brute(
    __global const float4* eval_points,
    __global const float4* ion_data,
    __global const float2* a_vec,
    __global const float2* b_vec,
    const int N_points,
    const int N_ions,
    const int N_rep,
    __global float* phi_out
){
    const int ip = get_global_id(0);
    if(ip >= N_points) return;

    float4 p = eval_points[ip];
    float3 r = (float3)(p.x, p.y, p.z);

    float2 a = a_vec[0];
    float2 b = b_vec[0];

    float phi = 0.0f;

    for(int n = -N_rep; n <= N_rep; n++){
        for(int m = -N_rep; m <= N_rep; m++){
            if(n*n + m*m > N_rep*N_rep) continue;

            float3 R = (float3)(n*a.x + m*b.x, n*a.y + m*b.y, 0.0f);

            for(int i = 0; i < N_ions; i++){
                float4 ion = ion_data[i];
                float3 ri = (float3)(ion.x, ion.y, ion.z);
                float q = ion.w;

                float3 dr = r - (ri + R);
                float r_mag = sqrt(dr.x*dr.x + dr.y*dr.y + dr.z*dr.z);

                if(r_mag > 1e-12f){
                    phi += q / r_mag;
                }
            }
        }
    }

    phi_out[ip] = phi * COULOMB_CONST;
}

// ------------------------------------------------------------------
// Kernel 5: Finite-cluster Coulomb sum (no PBC, local-memory tiling)
//
// Computes V(r) = sum_j q_j / |r - r_j| * COULOMB_CONST for a finite
// cluster of ions (no periodic boundary conditions). Used as a
// brute-force reference for Ewald summation tests.
//
// Accumulation: two-sum (error-free transform) double-single, giving
// ~48 bits of mantissa precision in float32. This reduces accumulation
// error to ~1.3e-6, below the per-term q/r float32 error (~1.9e-6).
// The remaining ~1.5e-6 RMSE is the float32 sqrt/division floor.
//
// CAVEAT: The bounds check `if(ip >= N_points) return;` MUST NOT be
// placed before the barrier(CLK_LOCAL_MEM_FENCE). If some threads in
// a workgroup return early, they skip the barrier, causing undefined
// behavior (hang/crash/garbage) because not all threads cooperate on
// loading ion_loc. The check must be inside the loop, around the
// computation block only, so ALL threads participate in loading.
// ------------------------------------------------------------------
__kernel void eval_potential_cluster(
    __global const float4* eval_points,
    __global const float4* ion_data,
    const int N_points,
    const int N_ions,
    __global float* phi_out,
    __local float4* ion_loc
){
    const int ip = get_global_id(0);
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);

    float3 r = (float3)(0.0f, 0.0f, 0.0f);
    if(ip < N_points){
        float4 p = eval_points[ip];
        r = (float3)(p.x, p.y, p.z);
    }

    // Double-single accumulator via two-sum (error-free transform)
    // (hi, lo) together represent ~48 bits of precision
    float phi_hi = 0.0f;
    float phi_lo = 0.0f;

    for(int base = 0; base < N_ions; base += lsz){
        int j = base + lid;
        if(j < N_ions){
            ion_loc[lid] = ion_data[j];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        int imax = N_ions - base;
        if(imax > lsz) imax = lsz;

        if(ip < N_points){
            for(int i = 0; i < imax; i++){
                float4 ion = ion_loc[i];
                float3 ri = (float3)(ion.x, ion.y, ion.z);
                float q = ion.w;
                float3 dr = r - ri;
                float r_mag = sqrt(dr.x*dr.x + dr.y*dr.y + dr.z*dr.z);
                if(r_mag > 1e-12f){
                    float term = q / r_mag;
                    // Two-sum: add term to (phi_hi, phi_lo)
                    // Step 1: two_sum(phi_hi, term) -> (s, e)
                    float s = phi_hi + term;
                    float bb = s - phi_hi;
                    float e = (phi_hi - (s - bb)) + (term - bb);
                    // Step 2: add phi_lo and e, then renormalize
                    float lo = phi_lo + e;
                    phi_hi = s + lo;
                    phi_lo = lo - (phi_hi - s);
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(ip < N_points){
        phi_out[ip] = (phi_hi + phi_lo) * COULOMB_CONST;
    }
}

// ---- From relax_multi.cl: additional surface kernels ----
// ======================================================================
//                          Surface Forces
// ======================================================================

inline float4 combineREQ(float4 a, float4 b){
    return (float4)(a.x+b.x, a.y*b.y, a.z*b.z, a.w*b.w);
}

inline float getHamakerLJ93( float3 dp, float3 n, __private float3* f, float4 REQH ){
    float z = dot(dp, n);
    z = fmax(z, 1e-6f);
    float ratio = REQH.x / z;
    float r3    = ratio*ratio*ratio; // (z0/z)^3
    float r9    = r3*r3*r3;          // (z0/z)^9
    float E = 0.5f * REQH.y * ( r9 - 3.0f*r3 );
    float F_scalar = ( 4.5f * REQH.y / z ) * ( r9 - r3 );
    *f = n * F_scalar;
    return E;
}

inline float getMorseSurface( float3 dp, float3 n, __private float3* f, float4 REQH, float K ){
    float z = dot(dp, n);
    float exp_term = exp( -K * (z - REQH.x) );
    float E = REQH.y * ( exp_term*exp_term - 2.0f*exp_term );
    float F_scalar = 2.0f * K * REQH.y * exp_term * ( exp_term - 1.0f );
    *f = n * F_scalar;
    return E;
}

__kernel void getSurfFlat(
    const int4 nDOFs,               // 1   (nAtoms,nnode, nSystems, 0)
    // Dynamical
    __global float4*  apos,         // 2  [natoms]
    __global float4*  fapos,        // 3  [natoms]
    // parameters
    __global float4*  REQs,         // 4  [natoms]
    // Surface params
    const float4 surf_pos0,         // 5
    const float4 surf_normal,       // 6
    const float4 surf_REQ,          // 7
    const float4 surf_param         // 8  (K, mode, 0, 0)
){
    const int iG = get_global_id (0);   // index of atom
    const int iS = get_global_id (1);   // index of system
    const int nAtoms = nDOFs.x;
    const int nnode  = nDOFs.y;

    if(iG >= nAtoms) return;

    const int i0a   = iS*nAtoms;         // index of first atom
    const int i0v   = iS*(nAtoms+nnode); // index of first vector

    const int iav = iG + i0v;
    const int iaa = iG + i0a;

    float3 p = apos[iav].xyz;
    float4 REQi = REQs[iaa];

    float4 REQij = combineREQ( surf_REQ, REQi );

    float3 f = (float3)(0.0f);
    float E = 0.0f;

    float3 dp = p - surf_pos0.xyz;
    float3 nn = surf_normal.xyz;
    float  K  = surf_param.x;
    int mode  = (int)surf_param.y;

    if(mode == 1){ // Hamaker LJ93
        E = getHamakerLJ93( dp, nn, &f, REQij );
    } else if (mode == 2){ // Morse
        E = getMorseSurface( dp, nn, &f, REQij, K );
    }

    fapos[iav] += (float4)(f, E);
}


inline float evalSurfMorseE3D(
    const float3 pos,
    const float4 REQi,
    __global float4*  atoms_s,
    __global float4*  REQ_s,
    __global float4*  surf_mpos,
    __global float4*  surf_mdip,
    __global float4*  surf_mQa,
    __global float4*  surf_mQb,
    __global float4*  surf_mQc,
    __global float4*  surf_qQa,
    __global float4*  surf_qQb,
    __global float4*  surf_qQc,
    const int na_surf,
    const int4 nPBC,
    const cl_Mat3 lvec,
    const float4 GFFParams,
    const float4 PLQH
){
    const float  K          = -GFFParams.y;
    const float  R2damp     =  GFFParams.x*GFFParams.x;
    const float3 shift_b    = lvec.b.xyz + lvec.a.xyz*(nPBC.x*-2.f-1.f);
    const float3 shift_c    = lvec.c.xyz + lvec.b.xyz*(nPBC.y*-2.f-1.f);
    const int bMacro        = (int)(GFFParams.z>0.5f);
    const float3 pos0       = pos + lvec.a.xyz*-nPBC.x + lvec.b.xyz*-nPBC.y + lvec.c.xyz*-nPBC.z;
    float E = 0.0f;
    for(int ja=0; ja<na_surf; ja++){
        float4 REQH = REQ_s[ja];
        float3 dp   = pos0 - atoms_s[ja].xyz;
        REQH.x   += REQi.x;
        REQH.yzw *= REQi.yzw;
        for(int iz=-nPBC.z; iz<=nPBC.z; iz++){
            for(int iy=-nPBC.y; iy<=nPBC.y; iy++){
                for(int ix=-nPBC.x; ix<=nPBC.x; ix++){
                    float4 fej = getMorsePLQH(dp, REQH, PLQH, K, R2damp);
                    E -= fej.w;
                    dp += lvec.a.xyz;
                }
                dp += shift_b;
            }
            dp += shift_c;
        }
    }
    if( bMacro && (fabs(PLQH.z) > 1e-12f) && (fabs(REQi.z) > 1e-12f) ){
        int nlayer = (int)(GFFParams.w + 0.5f);
        float4 fm = getMacroRectLayers( pos, REQi.z, surf_mpos[0], surf_mdip[0], surf_mQa[0], surf_mQb[0], surf_mQc[0], surf_qQa[0], surf_qQb[0], surf_qQc[0], nlayer );
        E += fm.w;
    }
    return E;
}


__kernel void getSurfaceIsoSurfMorse(
    const int4 ns,                // 1  (1,0,na_surf,0)
    __global float4*  atoms_s,    // 2
    __global float4*  REQ_s,      // 3
    __global float4*  surf_mpos,  // 4
    __global float4*  surf_mdip,  // 5
    __global float4*  surf_mQa,   // 6
    __global float4*  surf_mQb,   // 7
    __global float4*  surf_mQc,   // 8
    __global float4*  surf_qQa,   // 9
    __global float4*  surf_qQb,   // 10
    __global float4*  surf_qQc,   // 11
    const int4        nPBC,       // 12
    const cl_Mat3     lvec,       // 13
    const float4      GFFParams,  // 14
    const float4      probe_REQ,  // 15
    const float4      sel_PLQH,   // 16
    const float4      col_PLQH,   // 17
    const int4        surf_ns,    // 18 (nx,ny,nz,mode)
    const float4      surf_p0,    // 19 (x0,y0,zmin,threshold)
    const float4      surf_step,  // 20 (dx,dy,dz,zmax)
    __global float4*  surf_xyzq,  // 21 (x,y,z,ok)
    __global float2*  surf_zc     // 22 (z_report,color)
){
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    const int nx = surf_ns.x;
    const int ny = surf_ns.y;
    const int nz = surf_ns.z;
    const int mode = surf_ns.w;
    if((ix>=nx)||(iy>=ny)) return;
    const int i = ix + iy*nx;
    const float x_in = surf_p0.x + surf_step.x*(float)ix;
    const float y_in = surf_p0.y + surf_step.y*(float)iy;
    const float ax = lvec.a.x;
    const float ay = lvec.a.y;
    const float bx = lvec.b.x;
    const float by = lvec.b.y;
    const float det = ax*by - bx*ay;
    float x = x_in;
    float y = y_in;
    if(fabs(det) > 1e-12f){
        const float inv00 =  by/det;
        const float inv01 = -bx/det;
        const float inv10 = -ay/det;
        const float inv11 =  ax/det;
        float fu = inv00*x_in + inv01*y_in;
        float fv = inv10*x_in + inv11*y_in;
        fu -= rint(fu);
        fv -= rint(fv);
        x = ax*fu + bx*fv;
        y = ay*fu + by*fv;
    }
    const float zmin = surf_p0.z;
    const float thr  = surf_p0.w;
    const float dz   = surf_step.z;
    const float zmax = surf_step.w;
    float zh = NAN;
    float ch = NAN;
    int ok = 0;
    if(mode==0){
        float z_prev = zmax;
        float e_prev = evalSurfMorseE3D((float3)(x,y,z_prev), probe_REQ, atoms_s, REQ_s, surf_mpos, surf_mdip, surf_mQa, surf_mQb, surf_mQc, surf_qQa, surf_qQb, surf_qQc, ns.z, nPBC, lvec, GFFParams, sel_PLQH);
        for(int iz=nz-2; iz>=0; iz--){
            float z_cur = zmin + dz*(float)iz;
            float e_cur = evalSurfMorseE3D((float3)(x,y,z_cur), probe_REQ, atoms_s, REQ_s, surf_mpos, surf_mdip, surf_mQa, surf_mQb, surf_mQc, surf_qQa, surf_qQb, surf_qQc, ns.z, nPBC, lvec, GFFParams, sel_PLQH);
            float s0 = e_prev - thr;
            float s1 = e_cur  - thr;
            if( isfinite(s0) && isfinite(s1) && (((s0<=0.f)&&(s1>=0.f)) || ((s0>=0.f)&&(s1<=0.f))) ){
                float dv = s1 - s0;
                float t = (fabs(dv)<1e-16f) ? 0.5f : (-s0/dv);
                t = clamp(t, 0.0f, 1.0f);
                zh = z_prev + t*(z_cur-z_prev);
                ch = evalSurfMorseE3D((float3)(x,y,zh), probe_REQ, atoms_s, REQ_s, surf_mpos, surf_mdip, surf_mQa, surf_mQb, surf_mQc, surf_qQa, surf_qQb, surf_qQc, ns.z, nPBC, lvec, GFFParams, col_PLQH);
                ok = 1;
                break;
            }
            z_prev = z_cur;
            e_prev = e_cur;
        }
    }else{
        if(nz>=3){
            float z0 = zmin;
            float z1 = zmin + dz;
            float v0 = evalSurfMorseE3D((float3)(x,y,z0), probe_REQ, atoms_s, REQ_s, surf_mpos, surf_mdip, surf_mQa, surf_mQb, surf_mQc, surf_qQa, surf_qQb, surf_qQc, ns.z, nPBC, lvec, GFFParams, sel_PLQH);
            float v1 = evalSurfMorseE3D((float3)(x,y,z1), probe_REQ, atoms_s, REQ_s, surf_mpos, surf_mdip, surf_mQa, surf_mQb, surf_mQc, surf_qQa, surf_qQb, surf_qQc, ns.z, nPBC, lvec, GFFParams, sel_PLQH);
            for(int iz=2; iz<nz; iz++){
                float z2 = zmin + dz*(float)iz;
                float v2 = evalSurfMorseE3D((float3)(x,y,z2), probe_REQ, atoms_s, REQ_s, surf_mpos, surf_mdip, surf_mQa, surf_mQb, surf_mQc, surf_qQa, surf_qQb, surf_qQc, ns.z, nPBC, lvec, GFFParams, sel_PLQH);
                if( isfinite(v0) && isfinite(v1) && isfinite(v2) && (v1<=v0) && (v1<=v2) && ((v1<v0)||(v1<v2)) ){
                    float den = (z0-z1)*(z0-z2)*(z1-z2);
                    zh = z1;
                    if(fabs(den)>=1e-16f){
                        float A = (z2*(v1-v0) + z1*(v0-v2) + z0*(v2-v1)) / den;
                        float B = (z2*z2*(v0-v1) + z1*z1*(v2-v0) + z0*z0*(v1-v2)) / den;
                        if(fabs(A)>=1e-16f){
                            float zm = -B/(2.f*A);
                            if((zm>=fmin(z0,z2)) && (zm<=fmax(z0,z2))) zh = zm;
                        }
                    }
                    ch = evalSurfMorseE3D((float3)(x,y,zh), probe_REQ, atoms_s, REQ_s, surf_mpos, surf_mdip, surf_mQa, surf_mQb, surf_mQc, surf_qQa, surf_qQb, surf_qQc, ns.z, nPBC, lvec, GFFParams, col_PLQH);
                    ok = 1;
                    break;
                }
                z0 = z1; z1 = z2; v0 = v1; v1 = v2;
            }
        }
    }
    surf_xyzq[i] = (float4)(x, y, zh, ok ? 1.0f : 0.0f);
    surf_zc [i] = (float2)(zh, ch);
}


__kernel void getSurfaceIsoGridFF(
    const int4        grid_ns,      // 1
    __global float4*  BsplinePLQ,   // 2
    const float4      grid_invStep, // 3
    const float4      grid_p0,      // 4
    const float4      sel_PLQH,     // 5
    const float4      col_PLQH,     // 6
    const int4        surf_ns,      // 7  (nx,ny,nz,mode)
    const float4      surf_p0,      // 8  (x0,y0,zmin,threshold)
    const float4      surf_step,    // 9  (dx,dy,dz,zmax)
    const float4      surf_z0,      // 10 (z_top,0,0,0)
    __global float4*  surf_xyzq,    // 11
    __global float2*  surf_zc       // 12
){
    __local int4 xqs[4];
    __local int4 yqs[4];
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    const int iLx = get_local_id(0);
    const int iLy = get_local_id(1);
    const int nx = surf_ns.x;
    const int ny = surf_ns.y;
    const int nz = surf_ns.z;
    const int mode = surf_ns.w;
    if((iLy==0) && (iLx<4)){ xqs[iLx] = make_inds_pbc(grid_ns.x, iLx); }
    if((iLx==0) && (iLy<4)){ yqs[iLy] = make_inds_pbc(grid_ns.y, iLy); }
    barrier(CLK_LOCAL_MEM_FENCE);
    if((ix>=nx)||(iy>=ny)) return;
    const int i = ix + iy*nx;
    const float x = surf_p0.x + surf_step.x*(float)ix;
    const float y = surf_p0.y + surf_step.y*(float)iy;
    const float zmin = surf_p0.z;
    const float thr  = surf_p0.w;
    const float dz   = surf_step.z;
    const float zmax = surf_step.w;
    float zh = NAN;
    float ch = NAN;
    int ok = 0;
    if(mode==0){
        float z_prev = zmax;
        const float3 u_prev = ((float3)(x,y,z_prev) - grid_p0.xyz) * grid_invStep.xyz;
        float e_prev = fe3d_pbc_comb(u_prev, grid_ns.xyz, BsplinePLQ, sel_PLQH, xqs, yqs).w;
        for(int iz=nz-2; iz>=0; iz--){
            float z_cur = zmin + dz*(float)iz;
            const float3 u_cur = ((float3)(x,y,z_cur) - grid_p0.xyz) * grid_invStep.xyz;
            float e_cur = fe3d_pbc_comb(u_cur, grid_ns.xyz, BsplinePLQ, sel_PLQH, xqs, yqs).w;
            float s0 = e_prev - thr;
            float s1 = e_cur  - thr;
            if( isfinite(s0) && isfinite(s1) && (((s0<=0.f)&&(s1>=0.f)) || ((s0>=0.f)&&(s1<=0.f))) ){
                float dv = s1 - s0;
                float t = (fabs(dv)<1e-16f) ? 0.5f : (-s0/dv);
                t = clamp(t, 0.0f, 1.0f);
                zh = z_prev + t*(z_cur-z_prev);
                ch = fe3d_pbc_comb((((float3)(x,y,zh) - grid_p0.xyz) * grid_invStep.xyz), grid_ns.xyz, BsplinePLQ, col_PLQH, xqs, yqs).w;
                ok = 1;
                break;
            }
            z_prev = z_cur;
            e_prev = e_cur;
        }
    }else{
        if(nz>=3){
            float z0 = zmin;
            float z1 = zmin + dz;
            float v0 = fe3d_pbc_comb((((float3)(x,y,z0) - grid_p0.xyz) * grid_invStep.xyz), grid_ns.xyz, BsplinePLQ, sel_PLQH, xqs, yqs).w;
            float v1 = fe3d_pbc_comb((((float3)(x,y,z1) - grid_p0.xyz) * grid_invStep.xyz), grid_ns.xyz, BsplinePLQ, sel_PLQH, xqs, yqs).w;
            for(int iz=2; iz<nz; iz++){
                float z2 = zmin + dz*(float)iz;
                float v2 = fe3d_pbc_comb((((float3)(x,y,z2) - grid_p0.xyz) * grid_invStep.xyz), grid_ns.xyz, BsplinePLQ, sel_PLQH, xqs, yqs).w;
                if( isfinite(v0) && isfinite(v1) && isfinite(v2) && (v1<=v0) && (v1<=v2) && ((v1<v0)||(v1<v2)) ){
                    float den = (z0-z1)*(z0-z2)*(z1-z2);
                    zh = z1;
                    if(fabs(den)>=1e-16f){
                        float A = (z2*(v1-v0) + z1*(v0-v2) + z0*(v2-v1)) / den;
                        float B = (z2*z2*(v0-v1) + z1*z1*(v2-v0) + z0*z0*(v1-v2)) / den;
                        if(fabs(A)>=1e-16f){
                            float zm = -B/(2.f*A);
                            if((zm>=fmin(z0,z2)) && (zm<=fmax(z0,z2))) zh = zm;
                        }
                    }
                    ch = fe3d_pbc_comb((((float3)(x,y,zh) - grid_p0.xyz) * grid_invStep.xyz), grid_ns.xyz, BsplinePLQ, col_PLQH, xqs, yqs).w;
                    ok = 1;
                    break;
                }
                z0 = z1; z1 = z2; v0 = v1; v1 = v2;
            }
        }
    }
    surf_xyzq[i] = (float4)(x, y, zh, ok ? 1.0f : 0.0f);
    surf_zc [i] = (float2)(zh - surf_z0.x, ch);
}



// ======================================================================
//                           add_DipoleField()
// ======================================================================

__attribute__((reqd_work_group_size(32,1,1)))
__kernel void addDipoleField(
    const int n,                     // 1
    __global float4*  ps,            // 2
    __global float4*  dipols,        // 3
    __write_only image3d_t  FE_Coul, // 4
    const int4     nGrid,            // 5
    const cl_Mat3  dGrid,            // 6
    const float4   grid_p0           // 7
){
    __local float4 LATOMS[32];
    __local float4 LCLJS [32];
    const int iG = get_global_id (0);
    const int nG = get_global_size(0);
    const int iL = get_local_id  (0);
    const int nL = get_local_size(0);
    const int nab = nGrid.x*nGrid.y;
    const int ia  = iG%nGrid.x;
    const int ib  = (iG%nab)/nGrid.x;
    const int ic  = iG/nab;

    const int nMax = nab*nGrid.z;
    if(iG>nMax) return;

    //if(iG==0){printf("GPU::addDipoleField(nL=%i,nG=%i,nAtoms=%i,nPBC(%i,%i,%i))\n", nL, nG, n  );}

    float3 pos     = grid_p0.xyz + dGrid.a.xyz*ia + dGrid.b.xyz*ib  + dGrid.c.xyz*ic;
    float4 fe  = float4Zero;
    for (int i0=0; i0<n; i0+= nL ){
        int i = i0 + iL;
        //if(i>=nAtoms) break;  // wrong !!!!
        LATOMS[iL] = ps    [i];
        LCLJS [iL] = dipols[i];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int j=0; j<nL; j++){
            if( (j+i0)<n ){
                float4 P     = LCLJS [j];
                float4 atom  = LATOMS[j];
                float3 d     = pos - atom.xyz;
                float  invr2 = 1.f / dot(d,d);
                float  invr  = sqrt(invr2);
                float  invr3 = invr*invr2;
                // https://en.wikipedia.org/wiki/Electric_dipole_moment#Potential_and_field_of_an_electric_dipole
                // Efield(R) = const *(    R*(Q/|R|^3) + R*3*<p|R>/|R|^5 - p/|R|^3

                float  VP  =  dot( P.xyz, d )*invr2;
                float4 fei = (float4){
                    (d*( P.w + 3*VP ) - P.xyz )*invr3,   // Force  (E-filed )
                       ( P.w +   VP           )*invr     // Energy (Potential)
                }*COULOMB_CONST;
                fe    += fei;

            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    int4 coord = (int4){ia,ib,ic,0};
    write_imagef( FE_Coul, coord, fe );
}


