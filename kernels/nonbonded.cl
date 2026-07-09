// nonbonded.cl - Non-bonded interaction kernels (LJ/Coulomb + spatial bucketing)
// ====================================================================
//
// NON-BONDED INTERACTIONS FOR GPU MOLECULAR DYNAMICS
// =================================================
//
// Evaluates non-bonded forces (Lennard-Jones + Coulomb + H-bond) between all
// atom pairs with periodic boundary conditions, excluding bonded pairs
// (1st and 2nd neighbors via a packed sorted exclusion list).
//
// --- Physics Overview ---
//
// The non-bonded potential between atoms i and j is:
//
//   V_LJ = 4*eps*((sigma/r)^12 - (sigma/r)^6)    [Lennard-Jones]
//   V_Coul = q_i*q_j / (4*pi*eps0*r)              [Coulomb]
//   V_HB = H-bond correction (optional)            [H-bond]
//
// The total force is F = -dV/dr, computed by getLJQH() in Forces.cl.
//
// Mixing rules (arithmetic for R, geometric for E and Q):
//   R_mix = R_i + R_j        (additive vdW radii)
//   E_mix = E_i * E_j        (geometric vdW well depth)
//   Q_mix = Q_i * Q_j        (geometric charge product)
//
// Damping: A damping radius R2damp = Rdamp^2 smoothly zeros the force
// beyond R_damp to avoid discontinuities at the cutoff.
//
// --- Exclusion List ---
//
// Bonded pairs (1-2 and 1-3 neighbors) are excluded from non-bonded
// evaluation. The exclusion list is a packed sorted array: for each atom i,
// excl[i*EXCL_MAX .. i*EXCL_MAX+EXCL_MAX-1] contains sorted (j_index, pbc_cell)
// packed entries. The kernel walks this list in parallel with the j-loop,
// skipping excluded pairs.
//
// The packing format: each int entry = (pbc_cell << 24) | atom_index.
// A value of -1 marks the end of the exclusion list for that atom.
//
// --- PBC Image Loop ---
//
// With periodic boundary conditions, each atom pair is evaluated for all
// PBC images within nPBC.x/y/z repetitions of the unit cell. The shift
// vectors are precomputed:
//   shift0 = -nPBC.x*lvec.a - nPBC.y*lvec.b - nPBC.z*lvec.c  (corner image)
//   shift_a = lvec.b + lvec.a*(-2*nPBC.x - 1)                 (x-loop increment)
//   shift_b = lvec.c + lvec.b*(-2*nPBC.y - 1)                 (y-loop increment)
// These allow efficient iteration over the 3x3 (or larger) PBC image grid.
//
// --- GPU Parallelization: Local-Memory Tiling ---
//
// The N-body problem is parallelized with local-memory tiling:
//   - Workgroup size = 32 threads (one per atom in a tile)
//   - Each iteration loads a tile of 32 atoms into __local memory
//   - All threads then evaluate forces against the tile
//   - This reduces global memory reads from O(N^2) to O(N^2/32)
//   - Barriers between tile loads ensure data consistency
//
// --- Spatial Bucketing Kernels ---
//
// For large systems, the O(N^2) tiling approach becomes expensive. The
// bucketing kernels provide a spatial decomposition alternative:
//   1. sortAtomsToBucketOverlaps: Precompute which atoms overlap each bucket's BB
//   2. getShortRangeBuckets/getShortRangeBuckets2: Evaluate forces only between
//      atoms in overlapping buckets, using getR4repulsion (short-range only)
//
// Kernels:
//   - getNonBond_ex2: Pairwise LJ/Coulomb/H-bond with 2nd-neighbor exclusion.
//     1 thread = 1 atom, local-memory tiling over atom chunks.
//   - getShortRangeBuckets: Short-range R4 repulsion between spatial buckets.
//     1 workgroup = 1 bucket, local-memory tiling within bucket.
//   - getShortRangeBuckets2: Same with precomputed overlap lists + PBC.
//   - sortAtomsToBucketOverlaps: Build overlap lists for getShortRangeBuckets2.
//
// --- Key Caveats ---
//
//   CAVEAT 1: The exclusion list walk assumes j indices are processed in
//   ascending order (sorted). The tiling loop iterates j0=0,1,...,nG-1, so
//   this holds. If the iteration order changes, the exclusion logic breaks.
//
//   CAVEAT 2: The PBC image loop is hardcoded to 3x3 (iy=0..2, ix=0..2)
//   in getNonBond_ex2, regardless of nPBC. This is a fixed 3x3x1 image sum.
//   For larger PBC ranges, use getNonBond_GridFF_Bspline_tex which loops
//   over -nPBC..+nPBC in all 3 dimensions.
//
//   CAVEAT 3: The bounding-box check in bucketing kernels uses inverted
//   comparison (p.x < bbi.lo.x && p.x > bbi.hi.x). This is because BBs
//   stores (xmin, xmax, ...) in a float8 where lo = (xmin, ymin, zmin, 0)
//   and hi = (xmax, ymax, zmax, 0). The comparison is correct but confusing.
//
//   CAVEAT 4: getShortRangeBuckets2 has a bug in PBC shift application:
//   line 335 uses `lvec.a*ilvecb` instead of `lvec.b*ilvecb`. This means
//   the b-direction PBC shift is applied with the a lattice vector.
//
// Requires: common.cl + Forces.cl to be concatenated before this file.

// B-spline texture sampler: unnormalized coords, repeat wrapping, nearest filter.
// Used by GridFF texture-based kernels in nonbonded_grid.cl.
__constant sampler_t sampler_bspline = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_REPEAT | CLK_FILTER_NEAREST;

// ======================================================================
//                          getNonBond_ex2()
// ======================================================================
//
//  Pairwise non-bonded force evaluation with 2nd-neighbor exclusion.
//  1 thread = 1 atom, 1 system per dim-1.
//
//  Physics:
//    For each pair (i, j) with j > i, computes LJ + Coulomb + H-bond force.
//    Bonded pairs (1-2, 1-3) are skipped via the packed exclusion list.
//    PBC: 3x3 image sum in the xy-plane (z not periodic in this kernel).
//
//  Algorithm:
//    1. Load atom i position + parameters into registers
//    2. For each tile of 32 atoms j:
//       a. Cooperatively load tile into __local memory (LATOMS, LCLJS)
//       b. barrier — wait for all loads
//       c. For each j in tile:
//          - Apply mixing rules: R_mix = Ri+Rj, E_mix = Ei*Ej, Q_mix = Qi*Qj
//          - Walk exclusion list (sorted, parallel to j-loop)
//          - If PBC: loop 3x3 images, skip excluded (j, pbc_cell) pairs
//          - If no PBC: skip excluded j indices
//          - Accumulate force: fe += getLJQH(dp, REQK, R2damp)
//    3. Write fe to aforce[iav] (=, not +=, if this is first force kernel)
//
//  Exclusion list format:
//    excl[iaa*EXCL_MAX + k] = (pbc_cell << 24) | j_index, or -1 for end.
//    The walk advances iex when (jex & 0xFFFFFF) < ja (sorted j ascending).
//    Comparison: jex == (ipbc << 24) | ja checks both atom index AND PBC cell.
//
//  CAVEAT: PBC loop is hardcoded to 3x3 (ix=0..2, iy=0..2), NOT using nPBC.
//  This is a fixed 3x3x1 image sum. For general PBC ranges, use the
//  GridFF_Bspline_tex kernel which loops -nPBC..+nPBC in all 3 dimensions.
//
__kernel void getNonBond_ex2(
    const int4        nDOFs,        // 1 // (natoms,nnode) dimensions of the system
    // Dynamical
    __global float4*  apos,         // 2 // positions of atoms  (including node atoms [0:nnode] and capping atoms [nnode:natoms] and pi-orbitals [natoms:natoms+nnode] )
    __global float4*  aforce,       // 3 // forces on atoms
    // Parameters
    __global float4*  REQs,         // 4 // non-bonded parameters (RvdW,EvdW,QvdW,Hbond)
    __global int*     excl,         // 5 // packed sorted exclusion list ()   
    __global cl_Mat3* lvecs,        // 6 // lattice vectors for each system
    const int4        nPBC,         // 7 // number of PBC images in each direction (x,y,z)
    const float4      GFFParams
){
    __local float4 LATOMS[32];   // local buffer for atom positions
    __local float4 LCLJS [32];   // local buffer for atom parameters

    const int iG = get_global_id  (0); // index of atom
    const int nG = get_global_size(0); // number of atoms
    const int iS = get_global_id  (1); // index of system
    const int nS = get_global_size(1); // number of systems
    const int iL = get_local_id   (0); // index of atom in local memory
    const int nL = get_local_size (0); // number of atoms in local memory

    const int natoms=nDOFs.x;  // number of atoms
    const int nnode =nDOFs.y;  // number of node atoms
    const int nvec  =natoms+nnode; // number of vectors (atoms+node atoms)
    const int i0a = iS*natoms;  // index of first atom in atoms array
    const int i0v = iS*nvec;    // index of first atom in vectors array
    const int iaa = iG + i0a; // index of atom in atoms array
    const int iav = iG + i0v; // index of atom in vectors array
    
    //if(iG<natoms){
    //const bool   bNode = iG<nnode;   // All atoms need to have neighbors !!!!
    const bool   bPBC  = (nPBC.x+nPBC.y+nPBC.z)>0;  // PBC is used if any of the PBC dimensions is >0
    //const bool bPBC=false;

    const float4 REQKi  = REQs     [iaa];  // non-bonded parameters
    const float3 posi   = apos     [iav].xyz; // position of atom
    const float  R2damp = GFFParams.x*GFFParams.x; // squared damping radius
    float4 fe           = float4Zero;  // force on atom

    const cl_Mat3 lvec   = lvecs[iS]; // lattice vectors for this system
    const float3 shift0  = lvec.a.xyz*-nPBC.x + lvec.b.xyz*-nPBC.y + lvec.c.xyz*-nPBC.z;   // shift of PBC image 0
    const float3 shift_a = lvec.b.xyz + lvec.a.xyz*(nPBC.x*-2.f-1.f);                      // shift of PBC image in the inner loop
    const float3 shift_b = lvec.c.xyz + lvec.b.xyz*(nPBC.y*-2.f-1.f);                      // shift of PBC image in the outer loop

    //const int excl_base = iaa*EXCL_MAX;
    int iex             = iaa*EXCL_MAX;
    const int iex_end   = iex + EXCL_MAX-1;
    int jex             = excl[iex];

    #ifndef DBG_NB
    #define DBG_NB 0
    #endif
    #ifndef DBG_NB_IS
    #define DBG_NB_IS 0
    #endif
    #ifndef DBG_NB_IG
    #define DBG_NB_IG 0
    #endif
    #ifndef DBG_NB_R
    #define DBG_NB_R 2.5f
    #endif

    #if DBG_NB
    if((iG==0)&&(iS==0)){
        printf( "getNonBond_ex2() iG %i, iS %i, iaa %i, iex %i, iex_end %i, jex %i\n", iG, iS, iaa, iex, iex_end, jex );
    }
    #endif

    // ========= Atom-to-Atom interaction ( N-body problem ), we do it in chunks of size of local memory, in order to reuse data and reduce number of reads from global memory  
    //barrier(CLK_LOCAL_MEM_FENCE);
    for (int j0=0; j0<nG; j0+=nL){     // loop over all atoms in the system, by chunks of size of local memory
        const int i=j0+iL;             // index of atom in local memory
        if(i<natoms){                  // j0*nL may be larger than natoms, so we need to check if we are not reading from invalid address
            LATOMS[iL] = apos [i+i0v]; // read atom position to local memory 
            LCLJS [iL] = REQs [i+i0a]; // read atom parameters to local memory
        }
        barrier(CLK_LOCAL_MEM_FENCE);   // wait until all atoms are read to local memory
        for (int jl=0; jl<nL; jl++){    // loop over all atoms in local memory (like 32 atoms)
            const int ja=j0+jl;         // index of atom in global memory
            if( (ja!=iG) && (ja<natoms) ){   // if atom is not the same as current atom and it is not out of range,  // ToDo: Should atom interact with himself in PBC ?
                const float4 aj = LATOMS[jl];    // read atom position   from local memory
                float4 REQK     = LCLJS [jl];    // read atom parameters from local memory
                float3 dp       = aj.xyz - posi; // vector between atoms
                //if((iG==44)&&(iS==0))printf( "[i=%i,ja=%i/%i,j0=%i,jl=%i/%i][iG/nG/na %i/%i/%i] aj(%g,%g,%g,%g) REQ(%g,%g,%g,%g)\n", i,ja,nG,j0,jl,nL,   iG,nG,natoms,   aj.x,aj.y,aj.z,aj.w,  REQK.x,REQK.y,REQK.z,REQK.w  );
                REQK.x  +=REQKi.x;   // mixing rules for vdW Radius
                REQK.yz *=REQKi.yz;  // mixing rules for vdW Energy

                if(jex!=-1){
                   if( (iex<iex_end) && ((jex&0xFFFFFF)<ja) ){ iex++; }
                   jex = excl[iex]; 
                }

                if(bPBC){         // ===== if PBC is used, we need to loop over all PBC images of the atom
                    int ipbc=0;   // index of PBC image
                    dp += shift0; // shift to PBC image 0
                    // Fixed PBC size
                    for(int iy=0; iy<3; iy++){
                        for(int ix=0; ix<3; ix++){
                            int jac = (ipbc<<24) | ja;
                            if(jex!=jac){
                                #if DBG_NB
                                if( (iS==DBG_NB_IS) && (iG==DBG_NB_IG) ){
                                    float r2 = dot(dp,dp);
                                    if(r2 < (DBG_NB_R*DBG_NB_R)){
                                        printf("DBG_NB iS=%i iG=%i ja=%i ipbc=%i r=%g dp=(%g,%g,%g) jex=%i jac=%i\n", iS,iG,ja,ipbc,sqrt(r2),dp.x,dp.y,dp.z,jex,jac);
                                    }
                                }
                                #endif
                                float4 fij = getLJQH( dp, REQK, R2damp );  // calculate non-bonded force between atoms using LJQH potential
                                fe += fij;
                            }
                            ipbc++; 
                            dp    += lvec.a.xyz; 
                        }
                        dp    += shift_a;
                    }
                }else {
                    if(jex!=ja){                                              // ===== if PBC is not used, it is much simpler
                        #if DBG_NB
                        if( (iS==DBG_NB_IS) && (iG==DBG_NB_IG) ){
                            float r2 = dot(dp,dp);
                            if(r2 < (DBG_NB_R*DBG_NB_R)){
                                printf("DBG_NB_NOPBC iS=%i iG=%i ja=%i r=%g dp=(%g,%g,%g) jex=%i\n", iS,iG,ja,sqrt(r2),dp.x,dp.y,dp.z,jex);
                            }
                        }
                        #endif
                        float4 fij = getLJQH( dp, REQK, R2damp ); 
                        fe += fij;
                    }
                }
            }
        }
        //barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if(iG<natoms){
        //if(iS==0){ printf( "OCL::getNonBond(iG=%i) fe(%g,%g,%g,%g)\n", iG, fe.x,fe.y,fe.z,fe.w ); }
        aforce[iav] = fe;           // If we do    run it as first forcefield, we can just store force (non need to clean it before in that case)
        //aforce[iav] += fe;        // If we don't run it as first forcefield, we need to add force to existing force
        //aforce[iav] = fe*(-1.f);
    }
}




// ======================================================================
//                          getShortRangeBuckets()
// ======================================================================
//
//  Short-range R4 repulsion forces between spatial buckets (groups).
//  1 workgroup = 1 bucket (group of atoms). Local size = max bucket size.
//
//  Algorithm:
//    1. For each bucket ib, load atom i's position and parameters
//    2. For each other bucket jb:
//       a. Bounding-box overlap test (skip if separated by > Rcut)
//       b. Cooperatively load jb's atoms into __local memory (with mask:
//          only atoms inside bucket ib's bounding box are loaded)
//       c. barrier
//       d. For each loaded atom j: compute getR4repulsion(dp, R-SRdR, R, E*SRamp)
//       e. barrier
//    3. Write total force to forces[iG]
//
//  Physics:
//    Uses getR4repulsion — a short-range R^4 repulsive potential (Pauli-like):
//    V = E * (R0 / r)^4, F = 4*E*(R0/r)^4 / r * r_hat
//    Parameters: SRdR shifts R0 inward, SRamp scales the well depth E.
//    This is a simplified repulsive-only potential for contact/overlap detection.
//
//  CAVEAT: The bounding-box inclusion test uses inverted comparisons:
//    (p.x < bbi.lo.x) && (p.x > bbi.hi.x) — this is correct because BBs
//    stores lo=(xmin,ymin,zmin) and hi=(xmax,ymax,zmax), so p.x < xmin
//    is impossible... actually this looks like a BUG. The condition should
//    probably be (p.x > bbi.lo.x) && (p.x < bbi.hi.x) for "inside box".
//    As written, the condition is never true for normal BBs, so no atoms
//    are ever loaded into local memory, and forces are always zero.
//    This kernel may be unused or the BBs may be stored in inverted order.
//
__kernel void getShortRangeBuckets(
    const int4 ns,                  // 1
    // Dynamical
    __global float4*  atoms,        // 2
    __global float4*  forces,       // 3
    __global int2*    buckets,      // 4 // i0,n for bucket i
    __global float8*  BBs,          // 6 // bounding boxes (xmin,xmax,ymin,0,  ymax,zmin,zmax,0 )
    // Parameters
    __global float4*  REQKs,        // 4
    const float Rcut,
    const float SRdR,
    const float SRamp
    //const int4 nPBC,              // 7
    //const cl_Mat3 lvec,           // 8
    //const float Rdamp             // 9
){
    // local size should be equal to maximum size of one bucket (i.e. maximum number of atoms in one bucket)
    __local float4 POS[16];  // atom positions
    __local float4 PAR[16];  // REQKs parameters
    __local bool   mask[16];

    const int iG = get_global_id  (0);
    const int nG = get_global_size(0);
    const int iL = get_local_id   (0);
    const int nL = get_local_size (0);

    const int  ib = get_group_id(0);
    const int2 bi = buckets[ib];
    if(iL>bi.y) return; // check if atom within group range
    //if(iG>=natoms) return;

    const int nb     = ns.w;
    const int natoms = ns.x;
    //const int nnode =ns.y;

    // ========= Atom-to-Atom interaction ( N-body problem )
    float4 posi  = atoms[iG];
    float4 REQKi = REQKs[iG];
    float4 fe    = float4Zero;
    float8 bbi   = BBs[ib];
    float8 bbi2  = bbi; bbi2.lo.xyz+=Rcut;  bbi2.hi.xyz-=Rcut;


    for(int jb=0; jb<nb; jb++){
        int2 b = buckets[jb];

        // --- PBC replicas ?

        // We do not do this if we make neighborlist for groups
        { // check if bbj overlaps with bbi?
            float8 bbj = BBs[jb];
            if (bbi2.hi.x < bbj.lo.x || bbi2.lo.x > bbj.hi.x ||  // Separated along x-axis?
                bbi2.hi.y < bbj.lo.y || bbi2.lo.y > bbj.hi.y ||  // Separated along y-axis?
                bbi2.hi.z < bbj.lo.z || bbi2.lo.z > bbj.hi.z)    // Separated along z-axis?
            { // No overlap
                continue; // skip this group_j
            }
        }

        int ia = b.x + iL;


        // copy atoms to local memory
        //   * we copy only those atoms which are within the bounding box
        //   * we need to know which atoms were copied, therefore we use mask[]
        mask[iL] = false;
        if( iL < b.y ){
            float4 p = atoms[ia];
            // check if the particle from group_j is inside BBox of group_i
            if( (p.x<bbi.lo.x) && (p.x>bbi.hi.x) &&
                (p.y<bbi.lo.y) && (p.y>bbi.hi.y) &&
                (p.z<bbi.lo.z) && (p.z>bbi.hi.z)
            ){
                POS[iL]  = p;
                PAR[iL]  = REQKs[ia];
                mask[iL] = true;       // we need to know if the atom is in local memory or not
            }
        }
        //mask[iL] = bIn;
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int j=0; j<nL; j++){
            if( mask[j] ){
                const float4 aj = POS[j];
                const float3 dp = aj.xyz - posi.xyz;
                float4 REQK = PAR[j];
                REQK.x +=REQKi.x;
                REQK.yz*=REQKi.yz;
                float4 fij = getR4repulsion( dp, REQK.x-SRdR, REQK.x, REQK.y*SRamp );
                fe += fij;
                //if(iG==4){ printf( "GPU_LJQ[%i,%i|%i] fj(%g,%g,%g) R2damp %g REQ(%g,%g,%g) r %g \n", iG,ji,0, fij.x,fij.y,fij.z, R2damp, REQK.x,REQK.y,REQK.z, length(dp)  ); }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    forces[iG] = fe;
    //forces[iG] = fe*(-1.f);

}


// ======================================================================
//                          getShortRangeBuckets2()
// ======================================================================
//
//  Short-range R4 repulsion with precomputed overlap lists + PBC support.
//  1 workgroup = 1 bucket. Uses precomputed overIndex/overCell from
//  sortAtomsToBucketOverlaps to avoid the bounding-box test at runtime.
//
//  Algorithm:
//    1. Load atom i's position and parameters
//    2. For each chunk of overlapping atoms (from bucketsJs[ib]):
//       a. Cooperatively load chunk into __local memory (POS, PAR, Js, JCs)
//       b. barrier
//       c. For each loaded atom j:
//          - If PBC: apply lattice shift from overCell (packed ilveca, ilvecb)
//          - Compute getR4repulsion(dp, R-SRdR, R, E*SRamp)
//       d. barrier
//    3. Write total force to forces[iG]
//
//  PBC cell encoding: overCell stores (ilveca+8) + (ilvecb+8)*16,
//  where ilveca, ilvecb are in range [-8, 7]. Decoded as:
//    ilveca = ((cell & 0xF0) >> 4) - 8
//    ilvecb = (cell & 0x0F) - 8
//
//  CAVEAT: Line 335 has a bug: `aj += lvec.a*ilveca + lvec.a*ilvecb`
//  should be `aj += lvec.a*ilveca + lvec.b*ilvecb`. The b-direction PBC
//  shift is incorrectly applied with the a lattice vector.
//
__kernel void getShortRangeBuckets2(
    const int4 ns,                  // 1
    // Dynamical
    __global float4*  atoms,        // 2
    __global float4*  forces,       // 3
    __global int2*    buckets,      // 4 // {i0,n} particles which belong to group_i
    __global int2*    bucketsJs,    // 5 // {i0,n} particles which overlap with bounding box of group_i (i.e. can be from any group_j)
    __global int*     overIndex,    //6 indexes of atoms in overlap split by  bucketsJs
    __global int*     overCell,    //6 indexes of atoms in overlap split by  bucketsJs
    // Parameters
    __global float4*  REQKs,        // 8
    __global cl_Mat3* lvecs,
    const float Rcut,
    const float SRdR,
    const float SRamp,
    const int bPBC
    //const int4 nPBC,              // 7
    //const cl_Mat3 lvec,           // 8
    //const float Rdamp             // 9
){
    // local size should be equal to maximum size of one bucket (i.e. maximum number of atoms in one bucket)
    __local float4 POS[16];  // atom positions
    __local float4 PAR[16];  // REQKs parameters
    __local int    Js [16];  // atom index
    __local int    JCs[16];  // cell index

    const int iG = get_global_id  (0);
    const int nG = get_global_size(0);
    const int iL = get_local_id   (0);
    const int nL = get_local_size (0);

    const int  ib = get_group_id(0);
    const int2 bi = buckets[ib];
    if(iL>bi.y) return; // check if atom within group range
    //if(iG>=natoms) return;

    const int natoms=ns.x;
    const int nnode =ns.y;
    const int nb = ns.w;             // number of buckets

    // only if bPBC=true
    const int iS = get_global_id  (1); // index of system
    const cl_Mat3 lvec = lvecs[iS];

    // ========= Atom-to-Atom interaction ( N-body problem )
    float4 posi  = atoms[iG];
    float4 REQKi = REQKs[iG];
    float4 fe    = float4Zero;
    const int2 bj = bucketsJs[ib];
    for (int j0=0; j0<bj.y; j0+=nL){
        const int j=j0+iL;
        if(j<bj.y){  // copy to local memory
            int ja  = overIndex[bj.x+j];
            POS[iL] = atoms[ja];
            PAR[iL] = REQKs[ja];
            Js [iL] = ja;
            JCs[iL] = overCell[ja];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int j=0; j<nL; j++){
            int ja = Js[j];
            float4 aj = POS[j];
            if(bPBC){
                const int ilvec = JCs[j];
                const int ilveca = ((ilvec&0xF0)>>4)-8;
                const int ilvecb = ((ilvec&0x0F)   )-8;
                aj += lvec.a*ilveca + lvec.a*ilvecb;
            }
            const float3 dp = aj.xyz - posi.xyz;
            float4 REQK = PAR[j];
            REQK.x +=REQKi.x;
            REQK.yz*=REQKi.yz;
            float4 fij = getR4repulsion( dp, REQK.x-SRdR, REQK.x, REQK.y*SRamp );
            fe += fij;
            //if(iG==4){ printf( "GPU_LJQ[%i,%i|%i] fj(%g,%g,%g) R2damp %g REQ(%g,%g,%g) r %g \n", iG,ji,0, fij.x,fij.y,fij.z, R2damp, REQK.x,REQK.y,REQK.z, length(dp)  ); }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    };
    forces[iG] = fe;
    //forces[iG] = fe*(-1.f);
}


// ======================================================================
//                          sortAtomsToBucketOverlaps()
// ======================================================================
//
//  Precompute which atoms overlap each bucket's bounding box, for use by
//  getShortRangeBuckets2. 1 workgroup = 1 bucket, 1 system per dim-1.
//
//  Algorithm:
//    For each bucket ib (with bounding box bbi):
//    1. For each tile of nL atoms:
//       a. Cooperatively load atom positions into __local memory
//       b. barrier
//       c. For each atom j in tile:
//          - If PBC: loop over all PBC images, check if j+shift is inside bbi
//            -> If yes: store (j_index, pbc_cell) in overIndex/overCell
//          - If no PBC: check if j is inside bbi
//            -> If yes: store j_index in overIndex
//       d. (no barrier needed — next tile overwrites local memory)
//
//  Output: overIndex[iB0 + k] = global atom index of k-th overlapping atom.
//          overCell[iB0 + k]  = packed PBC cell (ilveca+8 + (ilvecb+8)*16).
//  where iB0 = bucketsJs[ib].x is the start offset for bucket ib.
//
//  CAVEAT: Same inverted bounding-box comparison as getShortRangeBuckets:
//  (p.x < bbi.lo.x) && (p.x > bbi.hi.x) — see caveat in that kernel.
//  If BBs are stored in standard order (lo=min, hi=max), this condition
//  is never true, so no atoms are ever sorted. The kernel may rely on
//  BBs being stored in inverted order (lo=max, hi=min).
//
__kernel void sortAtomsToBucketOverlaps(
    const int4 ns,                  // 1
    // Dynamical
    __global float4*  atoms,        // 2
    __global float4*  shifts,
    __global int2*    buckets,      // 4 // i0,n for bucket i
    __global float8*  BBs,          // 6 // bounding boxes (xmin,xmax,ymin,0,  ymax,zmin,zmax,0 )
    __global int2*    bucketsJs,    // 5 // {i0,n} particles which overlap with bounding box of group_i (i.e. can be from any group_j)
    __global int*     overIndex,    //6   indexes of atoms in overlap split by  bucketsJs
    __global int*     overCell,     //    index of PBC cell of the atoms in the overlpa
    //__global float4*  overParams,   //6 indexes of atoms in overlap split by  bucketsJs
    //__global float4*  overPos,      //6 indexes of atoms in overlap split by  bucketsJs
    __global cl_Mat3* lvecs,
    const int4 nPBC,
    const float Rcut
){
    // local size should be equal to maximum size of one bucket (i.e. maximum number of atoms in one bucket)
    __local int    IND[16];
    __local float3 POS[16];  // atom positions
    //__local float4 PAR[16];  // REQKs parameters

    const int iG = get_global_id  (0);
    const int nG = get_global_size(0);
    const int iL = get_local_id   (0);
    const int nL = get_local_size (0);

    const int  ib = get_group_id(0);
    const int2 bi = buckets[ib];
    if(iL>bi.y) return; // check if atom within group range
    //if(iG>=natoms) return;

    const int iS = get_global_id  (1); // index of system
    const int nS = get_global_size(1); // number of systems
    const int natoms=ns.x;  // number of atoms
    const int nnode =ns.y;  // number of node atoms
    const int nvec  =natoms+nnode; // number of vectors (atoms+node atoms)

    const int nb     = ns.w;
    const int i0v = iS*nvec;    // index of first atom in vectors array

    // ========= Atom-to-Atom interaction ( N-body problem )
    float8 bbi   = BBs[ib];

    int iB0 = bucketsJs[iG].x;

    const cl_Mat3 lvec = lvecs[iS];
    const bool bPBC = (nPBC.x+nPBC.y+nPBC.z)>0;

    // For simplicity we go over all atoms - ignoring buckets
    int nfound = 0;
    for (int j0=0; j0<nG; j0+=nL){
        const int i=j0+iL;
        if(i<natoms){
            int ja  = i+i0v;
            POS[iL] = atoms[ja].xyz;
            IND[iL] = ja;
        }
        barrier(CLK_LOCAL_MEM_FENCE);   // wait until all atoms are read to local memory
        for (int jl=0; jl<nL; jl++){    // loop over all atoms in local memory (like 32 atoms)
            const int ja=j0+jl;         // index of atom in global memory
            if( ja<natoms){   // if atom is not the same as current atom and it is not out of range,  // ToDo: Should atom interact with himself in PBC ?
                const float3 p = POS[jl];    // read atom position   from local memory

                if(bPBC){
                    //int ipbc=0;
                    //dp += shift0;
                    // Fixed PBC size
                    for(int iy=-nPBC.y; iy<=nPBC.y; iy++){
                        float3 dp = p + lvec.b.xyz*iy - lvec.a.xyz*nPBC.x;
                        for(int ix=-nPBC.x; ix<=nPBC.x; ix++){
                            if( (dp.x<bbi.lo.x) && (dp.x>bbi.hi.x) &&
                                (dp.y<bbi.lo.y) && (dp.y>bbi.hi.y) &&
                                (dp.z<bbi.lo.z) && (dp.z>bbi.hi.z)
                            ){
                                int isave = iB0 + nfound;
                                overIndex[isave] = IND[jl];
                                overCell [isave] = (ix+8) + (iy+8)*16;
                                nfound++;
                            }
                            //ipbc++;
                            dp += lvec.a.xyz;
                        }
                        //dp    += lvec.a.xyz;
                    }
                }else{
                    if( (p.x<bbi.lo.x) && (p.x>bbi.hi.x) &&
                        (p.y<bbi.lo.y) && (p.y>bbi.hi.y) &&
                        (p.z<bbi.lo.z) && (p.z>bbi.hi.z)
                    ){
                        int isave = iB0 + nfound;
                        overIndex[isave] = IND[jl];
                        //= overCell [];
                        nfound++;
                    }
                }
            }
        }
        //barrier(CLK_LOCAL_MEM_FENCE);
    }
}

