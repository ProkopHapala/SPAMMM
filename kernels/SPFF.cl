// spff.cl - SPFFsp3 force field: bonding interactions + MD integrator
//
// Implements the SPFFsp3 force field for molecular dynamics with pi-orbital
// degrees of freedom. Each node atom has up to 4 neighbors with bond, angle,
// torsion, and pi-pi alignment interactions. Pi-orbitals are additional DOFs
// that carry recoil forces gathered from back-neighbors during integration.
//
// Execution flow (one MD step):
//   1. getSPFFf4 — evaluate bonding forces (bonds, angles, pi-pi, H-bond)
//   2. getNonBond_ex2 (in nonbonded.cl) — evaluate non-bonded LJ/Coulomb forces
//   3. cleanForceSPFFf4 — zero/clear force buffers for next step
//   4. updateAtomsSPFFf4 — gather recoil forces, integrate positions/velocities,
//      normalize pi-orbitals, apply constraints and bounding boxes
//
// Kernels:
//   - getSPFFf4: Compute bonding forces for all node atoms (bonds, angles,
//     torsions, pi-pi alignment, H-bond correction). Stores atom forces in
//     fapos and recoil forces on neighbors in fneigh. 1 thread = 1 node atom.
//   - updateGroups: Compute group center-of-mass, forward direction, and up
//     vector for rigid groups of atoms (used for constrained groups).
//   - groupForce: Distribute external forces on groups back to individual atoms.
//   - updateAtomsSPFFf4: MD integrator — gather recoil from fneigh/bkNeighs,
//     apply constraints (springs, bounding boxes, inter-system bonds), update
//     velocities and positions, normalize pi-orbital vectors. Supports thermal
//     driving and force limiting.
//   - cleanForceSPFFf4: Clear/zero force arrays between MD steps.
//
// Helper functions: evalAngCos, evalAngleCosHalf (angular force/energy),
// evalPiAling (pi-pi alignment), evalBond (harmonic bond stretching).
// Requires: common.cl + Forces.cl to be concatenated before this file.

// ---- SPFF bonding helper functions ----
inline float evalAngCos( const float4 hr1, const float4 hr2, float K, float c0, __private float3* f1, __private float3* f2 ){
    float  c = dot(hr1.xyz,hr2.xyz);
    float3 hf1,hf2;
    hf1 = hr2.xyz - hr1.xyz*c;
    hf2 = hr1.xyz - hr2.xyz*c;
    float c_   = c-c0;
    float E    = K*c_*c_;
    float fang = -K*c_*2;
    hf1 *= fang*hr1.w;
    hf2 *= fang*hr2.w;
    *f1=hf1;
    *f2=hf2;
    return E;
}

// evaluate angular force and energy using cos(angle/2) formulation - a bit slower, but not good for angles > 90 deg
inline float evalAngleCosHalf( const float4 hr1, const float4 hr2, const float2 cs0, float k, __private float3* f1, __private float3* f2 ){
    // This is much better angular function than evalAngleCos() with just a little higher computational cost ( 2x sqrt )
    // the main advantage is that it is quasi-harmonic beyond angles > 90 deg
    float3 h  = hr1.xyz + hr2.xyz;  // h = a+b
    float  c2 = dot(h,h)*0.25f;     // cos(a/2) = |ha+hb|  (after normalization)
    float  s2 = 1.f-c2 + 1e-7;      // sin(a/2) = sqrt(1-cos(a/2)^2) ;  s^2 must be positive (otherwise we get NaNs)
    float2 cso = (float2){ sqrt(c2), sqrt(s2) }; // cso = cos(a/2) + i*sin(a/2)
    float2 cs = udiv_cmplx( cs0, cso );          // rotate back by equilibrium angle
    float  E         =  k*( 1 - cs.x );          // E = k*( 1 - cos(a/2) )  ; Do we need Energy? Just for debugging ?
    float  fr        = -k*(     cs.y );          // fr = k*( sin(a/2) )     ; force magnitude
    c2 *= -2.f;
    fr /=  4.f*cso.x*cso.y;   //    |h - 2*c2*a| =  1/(2*s*c) = 1/sin(a)
    float  fr1    = fr*hr1.w; // magnitude of force on atom a
    float  fr2    = fr*hr2.w; // magnitude of force on atom b
    *f1 =  h*fr1  + hr1.xyz*(fr1*c2);  //fa = (h - 2*c2*a)*fr / ( la* |h - 2*c2*a| ); force on atom a
    *f2 =  h*fr2  + hr2.xyz*(fr2*c2);  //fb = (h - 2*c2*b)*fr / ( lb* |h - 2*c2*b| ); force on atom b
    return E;
}

// evaluate angular force and energy for pi-pi alignment interaction
inline float evalPiAling( const float3 h1, const float3 h2,  float K, __private float3* f1, __private float3* f2 ){  // interaction between two pi-bonds
    float  c = dot(h1,h2); // cos(a) (assumes that h1 and h2 are normalized)
    float3 hf1,hf2;        // working forces or direction vectors
    hf1 = h2 - h1*c;       // component of h2 perpendicular to h1
    hf2 = h1 - h2*c;       // component of h1 perpendicular to h2
    bool sign = c<0; if(sign) c=-c; // if angle is > 90 deg we need to flip the sign of force
    float E    = -K*c;     // energy is -K*cos(a)
    float fang =  K;       // force magnitude
    if(sign)fang=-fang;    // flip the sign of force if angle is > 90 deg
    hf1 *= fang;           // force on atom a
    hf2 *= fang;           // force on atom b
    *f1=hf1;
    *f2=hf2;
    return E;
}

// evaluate bond force and energy for harmonic bond stretching
inline float evalBond( float3 h, float dl, float k, __private float3* f ){
    float fr = dl*k;   // force magnitude
    *f = h * fr;       // force on atom a
    return fr*dl*0.5;  // energy
}

// ---- Torque-based pi interaction helpers (for rotational pi dynamics) ----

// pi-sigma orthogonalization: returns torque on pi-orbital + recoil force on neighbor bond
inline float evalPiSigma_tq(const float3 hpi, const float4 h, const float K, const float c0, __private float3 *tqi, __private float3 *fj){
    const float c    = dot(hpi, h.xyz);
    const float c_   = c - c0;
    const float E    = K * c_ * c_;
    const float fang = -2.0f * K * c_;
    *tqi = cross(hpi, h.xyz) * fang;          // torque on pi
    const float s2   = fang * h.w;            // recoil scaling
    *fj  = (hpi - h.xyz * c) * s2;            // recoil force perpendicular to bond
    return E;
}

// pi-pi alignment: returns (torque on pi, energy) as float4
inline float4 evalPiAlign_tq(const float3 h1, const float3 h2, const float K){
    const float c = dot(h1,h2);
    const float E = -K * c;
    return (float4){ cross(h1, h2) * K, E };
}

// ---- Rotation helpers for pi-orbital rotational dynamics ----

float2 sinc_div_r2_taylor(float r2){
    float s = 1.0f + r2 * ( (-1.0f/6.0f)  + r2 * ( (1.0f/120.0f) + r2 * (-1.0f/5040.0f  ) ) );
    float c = 0.5f + r2 * ( (-1.0f/24.0f) + r2 * ( (1.0f/720.0f) + r2 * (-1.0f/40320.0f ) ) );
    return (float2){s, c};
}

float3 rotate_by_omega_taylor(float3 p, float3 w){
    float r2    = dot(w,w);
    float2 sc   = sinc_div_r2_taylor(r2);
    float3 wxp  = cross(w, p);
    float3 wwxp = cross(w, wxp);
    return p + wxp*sc.x + wwxp*sc.y;
}

// ---- getSPFFf4: bonding interactions (bonds, angles, torsions, pi-pi, H-bond) ----
// ======================================================================
//                          getSPFFf4()
// ======================================================================

// 1.  getSPFFf4() - computes bonding interactions between atoms and nodes and its neighbors (max. 4 neighbors allowed), the resulting forces on atoms are stored "fapos" array and recoil forces on neighbors are stored in "fneigh" array
//                   kernel run over all atoms and all systems in parallel to exploit GPU parallelism
//__attribute__((reqd_work_group_size(1,1,1)))
__kernel void getSPFFf4(
    const int4 nDOFs,               // 1   (nAtoms,nnode) dimensions of the system
    // Dynamical
    __global float4*  apos,         // 2  [natoms]     positions of atoms (including node atoms [0:nnode] and capping atoms [nnode:natoms] and pi-orbitals [natoms:natoms+nnode] )
    __global float4*  fapos,        // 3  [natoms]     forces on    atoms (just node atoms are evaluated)
    __global float4*  fneigh,       // 4  [nnode*4*2]  recoil forces on neighbors (and pi-orbitals)
    // parameters
    __global int4*    neighs,       // 5  [nnode]  neighboring atoms
    __global int4*    neighCell,    // 5  [nnode]  neighboring atom  cell index
    __global float4*  REQKs,        // 6  [natoms] non-boding parametes {R0,E0,Q} i.e. R0: van der Waals radii, E0: well depth and partial charge, Q: partial charge
    __global float4*  apars,        // 7  [nnode]  per atom forcefield parametrs {c0ss,Kss,c0sp}, i.e. c0ss: cos(equlibrium angle/2) for sigma-sigma; Kss: stiffness of sigma-sigma angle; c0sp: is cos(equlibrium angle) for sigma-pi
    __global float4*  bLs,          // 8  [nnode]  bond length    between node and each neighbor
    __global float4*  bKs,          // 9  [nnode]  bond stiffness between node and each neighbor
    __global float4*  Ksp,          // 10 [nnode]  stiffness of pi-alignment for each neighbor     (only node atoms have pi-pi alignemnt interaction)
    __global float4*  Kpp,          // 11 [nnode]  stiffness of pi-planarization for each neighbor (only node atoms have pi-pi alignemnt interaction)
    __global cl_Mat3* lvecs,        // 12 lattice vectors         for each system
    __global cl_Mat3* ilvecs,       // 13 inverse lattice vectors for each system
    __global float4*  pbc_shifts,   // 14 pbc shifts for each system
    const int npbc,                 // 15 number of pbc shifts
    const int bSubtractVdW          // 16 subtract vdW energy
){

    const int iG = get_global_id (0);   // intex of atom   (iG<nAtoms)
    const int iS = get_global_id (1);   // index of system (iS<nS)
    //const int nG = get_global_size(0);
    //const int nS = get_global_size(1);  // number of systems
    //const int iL = get_local_id  (0);
    //const int nL = get_local_size(0);
    const int nAtoms=nDOFs.x;  // number of atoms in the system
    const int nnode =nDOFs.y;  // number of nodes in the system
    //const int nvec  = nAtoms+nnode;

    if(iG>=nnode) return;

    const int i0a   = iS*nAtoms;         // index of first atom      in the system
    const int i0n   = iS*nnode;          // index of first node atom in the system
    const int i0v   = iS*(nAtoms+nnode); // index of first vector    in the system ( either atom or pi-orbital )

    const int iaa = iG + i0a;  // index of current atom (either node or capping atom)
    const int ian = iG + i0n;  // index of current node atom
    const int iav = iG + i0v;  // index of current vector ( either atom or pi-orbital )

    #define NNEIGH 4

    // ---- Dynamical
    float4  hs [4];              // direction vectors of bonds (h.xyz) and inverse bond lengths (h.w)
    float3  fbs[4];              // force on neighbor sigma    (fbs[i] is sigma recoil force on i-th neighbor)
    float3  fps[4];              // force on neighbor pi       (fps[i] is pi    recoil force on i-th neighbor)
    float3  fa  = float3Zero;    // force on center atom positon

    float E=0;                   // Total Energy of this atom
    // ---- Params
    const int4   ng  = neighs[iaa];    // neighboring atoms
    const float3 pa  = apos[iav].xyz;  // position of current atom
    const float4 par = apars[ian];     // (xy=s0_ss,z=ssK,w=piC0 ) forcefield parameters for current atom


    // Temp Arrays
    const int*   ings  = (int*  )&ng; // neighboring atoms, we cast it to int[] to be index it in for loop


    const float   ssC0   = par.x*par.x - par.y*par.y;                      // cos(2) = cos(x)^2 - sin(x)^2, because we store cos(ang0/2) to use in  evalAngleCosHalf , where ang0 is equilibrium angle
    for(int i=0; i<NNEIGH; i++){ fbs[i]=float3Zero; fps[i]=float3Zero; }   // clear recoil forces on neighbors

    float3 f1,f2;         // working forces

    #if DBG_UFF
    if((iG==0)&&(iS==0)){
        printf( "getSPFFf4() iG %i, iS %i, iaa %i bSubtractVdW %i\n", iG, iS, iaa, bSubtractVdW );
    }
    #endif

    { // ========= BONDS - here we evaluate pairwise interactions of node atoms with its 4 neighbors

        float3  fpi = float3Zero;                // force on pi-orbital
        const int4   ngC = neighCell[iaa];       // neighboring atom cell index
        const float3 hpi = apos[iav+nAtoms].xyz; // direction of pi-orbital
        const float4 vbL = bLs[ian];             // bond lengths
        const float4 vbK = bKs[ian];             // bond stiffness
        const float4 vKs = Ksp[ian];             // stiffness of sigma-pi othogonalization
        const float4 vKp = Kpp[ian];             // stiffness of pi-pi    alignment

        const int*   ingC  = (int*  )&ngC;   // neighboring atom cell index (we cast it to int[] to be index it in for loop)
        const float* bL    = (float*)&vbL;   // bond lengths
        const float* bK    = (float*)&vbK;   // bond stiffness
        const float* Kspi  = (float*)&vKs;   // stiffness of sigma-pi othogonalization
        const float* Kppi  = (float*)&vKp;   // stiffness of pi-pi    alignment

        const int ipbc0 = iS*npbc;  // index of first PBC shift for current system

        for(int i=0; i<NNEIGH; i++){  // loop over 4 neighbors
            float4 h;                 // direction vector of bond
            const int ing  = ings[i]; // index of i-th neighbor node atom
            const int ingv = ing+i0v; // index of i-th neighbor vector
            const int inga = ing+i0a; // index of i-th neighbor atom
            if(ing<0) break;

            // --- Compute bond direction vector and inverse bond length
            h.xyz    = apos[ingv].xyz - pa;  // direction vector of bond
            { // shift bond to the proper PBC cell
                int ic  = ingC[i];                  // index of i-th neighbor cell
                h.xyz  += pbc_shifts[ipbc0+ic].xyz; // shift bond to the proper PBC cell
            }
            float  l = length(h.xyz);  // compute bond length
            h.w      = 1./l;           // store ivnerse bond length
            h.xyz   *= h.w;            // normalize bond direction vector
            hs[i]    = h;              // store bond direction vector and inverse bond length

            float epp = 0; // pi-pi    energy
            float esp = 0; // pi-sigma energy

            // --- Evaluate bond-length stretching energy and forces
            if(iG<ing){
                E+= evalBond( h.xyz, l-bL[i], bK[i], &f1 );  fbs[i]-=f1;  fa+=f1;   // harmonic bond stretching, fa is force on center atom, fbs[i] is recoil force on i-th neighbor,

                // pi-pi alignment interaction
                float kpp = Kppi[i];
                if( (ing<nnode) && (kpp>1.e-6) ){   // Only node atoms have pi-pi alignemnt interaction
                    epp += evalPiAling( hpi, apos[ingv+nAtoms].xyz, kpp,  &f1, &f2 );   fpi+=f1;  fps[i]+=f2;    //   pi-alignment(konjugation), fpi is force on pi-orbital, fps[i] is recoil force on i-th neighbor's pi-orbital
                    E+=epp;
                }
            }

            // pi-sigma othogonalization interaction
            float ksp = Kspi[i];
            if(ksp>1.e-6){
                esp += evalAngCos( (float4){hpi,1.}, h, ksp, par.w, &f1, &f2 );   fpi+=f1; fa-=f2;  fbs[i]+=f2;    //   pi-planarization (orthogonality), fpi is force on pi-orbital, fbs[i] is recoil force on i-th neighbor
                E+=esp;
            }
        }

        // --- Store Pi-forces                      we store pi-forces here because we don't use them in the angular force evaluation
        const int i4p=(iG + iS*nnode*2 )*4 + nnode*4; // index of first pi-force for current atom
        for(int i=0; i<NNEIGH; i++){
            fneigh[i4p+i] = (float4){fps[i],0}; // store recoil pi-force on i-th neighbor
        }
        fapos[iav+nAtoms]  = (float4){fpi,0};  // store pi-force on pi-orbital of current atom

    }

    { //  ============== Angles   - here we evaluate angular interactions between pair of sigma-bonds of node atoms with its 4 neighbors

        for(int i=0; i<NNEIGH; i++){ // loop over first bond
            int ing = ings[i];
            if(ing<0) break;         // if there is no i-th neighbor we break the loop
            const float4 hi = hs[i];
            const int ingv = ing+i0v;
            const int inga = ing+i0a;
            for(int j=i+1; j<NNEIGH; j++){ // loop over second bond
                int jng  = ings[j];
                if(jng<0) break;           // if there is no j-th neighbor we break the loop
                const int jngv = jng+i0v;
                const int jnga = jng+i0a;
                const float4 hj = hs[j];

                E += evalAngleCosHalf( hi, hj, par.xy, par.z, &f1, &f2 );    // evaluate angular force and energy using cos(angle/2) formulation
                fa    -= f1+f2;

                if(bSubtractVdW)
                { // Remove non-bonded interactions from atoms that are bonded to common neighbor
                    float4 REQi=REQKs[inga];   // non-bonding parameters of i-th neighbor
                    float4 REQj=REQKs[jnga];   // non-bonding parameters of j-th neighbor
                    // combine non-bonding parameters of i-th and j-th neighbors using mixing rules
                    float4 REQij;
                    REQij.x  = REQi.x  + REQj.x;
                    REQij.yz = REQi.yz * REQj.yz;

                    float3 dp = (hj.xyz/hj.w) - (hi.xyz/hi.w);   // recover vector between i-th and j-th neighbors using stored vectos and inverse bond lengths, this should be faster than dp=apos[jngv].xyz-apos[ingv].xyz; from global memory
                    float4 fij = getLJQH( dp, REQij, 1.0f );     // compute non-bonded interaction between i-th and j-th neighbors using Lennard-Jones and Coulomb interactions and Hydrogen bond correction
                    f1 -=  fij.xyz;
                    f2 +=  fij.xyz;
                }

                fbs[i]+= f1;
                fbs[j]+= f2;
            }
        }

    }

    // ========= Save results - store forces on atoms and recoil on its neighbors  (pi-forces are already done)
    const int i4 =(iG + iS*nnode*2 )*4;
    //const int i4p=i4+nnode*4;
    for(int i=0; i<NNEIGH; i++){
        fneigh[i4 +i] = (float4){fbs[i],0};
        //fneigh[i4p+i] = (float4){fps[i],0};
    }
    //fapos[iav     ] = (float4){fa ,0}; // If we do  run it as first forcefield
    fapos[iav       ] += (float4){fa ,E};  // If we not run it as first forcefield, store energy in .w
    //fapos[iav+nAtoms]  = (float4){fpi,0};

}

// ======================================================================
//                          getSPFFf4_rot()
// ======================================================================
// Same as getSPFFf4 but uses torque-based pi interactions for rotational dynamics.
// Pi-orbital forces are torques (not linear forces), stored in aforce[iav+nAtoms].
//__attribute__((reqd_work_group_size(1,1,1)))
__kernel void getSPFFf4_rot(
    const int4 nDOFs,               // 1   (nAtoms,nnode) dimensions of the system
    __global float4*  apos,         // 2  [natoms]     positions of atoms
    __global float4*  aforce,       // 3  [natoms]     forces on atoms
    __global float4*  fneigh,       // 4  [nnode*4*2]  recoil forces on neighbors
    __global int4*    neighs,       // 5  [nnode]  neighboring atoms
    __global int4*    neighCell,    // 5  [nnode]  neighboring atom cell index
    __global float4*  REQs,         // 6  [natoms] non-bonding parameters
    __global float4*  apars,        // 7  [nnode]  per atom forcefield parameters
    __global float4*  bLs,          // 8  [nnode]  bond lengths
    __global float4*  bKs,          // 9  [nnode]  bond stiffness
    __global float4*  Ksp,          // 10 [nnode]  stiffness of pi-sigma orthogonalization
    __global float4*  Kpp,          // 11 [nnode]  stiffness of pi-pi alignment
    __global cl_Mat3* lvecs,        // 12 lattice vectors
    __global cl_Mat3* ilvecs,       // 13 inverse lattice vectors
    __global float4*  pbc_shifts,
    const int npbc,
    const int bSubtractVdW
){
    const int iG = get_global_id (0);
    const int iS = get_global_id (1);
    const int nAtoms=nDOFs.x;
    const int nnode =nDOFs.y;
    if(iG>=nnode) return;

    const int i0a   = iS*nAtoms;
    const int i0n   = iS*nnode;
    const int i0v   = iS*(nAtoms+nnode);
    const int iaa = iG + i0a;
    const int ian = iG + i0n;
    const int iav = iG + i0v;

    #define NNEIGH 4
    float4  hs [4];
    float3  fbs[4];
    float3  fa  = float3Zero;
    float E=0;
    const int4   ng  = neighs[iaa];
    const float3 pa  = apos[iav].xyz;
    const float4 par = apars[ian];
    const int*   ings  = (int*)&ng;
    const float   ssC0   = par.x*par.x - par.y*par.y;
    for(int i=0; i<NNEIGH; i++){ fbs[i]=float3Zero; }
    float3 f1,f2;

    {
        float3  fpi = float3Zero;
        const int4   ngC = neighCell[iaa];
        const float3 hpi = apos[iav+nAtoms].xyz;
        const float4 vbL = bLs[ian];
        const float4 vbK = bKs[ian];
        const float4 vKs = Ksp[ian];
        const float4 vKp = Kpp[ian];
        const int*   ingC  = (int*)&ngC;
        const float* bL    = (float*)&vbL;
        const float* bK    = (float*)&vbK;
        const float* Kspi  = (float*)&vKs;
        const float* Kppi  = (float*)&vKp;
        const int ipbc0 = iS*npbc;

        for(int i=0; i<NNEIGH; i++){
            float4 h;
            const int ing  = ings[i];
            const int ingv = ing+i0v;
            if(ing<0) break;
            h.xyz    = apos[ingv].xyz - pa;
            { int ic = ingC[i]; h.xyz += pbc_shifts[ipbc0+ic].xyz; }
            float  l = length(h.xyz);
            h.w      = 1.f/l;
            h.xyz   *= h.w;
            hs[i]    = h;

            if(iG<ing){
                float elb = evalBond( h.xyz, l-bL[i], bK[i], &f1 );  fbs[i]-=f1;  fa+=f1; E+=elb;
            }

            float kpp = Kppi[i];
            if( (ing<nnode) && (kpp>1.e-6f) ){
                float3 hpj = apos[ingv+nAtoms].xyz;
                float4 fepi = evalPiAlign_tq( hpi, hpj, kpp );
                E  += fepi.w;
                fpi += fepi.xyz;
            }

            float ksp = Kspi[i];
            if(ksp>1.e-6f){
                float esp = evalPiSigma_tq( hpi, h, ksp, par.w, &f1, &f2 );
                E  += esp; fa-=f2;  fbs[i]+=f2; fpi+=f1;
            }
        }

        aforce[iav+nAtoms]  = (float4){fpi,0};
    }

    {
        for(int i=0; i<NNEIGH; i++){
            int ing = ings[i];
            if(ing<0) break;
            const float4 hi = hs[i];
            for(int j=i+1; j<NNEIGH; j++){
                int jng  = ings[j];
                if(jng<0) break;
                const float4 hj = hs[j];
                float ea = evalAngleCosHalf( hi, hj, par.xy, par.z, &f1, &f2 );
                fa  -= f1+f2;
                E   += ea;
                fbs[i]+= f1;
                fbs[j]+= f2;
            }
        }
    }

    const int i4 =(iG + iS*nnode*2 )*4;
    for(int i=0; i<NNEIGH; i++){
        fneigh[i4 +i] = (float4){fbs[i],0};
    }
    aforce[iav ] += (float4){fa.x,fa.y,fa.z,E};
}

// ---- updateGroups ----
// ======================================================================
//                     updateGroups()
// ======================================================================

//__attribute__((reqd_work_group_size(1,1,1)))
__kernel void updateGroups(
    int               ngroup,      // 1 // number of groups (total, for all systems)
    __global int2*    granges,     // 2 // (i0,n) range of indexes specifying the group
    __global int*     g2a,         // 3 // indexes of atoms corresponding to groups defined by granges
    __global float4*  apos,        // 4 // positions of atoms  (including node atoms [0:nnode] and capping atoms [nnode:natoms] and pi-orbitals [natoms:natoms+nnode] )
    __global float4*  gcenters,    // 5 // centers of each groups (CoGs)
    __global float4*  gfws,        // 6 // forwad  orietantian vector for each group
    __global float4*  gups,        // 7 // up      orietantian vector for each group
    __global float4*  gweights     // 8 // up      orietantian vector for each group
){
    const int iG = get_global_id  (0); // index of atom
    if(iG>=ngroup) return; // make sure we are not out of bounds of current system

    // if(iG==0){
    //     printf( "GPU ngroup=%i \n", ngroup );
    //     for(int i=0; i<ngroup; i++){
    //         const int2 grange = granges[i];
    //         printf("GPU granges[%i] i0=%i n=%i \n", i, grange.x, grange.y  );
    //         for(int j=0; j<grange.y; j++){
    //             int ia = g2a[ grange.x + j ];
    //             //printf( "[%i] %i \n", j, ia );
    //             printf( "GPU gweights[%i](%g,%g,%g,%g)\n", ia, gweights[ia].x,gweights[ia].y,gweights[ia].z,gweights[ia].w );
    //         }
    //         printf("\n");
    //     }
    // }

    const int2 grange = granges[iG];

    float3 cog = (float3){0.0f,0.0f,0.0f};

    float wsum = 0.f;
    for(int i=0; i<grange.y; i++){
        int ia = g2a[ grange.x + i ];
        //const float4 pe = apos[ia];
        const float4 w = gweights[ia];
        cog    += apos[ia].xyz * w.x;
        wsum   += w.x;
    }
    cog *= ( 1.f/wsum );
    gcenters[iG] = (float4){cog,0.0f};

    float3 up  = (float3){0.0f,0.0f,0.0f};
    float3 fw  = (float3){0.0f,0.0f,0.0f};
    for(int i=0; i<grange.y; i++){
        int ia = g2a[ grange.x + i ];
        //const float4 pe = apos[ia];
        const float4 w = gweights[ia];
        const float3 d = apos[ia].xyz - cog.xyz;
        fw.xyz += d * w.y;
        up.xyz += d * w.z;
    }
    {  // Orthonormalize
        fw  = normalize( fw );
        up += fw * -dot( fw, up );
        up  = normalize( up );
    }

    //printf( "GPU[iG=%i] cog(%g,%g,%g) fw(%g,%g,%g) up(%g,%g,%g) \n", iG, cog.x,cog.y,cog.z,   fw.x,fw.y,fw.z,  up.x,up.y,up.z );
    gfws[iG] = (float4){fw,0.0f};
    gups[iG] = (float4){up,0.0f};
}

// ---- groupForce ----
// ======================================================================
//                     groupForce()
// ======================================================================

//__attribute__((reqd_work_group_size(1,1,1)))
__kernel void groupForce(
    const int4        n,            // 1 // (natoms,nnode) dimensions of the system
    __global float4*  apos,         // 2 // positions of atoms  (including node atoms [0:nnode] and capping atoms [nnode:natoms] and pi-orbitals [natoms:natoms+nnode] )
    __global float4*  aforce,       // 3 // forces on atoms
    __global int*     a2g,          // 4 // atom to group maping (index)
    __global float4*  gforces,      // 5 // linar forces appliaed to atoms of the group
    __global float4*  gtorqs,       // 6 // {hx,hy,hz,t} torques applied to atoms of the group
    __global float4*  gcenters,     // 7 // centers of rotation (for evaluation of the torque
    __global float4*  gfws,         // 8 // forward vector of group orientation
    __global float4*  gups,         // 9 // up      vector of group orientation
    __global float2*  gfweights    // 10 // weights for application of forces on atoms
){
    const int natoms = n.x;           // number of atoms
    const int nnode  = n.y;           // number of node atoms
    const int nGrpup = n.w;           // number of node atoms
    const int nvec   = natoms+nnode; // number of vectors (atoms+node atoms)
    const int iG = get_global_id  (0); // index of atom

    if(iG>=natoms) return; // make sure we are not out of bounds of current system

    const int iS = get_global_id  (1); // index of system
    const int nG = get_global_size(0); // number of atoms
    const int nS = get_global_size(1); // number of systems

    // if( (iG==0) && (iS==0) ){
    //     printf( "GPU::groupForce() natom=%i nnode=%i nvec=%i \n", natoms, nnode, nvec );
    // //     int ig_sel = 0;
    // //     int is = 0;
    // //     // for(int ia=0; ia<natoms; ia++){
    // //     //      int iav = ia + is*nvec;
    // //     //     printf( "%i ", a2g[iav] );
    // //     // }
    // //     // printf("\n");

    //     for(int is=0; is<nS; is++){
    //         // printf( "sys[%i] ", is );
    //         // for(int ia=0; ia<natoms; ia++){
    //         //     int iav = ia + is*nvec;
    //         //     printf( "%i ", a2g[iav] );
    //         // }
    //         // printf("\n");
    //         for(int ia=0; ia<natoms; ia++){
    //             int iav = ia + is*nvec;
    //             const int ig = a2g[iav];
    //             if(ig>=0){
    //                 //printf( "GPU:atom[%i|%i,%i] ig=%i(%i/%i) gforces(%10.6f,%10.6f,%10.6f)\n", is, ia, iav, ig, ig-is*nGrpup,nGrpup, gforces[ig].x, gforces[ig].y, gforces[ig].z  );
    //                 printf( "GPU:atom[isys=%i|ia=%i] gfweights[iav=%i](%10.6f,%10.6f) gtorqs[ig=%i](%10.6f,%10.6f,%10.6f,%10.6f)\n", is, ia,     iav,  gfweights[iav].x,gfweights[iav].y,    ig, gtorqs[ig].x, gtorqs[ig].y, gtorqs[ig].z, gtorqs[ig].w  );
    //             }
    //         }
    //     }
    // }

    //const int ian = iG + iS*nnode;
    const int iaa = iG + iS*natoms;  // index of atom in atoms array
    const int iav = iG + iS*nvec;    // index of atom in vectors array

    float4 fe    = aforce[iav]; // position of atom or pi-orbital
    const int ig = a2g[iav];  // index of the group to which this atom belongs

    float2  w = gfweights[ig];

    // --- apply linear forece from the group
    fe.xyz += gforces[ig].xyz * w.x;

    // ToDo: group vectors may be stored in Local Memory ?
    const float3 torq = gtorqs[ig].xyz;
    const float3 fw   = gfws  [ig].xyz;
    const float3 up   = gups  [ig].xyz;
    const float3 lf   = normalize( cross(fw,up) );
    const float3 tq   = fw * torq.x   +  up * torq.y    +   lf * torq.z;

    // --- apply torque from the group
    const float3 dp  = apos[iav].xyz - gcenters[ig].xyz;
    fe.xyz          += cross( dp, tq.xyz ) * w.x;

    // --- store results
    aforce[iav] = fe;

}

// ======================================================================
//                     updateAtomsSPFFf4()
// ======================================================================

/*
float2 KvaziFIREdamp( float c, float2 damp_lims, float2 clim ){
    float2 cvf;
    if      (c < clim.x ){   //-- force against veloctiy
        cvf.x = damp_lims.x; // v    // like 0.5 (strong damping)
        cvf.y = 0;           // f
    }else if(c > clim.y ){   //-- force alingned to velocity
        cvf.x = 1-damping;   // v    // like 0.99 (weak dampong damping)
        cvf.y =   damping;   // f
    }else{                   // -- force ~ perpendicular to velocity
        float f = (c-clim.x )/( clim.y - clim.x  );
        cvf.x = (1.-damping)*f;
        cvf.y =     damping *f;
    }
    return cvf;
}
*/

/*
def KvaziFIREdamp( c, clim, damps ):
    # ----- velocity & force ~ perpendicular
    t = (c-clim[0] )/( clim[1] - clim[0]  )
    cv = damps[0] + (damps[1]-damps[0])*t
    #cf =     damps[1] *t*(1-t)*4
    cf =     damps[1]*t*(1-t)*2
    # ----- velocity & force ~ against each other
    mask_lo     =  c < clim[0]
    cv[mask_lo] = damps[0]  # v    // like 0.5 (strong damping)
    cf[mask_lo] = 0             # f
    # ----- velocity & force ~ alligned
    mask_hi     =  c > clim[1]
    cv[mask_hi] = damps[1]  # v    // like 0.99 (weak dampong damping)
    cf[mask_hi] = 0           # f
    return cv,cf
*/


// Damping function for FIRE algorithm, modified to reduction of forece and velocity arrays to make it more suitable for parallelization
float2 KvaziFIREdamp( float c, float2 clim, float2 damps ){
    float2 cvf;
    if      (c < clim.x ){   //-- force against veloctiy
        cvf.x = damps.x;     // v    // like 0.5 (strong damping)
        cvf.y = 0;           // f
    }else if(c > clim.y ){   //-- force alingned to velocity
        cvf.x = damps.y;     // v    // like 0.99 (weak dampong damping)
        cvf.y = 0;           // f
    }else{                   // -- force ~ perpendicular to velocity
        float t = (c-clim.x )/( clim.y - clim.x );
        cvf.x = damps.x + (damps.y-damps.x)*t;
        cvf.y = damps.y*t*(1.f-t)*2.f;
    }
    return cvf;
}

unsigned int hash_wang(unsigned int bits) {
    //unsigned int bits = __float_as_int(value);
    bits = (bits ^ 61) ^ (bits >> 16);
    bits *= 9;
    bits = bits ^ (bits >> 4);
    bits *= 0x27d4eb2d;
    bits = bits ^ (bits >> 15);
    return bits;
}

float hashf_wang( float val, float xmin, float xmax) {
    //return ( (float)(bits)*(2147483647.0f );
    return (((float)( hash_wang(  __float_as_int(val) ) )) * 4.6566129e-10 )  *(xmax-xmin)+ xmin;
}

// ---- updateAtomsSPFFf4: SPFF integrator with recoil, pi-orbital norm ----
// Assemble recoil forces from neighbors and  update atoms positions and velocities
//__attribute__((reqd_work_group_size(1,1,1)))
__kernel void updateAtomsSPFFf4(
    const int4        n,            // 1 // (natoms,nnode) dimensions of the system
    __global float4*  apos,         // 2 // positions of atoms  (including node atoms [0:nnode] and capping atoms [nnode:natoms] and pi-orbitals [natoms:natoms+nnode] )
    __global float4*  avel,         // 3 // velocities of atoms
    __global float4*  aforce,       // 4 // forces on atoms
    __global float4*  cvf,          // 5 // damping coefficients for velocity and force
    __global float4*  fneigh,       // 6 // recoil forces on neighbors (and pi-orbitals)
    __global int4*    bkNeighs,     // 7 // back neighbors indices (for recoil forces)
    __global float4*  constr,       // 8 // constraints (x,y,z,K) for each atom
    __global float4*  constrK,      // 9 // constraints stiffness (kx,ky,kz,?) for each atom
    __global float4*  MDparams,     // 10 // MD parameters (dt,damp,Flimit)
    __global float4*  TDrives,      // 11 // Thermal driving (T,gamma_damp,seed,?)
    __global cl_Mat3* bboxes,       // 12 // bounding box (xmin,ymin,zmin)(xmax,ymax,zmax)(kx,ky,kz)
    __global int*     sysneighs,    // 13 // // for each system contains array int[nMaxSysNeighs] of nearby other systems
    __global float4*  sysbonds      // 14 // // contains parameters of bonds (constrains) with neighbor systems   {Lmin,Lmax,Kpres,Ktens}
){
    const int natoms=n.x;           // number of atoms
    const int nnode =n.y;           // number of node atoms
    const int nMaxSysNeighs = n.w;  // max number of inter-system interactions; if <0 shwitch inter system interactions off
    const int nvec  = natoms+nnode; // number of vectors (atoms+node atoms)
    const int iG = get_global_id  (0); // index of atom

    if(iG>=nvec) return;

    const int iS = get_global_id  (1); // index of system
    const int nG = get_global_size(0); // number of atoms
    const int nS = get_global_size(1); // number of systems

    //const int ian = iG + iS*nnode;
    const int iaa = iG + iS*natoms;  // index of atom in atoms array
    const int iav = iG + iS*nvec;    // index of atom in vectors array

    const float4 MDpars  = MDparams[iS]; // (dt,damp,Flimit)
    const float4 TDrive = TDrives[iS];

    // if((iS==0)&&(iG==0)){
    //     //printf("MDpars[%i] (%g,%g,%g,%g) \n", iS, MDpars.x,MDpars.y,MDpars.z,MDpars.w);
    //     for(int is=0; is<nS; is++){
    //         //printf( "GPU::TDrives[%i](%g,%g,%g,%g)\n", i, TDrives[i].x,TDrives[i].y,TDrives[i].z,TDrives[i].w );
    //         //printf( "GPU::bboxes[%i](%g,%g,%g)(%g,%g,%g)(%g,%g,%g)\n", is, bboxes[is].a.x,bboxes[is].a.y,bboxes[is].a.z,   bboxes[is].b.x,bboxes[is].b.y,bboxes[is].b.z,   bboxes[is].c.x,bboxes[is].c.y,bboxes[is].c.z );
    //         for(int ia=0; ia<natoms; ia++){
    //             int ic = ia+is*natoms;
    //             if(constr[ia+is*natoms].w>0) printf( "GPU:sys[%i]atom[%i] constr(%g,%g,%g|%g) constrK(%g,%g,%g|%g)\n", is, ia, constr[ic].x,constr[ic].y,constr[ic].z,constr[ic].w,   constrK[ic].x,constrK[ic].y,constrK[ic].z,constrK[ic].w  );
    //         }
    //     }
    // }

    const int iS_DBG = 5; // debug system
    //const int iG_DBG = 0;
    const int iG_DBG = 1; // debug atom

    //if((iG==iG_DBG)&&(iS==iS_DBG))printf( "updateAtomsSPFFf4() natoms=%i nnode=%i nvec=%i nG %i iS %i/%i  dt=%g damp=%g Flimit=%g \n", natoms,nnode, nvec, iS, nG, nS, MDpars.x, MDpars.y, MDpars.z );
    // if((iG==iG_DBG)&&(iS==iS_DBG)){
    //     int i0a = iS*natoms;
    //     for(int i=0; i<natoms; i++){
    //         printf( "GPU:constr[%i](%7.3f,%7.3f,%7.3f |K= %7.3f) \n", i, constr[i0a+i].x,constr[i0a+i].y,constr[i0a+i].z,  constr[i0a+i].w   );
    //     }
    // }
    if(iG>=(natoms+nnode)) return; // make sure we are not out of bounds of current system

    //aforce[iav] = float4Zero;

    const float4 fe0     = aforce[iav]; // force on atom or pi-orbital (before recoil)
    float4 fe      = fe0;
    const bool bPi = iG>=natoms;  // is it pi-orbital ?

    // ------ Gather Forces from back-neighbors

    int4 ngs = bkNeighs[ iav ]; // back neighbors indices

    //if(iS==5)printf( "iG,iS %i %i ngs %i,%i,%i,%i \n", iG, iS, ngs.x,ngs.y,ngs.z,ngs.w );
    //if( (iS==0)&&(iG==0) ){ printf( "GPU:fe.1[iS=%i,iG=%i](%g,%g,%g,%g) \n", fe.x,fe.y,fe.z,fe.w ); }

    // sum all recoil forces from back neighbors   - WARRNING : bkNeighs must be properly shifted on CPU by adding offset of system iS*nvec*4
    {
    float4 frec = float4Zero;
    if(ngs.x>=0){ frec += fneigh[ngs.x]; } // if neighbor index is negative it means that there is no neighbor, so we skip it
    if(ngs.y>=0){ frec += fneigh[ngs.y]; }
    if(ngs.z>=0){ frec += fneigh[ngs.z]; }
    if(ngs.w>=0){ frec += fneigh[ngs.w]; }
    fe += frec;

    #if DBG_UFF
    if((iG==iGdbg)&&(iS==iSdbg)){
        printf("DBG updateAtomsSPFFf4(relax_multi.cl) iS=%i iG=%i iav=%i bPi=%i fe0=(%g,%g,%g|%g) frec=(%g,%g,%g|%g) fe=(%g,%g,%g|%g) ngs=(%i,%i,%i,%i)\n",
            iS,iG,iav,(int)bPi, fe0.x,fe0.y,fe0.z,fe0.w, frec.x,frec.y,frec.z,frec.w, fe.x,fe.y,fe.z,fe.w, ngs.x,ngs.y,ngs.z,ngs.w );
        if(!bPi){
            if(ngs.x>=0){ float4 t=fneigh[ngs.x]; printf("DBG updateAtomsSPFFf4(relax_multi.cl) recoil0 idx=%i fneigh=(%g,%g,%g|%g)\n", ngs.x, t.x,t.y,t.z,t.w ); }
            if(ngs.y>=0){ float4 t=fneigh[ngs.y]; printf("DBG updateAtomsSPFFf4(relax_multi.cl) recoil1 idx=%i fneigh=(%g,%g,%g|%g)\n", ngs.y, t.x,t.y,t.z,t.w ); }
            if(ngs.z>=0){ float4 t=fneigh[ngs.z]; printf("DBG updateAtomsSPFFf4(relax_multi.cl) recoil2 idx=%i fneigh=(%g,%g,%g|%g)\n", ngs.z, t.x,t.y,t.z,t.w ); }
            if(ngs.w>=0){ float4 t=fneigh[ngs.w]; printf("DBG updateAtomsSPFFf4(relax_multi.cl) recoil3 idx=%i fneigh=(%g,%g,%g|%g)\n", ngs.w, t.x,t.y,t.z,t.w ); }
        }
    }
    #endif
    }
 // ---- Limit Forces - WARNING: this can lead to drift; prefer limiting in forcefield kernels when possible
    float Flimit = MDpars.z;
    if(Flimit>0){
        float fr2 = dot(fe.xyz,fe.xyz);  // squared force
        if( fr2 > (Flimit*Flimit) ){  fe.xyz*=(Flimit/sqrt(fr2)); }  // if force is too big, we scale it down to Flimit
    }

    // =============== FORCE DONE
    aforce[iav] = fe;             // store force before limit
    //aforce[iav] = float4Zero;   // clean force   : This can be done in the first forcefield run (best is NBFF)

    // =============== DYNAMICS

    float4 ve = avel[iav]; // velocity of atom or pi-orbital
    float4 pe = apos[iav]; // position of atom or pi-orbital

    // -------- Fixed Atoms and Bounding Box
    if(iG<natoms){                  // only atoms have constraints, not pi-orbitals
        // ------- bboxes
        const cl_Mat3 B = bboxes[iS];
        // if(B.c.x>0.0f){ if(pe.x<B.a.x){ fe.x+=(B.a.x-pe.x)*B.c.x; }else if(pe.x>B.b.x){ fe.x+=(B.b.x-pe.x)*B.c.x; }; }
        // if(B.c.y>0.0f){ if(pe.y<B.a.y){ fe.y+=(B.a.y-pe.y)*B.c.y; }else if(pe.y>B.b.y){ fe.y+=(B.b.y-pe.y)*B.c.y; }; }
        if(B.c.z>0.0f){ if(pe.z<B.a.z){ fe.z+=(B.a.z-pe.z)*B.c.z; }else if(pe.z>B.b.z){ fe.z+=(B.b.z-pe.z)*B.c.z; }; }
        // ------- constrains
        float4 cons = constr[ iaa ]; // constraints (x,y,z,K)
        if( cons.w>0.f ){            // if stiffness is positive, we have constraint
            float4 cK = constrK[ iaa ];
            cK = max( cK, (float4){0.0f,0.0f,0.0f,0.0f} );
            const float3 fc = (cons.xyz - pe.xyz)*cK.xyz;
            fe.xyz += fc; // add constraint force
            // if(iS==0){printf( "GPU::constr[ia=%i|iS=%i] (%g,%g,%g|K=%g) fc(%g,%g,%g) cK(%g,%g,%g)\n", iG, iS, cons.x,cons.y,cons.z,cons.w, fc.x,fc.y,fc.z , cK.x, cK.y, cK.z ); }
        }
    }

    // -------- Inter system interactions
    if( nMaxSysNeighs>0 ){
        for(int i=0; i<nMaxSysNeighs; i++){
            const int j     = iS*nMaxSysNeighs + i;
            const int    jS = sysneighs[j];
            const float4 bj = sysbonds [j];
            const float4 pj = apos[jS*nvec + iG];
            float3 d        = pj.xyz - pe.xyz;
            float  l = length( d );
            if      (l<bj.x){
                d*=(l-bj.x)*bj.z/l;  // f = dx*kPress
            }else if(l>bj.y){
                d*=(bj.y-l)*bj.w/l;  // f = dx*kTens
            }
            fe.xyz += d;
        }
    }

    // ------ Simple damped MD (leap-frog when damp=1.0)
    if(bPi){
        fe.xyz += pe.xyz * -dot( pe.xyz, fe.xyz );   // project out radial component for pi-orbitals
        ve.xyz += pe.xyz * -dot( pe.xyz, ve.xyz );
    }
    const float dt   = MDpars.x;
    const float damp = MDpars.y;
    float inv_mass = (pe.w > 1e-8f) ? (1.0f / pe.w) : 1.0f;
    ve.xyz *= damp;
    ve.xyz += fe.xyz * dt * inv_mass;
    pe.xyz += ve.xyz * dt;
    if(bPi){
        pe.xyz=normalize(pe.xyz);                   // normalize pi-orbitals
    }
    ve.w=0;
    avel[iav] = ve;
    apos[iav] = pe;   // pe.w still holds mass
}
// ======================================================================
//                     cleanForceSPFFf4()
// ======================================================================
// Clean forces on atoms and neighbors to prepare for next forcefield evaluation
//__attribute__((reqd_work_group_size(1,1,1)))
__kernel void cleanForceSPFFf4(
    const int4        n,           // 2
    __global float4*  aforce,      // 5
    __global float4*  fneigh       // 6
){
    const int natoms=n.x;
    const int nnode =n.y;
    const int iG = get_global_id  (0);
    const int iS = get_global_id  (1);
    const int nG = get_global_size(0);
    const int nS = get_global_size(1);
    const int nvec = natoms+nnode;

    const int iav = iG + iS*nvec;
    const int ian = iG + iS*nnode;

    aforce[iav]=float4Zero;
    //aforce[iav]=(float4){iG,iS,iav,0.0};

    //if(iav==0){ printf("GPU::cleanForceSPFFf4() iS %i nG %i nS %i \n", iS, nG, nS );}
    //if(iG==0){ for(int i=0;i<(natoms+nnode);i++ ){printf("cleanForceSPFFf4[%i](%g,%g,%g)\n",i,aforce[i].x,aforce[i].y,aforce[i].z);} }
    if(iG<nnode){
        const int i4 = ian*4;
        fneigh[i4+0]=float4Zero;
        fneigh[i4+1]=float4Zero;
        fneigh[i4+2]=float4Zero;
        fneigh[i4+3]=float4Zero;
    }
    //if(iG==0){ printf( "GPU::updateAtomsSPFFf4() END\n" ); }
}

// ======================================================================
//                     updateAtomsSPFFf4_rot()
// ======================================================================
// MD integrator with rotational pi-orbital dynamics.
// Pi-orbitals are integrated as rotational DOFs (torque -> angular velocity -> rotation).
// Atoms use simple leap-frog integration.
//__attribute__((reqd_work_group_size(1,1,1)))
__kernel void updateAtomsSPFFf4_rot(
    const int4        nDOFs,            // 1 // (natoms,nnode) dimensions of the system
    __global float4*  apos,         // 2 // positions of atoms
    __global float4*  avel,         // 3 // velocities of atoms (angular velocity for pi)
    __global float4*  aforce,       // 4 // forces on atoms (torques on pi)
    __global float4*  cvf,          // 5 // damping coefficients
    __global float4*  fneigh,       // 6 // recoil forces on neighbors
    __global int4*    bkNeighs,     // 7 // back neighbors indices
    __global float4*  constr,       // 8 // constraints (x,y,z,K) for each atom
    __global float4*  constrK,      // 9 // constraints stiffness (kx,ky,kz,?) for each atom
    __global float4*  MDparams,     // 10 // MD parameters (dt,damp,Flimit)
    __global float4*  TDrives,      // 11 // Thermal driving (T,gamma_damp,seed,?)
    __global cl_Mat3* bboxes,       // 12 // bounding box
    __global int*     sysneighs,    // 13 // inter-system neighbor indices
    __global float4*  sysbonds,     // 14 // inter-system bond parameters
    __global float4*  aforce_old    // 15 // previous step forces
){
    const int natoms=nDOFs.x;
    const int nnode =nDOFs.y;
    const int nMaxSysNeighs = nDOFs.z;
    const int nvec  = natoms+nnode;
    const int iG = get_global_id  (0);
    if(iG>=nvec) return;
    const int iS = get_global_id  (1);

    const int iaa = iG + iS*natoms;
    const int iav = iG + iS*nvec;

    const float4 MDpars  = MDparams[iS]; // (dt,damp,Flimit)
    const float4 TDrive = TDrives[iS];

    if(iG>=(natoms+nnode)) return;

    float4 fe      = aforce[iav];
    const bool bPi = iG>=natoms;

    int4 ngs = bkNeighs[ iav ];

    if(!bPi){
        if(ngs.x>=0){ fe += fneigh[ngs.x]; }
        if(ngs.y>=0){ fe += fneigh[ngs.y]; }
        if(ngs.z>=0){ fe += fneigh[ngs.z]; }
        if(ngs.w>=0){ fe += fneigh[ngs.w]; }
    }

    float Flimit = MDpars.z;
    if(Flimit>0){
        float fr2 = dot(fe.xyz,fe.xyz);
        if( fr2 > (Flimit*Flimit) ){  fe.xyz*=(Flimit/sqrt(fr2)); }
    }

    aforce[iav] = fe;

    float4 ve = avel[iav];
    float4 pe = apos[iav];

    // Constraints and bounding box (atoms only)
    if(iG<natoms){
        const cl_Mat3 B = bboxes[iS];
        if(B.c.z>0.0f){ if(pe.z<B.a.z){ fe.z+=(B.a.z-pe.z)*B.c.z; }else if(pe.z>B.b.z){ fe.z+=(B.b.z-pe.z)*B.c.z; }; }
        float4 cons = constr[ iaa ];
        if( cons.w>0.f ){
            float4 cK = constrK[ iaa ];
            cK = max( cK, (float4){0.0f,0.0f,0.0f,0.0f} );
            const float3 fc = (cons.xyz - pe.xyz)*cK.xyz;
            fe.xyz += fc;
        }
    }

    // Inter-system interactions
    if( nMaxSysNeighs>0 ){
        for(int i=0; i<nMaxSysNeighs; i++){
            const int j     = iS*nMaxSysNeighs + i;
            const int    jS = sysneighs[j];
            const float4 bj = sysbonds [j];
            const float4 pj = apos[jS*nvec + iG];
            float3 d        = pj.xyz - pe.xyz;
            float  l = length( d );
            if      (l<bj.x){ d*=(l-bj.x)*bj.z/l; }
            else if (l>bj.y){ d*=(bj.y-l)*bj.w/l; }
            fe.xyz += d;
        }
    }

    const float dt   = MDpars.x;
    const float damp = MDpars.y;

    if (bPi){
        // ROTATIONAL DYNAMICS FOR PI-ORBITAL
        float inv_I  = 1.0f;
        ve.xyz *= damp;
        ve.xyz += (fe.xyz * inv_I) * dt;
        pe.xyz  = rotate_by_omega_taylor( pe.xyz, ve.xyz*dt );
        pe.xyz  = normalize(pe.xyz);
    } else {
        // LEAP-FROG FOR ATOMS with damping
        float inv_mass = (pe.w > 1e-8f) ? (1.0f / pe.w) : 1.0f;
        ve.xyz *= damp;
        ve.xyz += fe.xyz * dt * inv_mass;
        pe.xyz += ve.xyz * dt;
    }
    pe.w = 0.0f; ve.w = 0.0f;
    avel[iav] = ve;
    apos[iav] = (float4){ pe.xyz, 0.0f };
}



