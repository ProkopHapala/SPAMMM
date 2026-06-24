// nonbonded_grid.cl - GridFF-augmented non-bonded kernels + spatial bucketing
//
// Kernels:
//   - getNonBond_GridFF_Bspline_ex2: LJ/Coulomb + GridFF B-spline substrate forces
//   - getNonBond_GridFF_Bspline_tex: Same with texture-based GridFF sampling
//   - getShortRangeBuckets / getShortRangeBuckets2 / sortAtomsToBucketOverlaps
//
// Requires: common.cl + Forces.cl + gridFF.cl + surface.cl before this file.
// (make_inds_pbc, fe3d_pbc_comb, fe3d_pbc_comb_tex from gridFF.cl;
//  getR4repulsion from surface.cl)

// ---- getNonBond_GridFF_Bspline_ex2: non-bonded + GridFF B-spline ----
__kernel void getNonBond_GridFF_Bspline_ex2(
    const int4 ns,                  // 1 // dimensions of the system (natoms,nnode,nvec)
    // Dynamical
    __global float4*  atoms,        // 2 // positions of atoms
    __global float4*  forces,       // 3 // forces on atoms
    // Parameters
    __global float4*  REQKs,        // 4 // parameters of Lenard-Jones potential, Coulomb and Hydrogen Bond (RvdW,EvdW,Q,H)
    //__global int4*    neighs,       // 5 // indexes neighbors of atoms
    //__global int4*    neighCell,    // 6 // indexes of cells of neighbor atoms
        __global int*  excl,         // 5 // packed sorted exclusion list ()   
    __global cl_Mat3*  lvecs,        // 7 // lattice vectors of the system
    const int4 nPBC,                // 8 // number of PBC images in each direction
    const float4  GFFParams,        // 9 // parameters of Grid-Force-Field (GFF) (RvdW,EvdW,Q,H)
    // GridFF
    __global float4*  BsplinePLQ,   // 10 // Grid-Force-Field (GFF) for Pauli repulsion
    const int4     grid_ns,         // 11 // origin of the grid
    const float4   grid_invStep,    // 12 // origin of the grid
    const float4   grid_p0          // 13 // origin of the grid
){
    __local float4 LATOMS[32];         // local memory chumk of positions of atoms
    __local float4 LCLJS [32];         // local memory chumk of atom parameters
    const int iG = get_global_id  (0); // index of atom in the system
    const int iS = get_global_id  (1); // index of system
    const int iL = get_local_id   (0); // index of atom in the local memory chunk
    const int nG = get_global_size(0); // total number of atoms in the system
    const int nS = get_global_size(1); // total number of systems
    const int nL = get_local_size (0); // number of atoms in the local memory chunk

    const int natoms=ns.x;         // number of atoms in the system
    const int nnode =ns.y;         // number of nodes in the system
    const int nvec  =natoms+nnode; // number of vectos (atoms and pi-orbitals) in the system

    //const int i0n = iS*nnode;    // index of the first node in the system
    const int i0a = iS*natoms;     // index of the first atom in the system
    const int i0v = iS*nvec;       // index of the first vector (atom or pi-orbital) in the system
    //const int ian = iG + i0n;    // index of the atom in the system
    const int iaa = iG + i0a;      // index of the atom in the system
    const int iav = iG + i0v;      // index of the vector (atom or pi-orbital) in the system

    const float4 REQKi = REQKs    [iaa];           // parameters of Lenard-Jones potential, Coulomb and Hydrogen Bond (RvdW,EvdW,Q,H) of the atom
    const float3 posi  = atoms    [iav].xyz;       // position of the atom
    float4 fe          = float4Zero;              // forces on the atom

    const int iS_DBG = 0;
    const int iG_DBG = 0;

    // =================== Non-Bonded interaction ( molecule-molecule )

    { // insulate nbff

    const cl_Mat3 lvec = lvecs[iS]; // lattice vectors of the system

    //if((iG==iG_DBG)&&(iS==iS_DBG)){  printf( "GPU::getNonBond_GridFF_Bspline() natoms,nnode,nvec(%i,%i,%i) nS,nG,nL(%i,%i,%i) \n", natoms,nnode,nvec, nS,nG,nL ); }
    //if((iG==iG_DBG)&&(iS==iS_DBG)) printf( "GPU::getNonBond_GridFF_Bspline() nPBC_(%i,%i,%i) lvec (%g,%g,%g) (%g,%g,%g) (%g,%g,%g)\n", nPBC.x,nPBC.y,nPBC.z, lvec.a.x,lvec.a.y,lvec.a.z,  lvec.b.x,lvec.b.y,lvec.b.z,   lvec.c.x,lvec.c.y,lvec.c.z );
    // if((iG==iG_DBG)&&(iS==iS_DBG)){
    //     printf( "GPU::getNonBond_GridFF_Bspline() natoms,nnode,nvec(%i,%i,%i) nS,nG,nL(%i,%i,%i) \n", natoms,nnode,nvec, nS,nG,nL );
    //     for(int i=0; i<nS*nG; i++){
    //         int ia = i%nS;
    //         int is = i/nS;
    //         if(ia==0){ cl_Mat3 lvec = lvecs[is];  printf( "GPU[%i] lvec(%6.3f,%6.3f,%6.3f)(%6.3f,%6.3f,%6.3f)(%6.3f,%6.3f,%6.3f) \n", is, lvec.a.x,lvec.a.y,lvec.a.z,  lvec.b.x,lvec.b.y,lvec.b.z,   lvec.c.x,lvec.c.y,lvec.c.z  ); }
    //         //printf( "GPU[%i,%i] \n", is,ia,  );
    //     }
    // }

    //if(iG>=natoms) return;

    //const bool   bNode = iG<nnode;   // All atoms need to have neighbors !!!!
    const bool   bPBC  = (nPBC.x+nPBC.y+nPBC.z)>0; // Periodic boundary conditions if any of nPBC.x,nPBC.y,nPBC.z is non-zero
    const float  R2damp = GFFParams.x*GFFParams.x; // damping radius for Lenard-Jones potential

    //if(iG==0){ for(int i=0; i<natoms; i++)printf( "GPU[%i] ng(%i,%i,%i,%i) REQ(%g,%g,%g) \n", i, neighs[i].x,neighs[i].y,neighs[i].z,neighs[i].w, REQKs[i].x,REQKs[i].y,REQKs[i].z ); }

    const float3 shift0  = lvec.a.xyz*nPBC.x + lvec.b.xyz*nPBC.y + lvec.c.xyz*nPBC.z;  // shift of the first PBC image
    const float3 shift_a = lvec.b.xyz + lvec.a.xyz*(nPBC.x*-2.f-1.f);                  // shift of lattice vector in the inner loop
    const float3 shift_b = lvec.c.xyz + lvec.b.xyz*(nPBC.y*-2.f-1.f);                  // shift of lattice vector in the outer loop

      //const int excl_base = iaa*EXCL_MAX;
    int iex             = iaa*EXCL_MAX;
    const int iex_end   = iex + EXCL_MAX-1;
    int jex             = excl[iex];

    // if((iG==0)&&(iS==0)){
    //     printf( "getNonBond_GridFF_Bspline_ex2() iG %i, iS %i, iaa %i, iex %i, iex_end %i, jex %i\n", iG, iS, iaa, iex, iex_end, jex );
    // }

    // ========= Atom-to-Atom interaction ( N-body problem ), we do it in chunks of size of local memory, in order to reuse data and reduce number of reads from global memory  
    //barrier(CLK_LOCAL_MEM_FENCE);
    for (int j0=0; j0<nG; j0+=nL){     // loop over all atoms in the system, by chunks of size of local memory
        const int i=j0+iL;             // index of atom in local memory
        if(i<natoms){                  // j0*nL may be larger than natoms, so we need to check if we are not reading from invalid address
            LATOMS[iL] = atoms[i+i0v]; // read atom position to local memory 
            LCLJS [iL] = REQKs[i+i0a]; // read atom parameters to local memory
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
                        float4 fij = getLJQH( dp, REQK, R2damp ); 
                        fe += fij;
                    }
                }
            }
        }
        //barrier(CLK_LOCAL_MEM_FENCE);
    }

    } // insulate nbff

    if(iG>=natoms) return; // natoms <= nG, because nG must be multiple of nL (loccal kernel size). We cannot put this check at the beginning of the kernel, because it will break reading of atoms to local memory

    // ========== Molecule-Grid interaction with GridFF using tricubic Bspline ================== (see. kernel sample3D_comb() in GridFF.cl

    __local int4 xqs[4];
    __local int4 yqs[4];
    { // insulate gridff
        if      (iL<4){             xqs[iL]=make_inds_pbc(grid_ns.x,iL); }
        else if (iL<8){ int i=iL-4; yqs[i ]=make_inds_pbc(grid_ns.y,i ); };
        //const float3 inv_dg = 1.0f / grid_d.xyz;
        barrier(CLK_LOCAL_MEM_FENCE);

        const float ej = exp( GFFParams.y * REQKi.x ); // exp(-alphaMorse*RvdW) pre-factor for factorized Morse potential
        const float4 PLQH = (float4){
            ej*ej*REQKi.y,                   // prefactor London dispersion (attractive part of Morse potential)
            ej*   REQKi.y,                   // prefactor Pauli repulsion   (repulsive part of Morse potential)
            REQKi.z,
            0.0f
        };
        //const float3 p = ps[iG].xyz;
        const float3 u = (posi - grid_p0.xyz) * grid_invStep.xyz;

        float4 fg = fe3d_pbc_comb(u, grid_ns.xyz, BsplinePLQ, PLQH, xqs, yqs);

        //if((iG==iG_DBG)&&(iS==iS_DBG)){  printf( "GPU::getNonBond_GridFF_Bspline() fg(%g,%g,%g|%g) u(%g,%g,%g) posi(%g,%g,%g) grid_invStep(%g,%g,%g)\n", fg.x,fg.y,fg.z,fg.w,  u.x,u.y,u.z, posi.x,posi.y,posi.z, grid_invStep.x, grid_invStep.y, grid_invStep.z  ); }

        fg.xyz *= -grid_invStep.xyz;
        fe += fg;
        //fes[iG] = fe;
    }  // insulate gridff

    forces[iav] = fe;        // If we do    run it as first forcefield, in this case we do not need to clear forces before running this forcefield
    //forces[iav] += fe;     // If we don't run it as first forcefield, we need to add forces to the forces calculated by previous forcefields
    //forces[iav] = fe*(-1.f);


}





// ---- getNonBond_GridFF_Bspline_tex: texture-based GridFF variant ----
__kernel void getNonBond_GridFF_Bspline_tex( // Renamed kernel to distinguish from buffer version
    const int4 ns,                  // 1 // dimensions of the system (natoms,nnode,nvec)
    // Dynamical
    __global float4*  atoms,        // 2 // positions of atoms
    __global float4*  forces,       // 3 // forces on atoms
    // Parameters
    __global float4*  REQKs,        // 4 // parameters of Lenard-Jones potential, Coulomb and Hydrogen Bond (RvdW,EvdW,Q,H)
    __global int4*    neighs,       // 5 // indexes neighbors of atoms
    __global int4*    neighCell,    // 6 // indexes of cells of neighbor atoms
    __global cl_Mat3* lvecs,        // 7 // lattice vectors of the system
    const int4 nPBC,                // 8 // number of PBC images in each direction
    const float4  GFFParams,        // 9 // parameters of Grid-Force-Field (GFF) (RvdW_cutoff_factor_for_LJ, alphaMorse, Q_atom, H_bond_params_unused)
    // GridFF - Using Texture
    __read_only image3d_t BsplinePLQH_tex, // 10 // Grid-Force-Field (GFF) data (Pauli,London,Coulomb,HBond) in a 3D texture (Renamed texture)
    const int4     grid_ns,         // 11 // dimensions of the grid (matches buffer code name)
    const float4   grid_invStep,    // 12 // inverse of grid cell dimensions
    const float4   grid_p0          // 13 // origin of the grid
){
    __local float4 LATOMS[32];         // local memory chumk of positions of atoms
    __local float4 LCLJS [32];         // local memory chumk of atom parameters
    const int iG = get_global_id  (0); // index of atom in the system
    const int iS = get_global_id  (1); // index of system
    const int iL = get_local_id   (0); // index of atom in the local memory chunk
    const int nG = get_global_size(0); // total number of atoms in the system
    const int nS = get_global_size(1); // total number of systems
    const int nL = get_local_size (0); // number of atoms in the local memory chunk

    const int natoms=ns.x;         // number of atoms in the system
    const int nnode =ns.y;         // number of nodes in the system
    const int nvec  =natoms+nnode; // number of vectos (atoms and pi-orbitals) in the system

    //const int i0n = iS*nnode;    // index of the first node in the system
    const int i0a = iS*natoms;     // index of the first atom in the system
    const int i0v = iS*nvec;       // index of the first vector (atom or pi-orbital) in the system
    //const int ian = iG + i0n;    // index of the atom in the system
    const int iaa = iG + i0a;      // index of the atom in the system
    const int iav = iG + i0v;      // index of the vector (atom or pi-orbital) in the system

    const float4 REQKi = REQKs    [iaa];           // parameters of Lenard-Jones potential, Coulomb and Hydrogen Bond (RvdW,EvdW,Q,H) of the atom
    const float3 posi  = atoms    [iav].xyz;       // position of the atom
    float4 fe          = float4Zero;              // forces on the atom

    const int iS_DBG = 0;
    const int iG_DBG = 0;

    // =================== Non-Bonded interaction ( molecule-molecule )

    { // insulate nbff

    const cl_Mat3 lvec = lvecs[iS]; // lattice vectors of the system

    //if((iG==iG_DBG)&&(iS==iS_DBG)){  printf( "GPU::getNonBond_GridFF_Bspline() natoms,nnode,nvec(%i,%i,%i) nS,nG,nL(%i,%i,%i) \n", natoms,nnode,nvec, nS,nG,nL ); }
    //if((iG==iG_DBG)&&(iS==iS_DBG)) printf( "GPU::getNonBond_GridFF_Bspline() nPBC_(%i,%i,%i) lvec (%g,%g,%g) (%g,%g,%g) (%g,%g,%g)\n", nPBC.x,nPBC.y,nPBC.z, lvec.a.x,lvec.a.y,lvec.a.z,  lvec.b.x,lvec.b.y,lvec.b.z,   lvec.c.x,lvec.c.y,lvec.c.z );
    // if((iG==iG_DBG)&&(iS==iS_DBG)){
    //     printf( "GPU::getNonBond_GridFF_Bspline() natoms,nnode,nvec(%i,%i,%i) nS,nG,nL(%i,%i,%i) \n", natoms,nnode,nvec, nS,nG,nL );
    //     for(int i=0; i<nS*nG; i++){
    //         int ia = i%nS;
    //         int is = i/nS;
    //         if(ia==0){ cl_Mat3 lvec = lvecs[is];  printf( "GPU[%i] lvec(%6.3f,%6.3f,%6.3f)(%6.3f,%6.3f,%6.3f)(%6.3f,%6.3f,%6.3f) \n", is, lvec.a.x,lvec.a.y,lvec.a.z,  lvec.b.x,lvec.b.y,lvec.b.z,   lvec.c.x,lvec.c.y,lvec.c.z  ); }
    //         //printf( "GPU[%i,%i] \n", is,ia,  );
    //     }
    // }

    //if(iG>=natoms) return;

    //const bool   bNode = iG<nnode;   // All atoms need to have neighbors !!!!
    const bool   bPBC  = (nPBC.x+nPBC.y+nPBC.z)>0; // Periodic boundary conditions if any of nPBC.x,nPBC.y,nPBC.z is non-zero
    const int4   ng    = neighs   [iaa];           // indexes of neighbors of the atom
    const int4   ngC   = neighCell[iaa];           // indexes of cells of neighbors of the atom

    const float  R2damp = GFFParams.x*GFFParams.x; // damping radius for Lenard-Jones potential

    //if(iG==0){ for(int i=0; i<natoms; i++)printf( "GPU[%i] ng(%i,%i,%i,%i) REQ(%g,%g,%g) \n", i, neighs[i].x,neighs[i].y,neighs[i].z,neighs[i].w, REQKs[i].x,REQKs[i].y,REQKs[i].z ); }

    const float3 shift0  = lvec.a.xyz*nPBC.x + lvec.b.xyz*nPBC.y + lvec.c.xyz*nPBC.z;  // shift of the first PBC image
    const float3 shift_a = lvec.b.xyz + lvec.a.xyz*(nPBC.x*-2.f-1.f);                  // shift of lattice vector in the inner loop
    const float3 shift_b = lvec.c.xyz + lvec.b.xyz*(nPBC.y*-2.f-1.f);                  // shift of lattice vector in the outer loop

    // ========= Atom-to-Atom interaction ( N-body problem )     - we do it by chunks of nL atoms in order to reuse data and reduce number of global memory reads
    for (int j0=0; j0<natoms; j0+= nL ){ // loop over atoms in the system by chunks of nL atoms which fit into local memory
        const int i = j0 + iL;           // global index of atom in the system
        LATOMS[iL] = atoms [i+i0v];      // load positions  of atoms into local memory
        LCLJS [iL] = REQKs [i+i0a];      // load parameters of atoms into local memory
        barrier(CLK_LOCAL_MEM_FENCE);    // wait until all atoms are loaded into local memory
        for (int jl=0; jl<nL; jl++){     // loop over atoms in the local memory chunk
            const int ja=jl+j0;          // global index of atom in the system
            if( (ja!=iG) && (ja<natoms) ){ // atom should not interact with himself, and should be in the system ( j0*nL+iL may be out of range of natoms )
                const float4 aj   = LATOMS[jl]; // position of the atom
                float4       REQK = LCLJS [jl]; // parameters of the atom
                float3 dp   = aj.xyz - posi;    // vector between atoms
                REQK.x  +=REQKi.x;              // mixing of RvdW radii
                REQK.yz *=REQKi.yz;             // mixing of EvdW and Q
                const bool bBonded = ((ja==ng.x)||(ja==ng.y)||(ja==ng.z)||(ja==ng.w));   // atom is bonded if it is one of the neighbors
                if(bPBC){       // ==== with periodic boundary conditions we need to consider all PBC images of the atom
                    int ipbc=0; // index of PBC image
                    //if( (i0==0)&&(j==0)&&(iG==0) )printf( "pbc NONE dp(%g,%g,%g)\n", dp.x,dp.y,dp.z );
                    dp -= shift0;  // shift to the first PBC image
                    for(int iz=-nPBC.z; iz<=nPBC.z; iz++){
                        for(int iy=-nPBC.y; iy<=nPBC.y; iy++){
                            for(int ix=-nPBC.x; ix<=nPBC.x; ix++){
                                if( !( bBonded &&(  // if bonded in any of PBC images, then we have to check both index of atom and index of PBC image to decide if we should skip this interaction
                                          ((ja==ng.x)&&(ipbc==ngC.x)) // check 1. neighbor and its PBC cell
                                        ||((ja==ng.y)&&(ipbc==ngC.y)) // check 2. neighbor and its PBC cell
                                        ||((ja==ng.z)&&(ipbc==ngC.z)) // ...
                                        ||((ja==ng.w)&&(ipbc==ngC.w)) // ...
                                ))){
                                    //fe += getMorseQ( dp+shifts, REQK, R2damp );
                                    float4 fij = getLJQH( dp, REQK, R2damp );  // calculate Lenard-Jones, Coulomb and Hydrogen-bond forces between atoms
                                    //if((iG==iG_DBG)&&(iS==iS_DBG)){  printf( "GPU_LJQ[%i,%i|%i] fj(%g,%g,%g) R2damp %g REQ(%g,%g,%g) r %g \n", iG,ji,ipbc, fij.x,fij.y,fij.z, R2damp, REQK.x,REQK.y,REQK.z, length(dp+shift)  ); }
                                    fe += fij; // accumulate forces
                                }
                                ipbc++;         // increment index of PBC image
                                dp+=lvec.a.xyz; // shift to the next PBC image
                            }
                            dp+=shift_a;        // shift to the next PBC image
                        }
                        dp+=shift_b;            // shift to the next PBC image
                    }
                }else{ //  ==== without periodic boundary it is much simpler, not need to care about PBC images
                    if(bBonded) continue;  // Bonded ?
                    float4 fij = getLJQH( dp, REQK, R2damp ); // calculate Lenard-Jones, Coulomb and Hydrogen-bond forces between atoms
                    fe += fij;
                    //if((iG==iG_DBG)&&(iS==iS_DBG)){  printf( "GPU_LJQ[%i,%i] fj(%g,%g,%g) R2damp %g REQ(%g,%g,%g) r %g \n", iG,ji, fij.x,fij.y,fij.z, R2damp, REQK.x,REQK.y,REQK.z, length(dp)  ); }
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE); // wait until all atoms are processed, ToDo: not sure if it is needed here ?
    }

    } // insulate nbff

    if(iG>=natoms) return; // natoms <= nG, because nG must be multiple of nL (loccal kernel size). We cannot put this check at the beginning of the kernel, because it will break reading of atoms to local memory

    // ========== Molecule-Grid interaction with GridFF using tricubic Bspline (Texture based) ==================

    __local int4 xqs[4]; // Local memory for PBC index patterns for X
    __local int4 yqs[4]; // Local memory for PBC index patterns for Y

    { // insulate gridff
        // Initialize local memory for PBC index patterns. Only first 8 work-items do this.
        if      (iL<4){ xqs[iL]=make_inds_pbc(grid_ns.x,iL); }
        else if (iL<8){ yqs[iL-4]=make_inds_pbc(grid_ns.y,iL-4); };
        barrier(CLK_LOCAL_MEM_FENCE); // Ensure local memory is populated

        // Coefficients for combining Pauli, London, Coulomb, H-bond from the grid field
        // Matches original calculation using GFFParams.y (alphaMorse) and REQKi
        const float alphaMorse = GFFParams.y;
        const float ej = exp( alphaMorse * REQKi.x ); // REQKi.x is RvdW of atom_i
        const float4 PLQH = (float4){
            ej*ej*REQKi.y,                   // Pauli coeff: EvdW_i * exp(2 * alphaMorse * RvdW_i)
            ej*   REQKi.y,                   // London coeff: EvdW_i * exp(alphaMorse * RvdW_i)
            REQKi.z,                         // Coulomb coeff: Q_i
            0.0f                             // H-bond coeff (assuming zeroed out)
        };

        // Calculate normalized coordinates 'u' for B-spline interpolation
        const float3 u = (posi - grid_p0.xyz) * grid_invStep.xyz;

        // Perform 3D B-spline interpolation using texture
        // fg contains (dE/dux, dE/duy, dE/duz, Energy)
        float4 fg = fe3d_pbc_comb_tex(u, grid_ns.xyz, BsplinePLQH_tex, sampler_bspline, PLQH, xqs, yqs);

        #if DBG_UFF
        if( (iG==0) && (iS==0) ){
            printf("DBG GridFF_Bspline tex pos(%g,%g,%g) u(%g,%g,%g) grid_p0(%g,%g,%g) invStep(%g,%g,%g) REQKi(%g,%g,%g,%g) PLQH(%g,%g,%g,%g) fg_raw(%g,%g,%g,%g)\n",
                posi.x,posi.y,posi.z, u.x,u.y,u.z, grid_p0.x,grid_p0.y,grid_p0.z, grid_invStep.x,grid_invStep.y,grid_invStep.z,
                REQKi.x,REQKi.y,REQKi.z,REQKi.w, PLQH.x,PLQH.y,PLQH.z,PLQH.w, fg.x,fg.y,fg.z,fg.w );
        }
        #endif

        fg.xyz *= -grid_invStep.xyz; // dux/dx = grid_invStep.x, etc.   Fx = -dE/dx = - (dE/dux) * (dux/dx)

        fe += fg; // Add GridFF force and energy to atom's total
        // fes[iG] = fe; // If you have a separate energy buffer
    }  // insulate gridff

    // Store the total force and energy for this atom
    forces[iav] = fe;
    // Use forces[iav] += fe; if forces buffer accumulates from multiple kernels
}



