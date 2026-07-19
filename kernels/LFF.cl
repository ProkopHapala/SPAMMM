/// @file LFF.cl
/// @brief Projective Jacobi on UFF-derived distance springs — soft FAF/E-field outer, hard geometry inner.
///
/// Motivation: soft substrate and stiff bonds act on the **same** atoms; mass scaling cannot
/// accelerate soft motion without soft-bonding the hard terms. LFF treats hard terms as springs
/// projected with **M/dt²** so the outer soft step can use a large `dt`.
///
/// Design:
/// - One workgroup = one molecule (`LFF_WG_SIZE`); threads beyond `natoms` gate work but hit barriers.
/// - Packed `neighs`/`KLs` (MAX_NEIGHBORS=24 for PAH K12+K13+K14; FireCore's 8 is too small).
/// - FAF helpers inlined (standalone program; not concatenated with surface.cl).
///
/// Landmine: K14 from V/(dl/dφ)² blows up near planar PAH — host must cap K and set l0 from geometry.
/// Topic: doc/Topics/ForceFields/LFF_ProjectiveRelax.md

#define COULOMB_CONST   14.3996448915f
#define _R2damp         1.e-8f

#ifndef LFF_WG_SIZE
#define LFF_WG_SIZE     64
#endif
#ifndef MAX_NEIGHBORS
#define MAX_NEIGHBORS   24
#endif
#ifndef FAF_BASIS_MAX
#define FAF_BASIS_MAX   128
#endif
#ifndef FAF_TYPES_MAX
#define FAF_TYPES_MAX   8
#endif

inline float4 getLJQH( float3 dp, float4 REQ, float R2damp ){
    float   r2    = dot(dp,dp);
    float   ir2_  = 1.f/(  r2 +  R2damp);
    float   Ec    =  COULOMB_CONST*REQ.z*sqrt( ir2_ );
    float  ir2 = 1.f/r2;
    float  u2  = REQ.x*REQ.x*ir2;
    float  u6  = u2*u2*u2;
    float vdW  = u6*REQ.y;
    float E    =       (u6-2.f)*vdW     + Ec  ;
    float fr   = -12.f*(u6-1.f)*vdW*ir2 - Ec*ir2_;
    return  (float4){ dp*fr, E };
}

inline float folded_eval_basis_lff(float u, float v, float z, float4 prm){
    const float twopi = 6.283185307179586f;
    return native_cos(twopi*prm.x*u) * native_cos(twopi*prm.y*v) * native_exp(-prm.z * fmax(0.0f, z - prm.w));
}
inline float3 folded_eval_grad_lff(float u, float v, float z, float4 prm, float4 invL){
    const float twopi = 6.283185307179586f;
    float ku=prm.x, kv=prm.y, az=prm.z, z0=prm.w;
    float bx=native_cos(twopi*ku*u), by=native_cos(twopi*kv*v);
    float bz=native_exp(-az*fmax(0.0f,z-z0));
    float dEdu=(-twopi*ku*native_sin(twopi*ku*u))*by*bz;
    float dEdv=bx*(-twopi*kv*native_sin(twopi*kv*v))*bz;
    float dEdz=(z>=z0)?(bx*by*(-az*bz)):0.0f;
    return (float3)(dEdu*invL.x+dEdv*invL.z, dEdu*invL.y+dEdv*invL.w, dEdz);
}

// Projective Jacobi on springs; optional FAF soft force in outer loop.
__kernel void lff_jacobi(
    __global const int*    mols,
    __global       float4* apos,
    __global       float4* avel,
    __global const int*    neighs,
    __global const float2* KLs,
    __global const int*    fixed_mask,
    const float3           Efield,
    const float            dt,
    const int              nOuter,
    const int              nInner,
    const float            bMix,
    const float            damp,
    const int              do_faf,
    __global const float*  folded_coeffs,
    __global const float4* folded_kxyz,
    __global const int*    folded_atom_type,
    const int4             folded_meta,
    const float4           folded_lvec2d
){
    const int imol = get_group_id(0);
    const int isys = get_group_id(1);
    const int lid  = get_local_id(0);
    const int lsz  = get_local_size(0);

    const int nMolPerSys  = get_num_groups(0);
    const int atomsPerSys = mols[nMolPerSys];
    const int sys_atom0   = isys * atomsPerSys;

    const int ia0    = sys_atom0 + mols[imol    ];
    const int ia1    = sys_atom0 + mols[imol + 1];
    const int natoms = ia1 - ia0;
    const int ok = (lid < natoms && natoms <= LFF_WG_SIZE) ? 1 : 0;

    __local float4 lpos[LFF_WG_SIZE];
    __local float4 LBASIS[FAF_BASIS_MAX];
    __local float  LCOEFFS[FAF_TYPES_MAX * FAF_BASIS_MAX];

    float3 pi = (float3)(0);
    float3 vi = (float3)(0);
    float  mi = 1.0f;
    float  Qi = 0.0f;
    int idx = ia0;
    int neigh_base = 0;
    int is_fixed = 0;
    float inv_mass = 1.0f;
    float inv_dt = 1.0f / fmax(dt, 1e-12f);
    float inv_dt2 = inv_dt * inv_dt;
    float Ii = inv_dt2;
    int nneigh = 0;
    int    ng_idx[MAX_NEIGHBORS];
    float2 ng_KLs[MAX_NEIGHBORS];
    float4 invLvec2d = (float4)(0);
    int nbasis = 0, ntypes = 0;

    if(ok){
        idx = ia0 + lid;
        neigh_base = idx * MAX_NEIGHBORS;
        pi = apos[idx].xyz;
        vi = avel[idx].xyz;
        mi = fmax(avel[idx].w, 1e-8f);
        Qi = apos[idx].w;
        lpos[lid] = (float4){pi, mi};
        inv_mass = 1.0f / mi;
        Ii = mi * inv_dt2;
        is_fixed = fixed_mask[idx] != 0;
        for (int jj = 0; jj < MAX_NEIGHBORS; ++jj) {
            const int j = neighs[neigh_base + jj];
            ng_idx[jj] = j;
            ng_KLs[jj] = KLs[neigh_base + jj];
            if (j < 0) break;
            ++nneigh;
        }
    }
    if(do_faf){
        nbasis = folded_meta.x; ntypes = folded_meta.y;
        if(nbasis>0 && nbasis<=FAF_BASIS_MAX && ntypes>0 && ntypes<=FAF_TYPES_MAX){
            for(int j=lid; j<nbasis; j+=lsz) LBASIS[j]=folded_kxyz[j];
            for(int j=lid; j<nbasis*ntypes; j+=lsz) LCOEFFS[j]=folded_coeffs[j];
            float ax=folded_lvec2d.x, bx=folded_lvec2d.y, ay=folded_lvec2d.z, by=folded_lvec2d.w;
            float det=ax*by-bx*ay;
            if(fabs(det)>1e-12f) invLvec2d=(float4)(by/det,-bx/det,-ay/det,ax/det);
            else nbasis=0;
        } else nbasis=0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int outer = 0; outer < nOuter; ++outer) {
        float3 pi_old = pi;
        float3 fex = (float3)(0.0f, 0.0f, 0.0f);
        if(ok && !is_fixed){
            fex = Efield * Qi;
            if(do_faf && nbasis>0){
                float3 pos = pi;
                float u = invLvec2d.x*pos.x + invLvec2d.y*pos.y;
                float v = invLvec2d.z*pos.x + invLvec2d.w*pos.y;
                u -= floor(u); v -= floor(v);
                int ityp = folded_atom_type[idx];
                if(ityp>=0 && ityp<ntypes){
                    float3 F=(float3)(0,0,0);
                    int ioff=ityp*nbasis;
                    for(int ib=0; ib<nbasis; ib++){
                        float c=LCOEFFS[ioff+ib]; float4 prm=LBASIS[ib];
                        F -= c*folded_eval_grad_lff(u,v,pos.z,prm,invLvec2d);
                    }
                    fex += F;
                }
            }
            vi *= damp;
            vi += fex * (dt * inv_mass);
            pi += vi * dt;
            lpos[lid] = (float4){pi, mi};
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        float3 mom_vec = (float3)(0.0f, 0.0f, 0.0f);
        for (int iter = 0; iter < nInner; ++iter) {
            if(ok && !is_fixed){
                float3  bi = pi * Ii;
                float  Aii = Ii;
                for (int jj=0; jj<nneigh; ++jj){
                    int j = ng_idx[jj];
                    if (j < 0) break;
                    float2 kl = ng_KLs[jj];
                    int jloc = j - ia0;
                    if(jloc<0 || jloc>=natoms) continue;
                    float3 pj = lpos[jloc].xyz;
                    float3 dij = pi - pj;
                    float len = length(dij);
                    float inv_len = len > 1e-8f ? 1.0f/len : 0.0f;
                    float3 rest_pos = pj + dij * (kl.y * inv_len);
                    bi  += rest_pos * kl.x;
                    Aii += kl.x;
                }
                float3 pi_new = bi * (1.0f / Aii);
                if(bMix > 1e-8f){
                    float3 pi_ = pi_new + mom_vec * bMix;
                    mom_vec = pi_ - pi;
                    pi = pi_;
                } else {
                    pi = pi_new;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            if(ok) lpos[lid] = (float4)(pi, mi);
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if(ok && !is_fixed){
            vi = (pi - pi_old) * inv_dt;
            apos[idx] = (float4)(pi, Qi);
            avel[idx] = (float4)(vi, mi);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}


__kernel void lff_nb_jacobi(
    __global const int*    mols,
    __global       float4* apos,
    __global       float4* avel,
    __global       float4* REQHs,
    __global const int*    neighs,
    __global const float2* KLs,
    __global const int*    fixed_mask,
    const float3           Efield,
    const float            dt,
    const int              nOuter,
    const int              nInner,
    const float            bMix
){
    // Legacy NB path kept for FireCore parity; prefer lff_jacobi + FAF for adsorbates.
    const int imol = get_group_id(0);
    const int isys = get_group_id(1);
    const int lid  = get_local_id(0);
    const int lsz  = get_local_size(0);
    const int nMolPerSys  = get_num_groups(0);
    const int atomsPerSys = mols[nMolPerSys];
    const int sys_atom0   = isys * atomsPerSys;
    const int ia0    = sys_atom0 + mols[imol    ];
    const int ia1    = sys_atom0 + mols[imol + 1];
    const int natoms = ia1 - ia0;
    const int ok = (lid < natoms && natoms <= LFF_WG_SIZE) ? 1 : 0;

    __local float4 lpos[LFF_WG_SIZE];
    __local float4 lREQ[LFF_WG_SIZE];

    float3 pi=(float3)(0); float3 vi=(float3)(0); float mi=1.f; float Qi=0.f;
    int idx=ia0, neigh_base=0, is_fixed=0, nneigh=0;
    float inv_mass=1.f, inv_dt=1.f/fmax(dt,1e-12f), Ii=inv_dt*inv_dt;
    int ng_idx[MAX_NEIGHBORS]; float2 ng_KLs[MAX_NEIGHBORS];
    float4 REQKi = (float4)(0);

    if(ok){
        idx=ia0+lid; neigh_base=idx*MAX_NEIGHBORS;
        pi=apos[idx].xyz; vi=avel[idx].xyz; mi=fmax(avel[idx].w,1e-8f); Qi=apos[idx].w;
        REQKi=REQHs[idx]; lpos[lid]=(float4){pi,mi};
        inv_mass=1.f/mi; Ii=mi*inv_dt*inv_dt; is_fixed=fixed_mask[idx]!=0;
        for(int jj=0;jj<MAX_NEIGHBORS;++jj){ int j=neighs[neigh_base+jj]; ng_idx[jj]=j; ng_KLs[jj]=KLs[neigh_base+jj]; if(j<0)break; ++nneigh; }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for(int outer=0; outer<nOuter; ++outer){
        float4 fe=(float4)(0);
        if(ok && !is_fixed){
            fe.xyz = Efield*Qi;
            for(int j0=0;j0<atomsPerSys;j0+=LFF_WG_SIZE){
                const int chunkStart=sys_atom0+j0;
                const int jLoad=chunkStart+lid;
                if(jLoad<sys_atom0+atomsPerSys){ lpos[lid]=apos[jLoad]; lREQ[lid]=REQHs[jLoad]; }
                else { lpos[lid]=(float4)(0); lREQ[lid]=(float4)(0); }
                barrier(CLK_LOCAL_MEM_FENCE);
                const int chunkCount=(j0+LFF_WG_SIZE<=atomsPerSys)?LFF_WG_SIZE:(atomsPerSys-j0);
                for(int jl=0;jl<chunkCount;jl++){
                    const int ja=chunkStart+jl;
                    if(ja==idx) continue;
                    if((ja>=ia0)&&(ja<ia1)){
                        bool bonded=false;
                        for(int jj=0;jj<nneigh;jj++){ if(ng_idx[jj]==ja){bonded=true;break;} }
                        if(bonded) continue;
                    }
                    float4 REQK=lREQ[jl]; float3 dp=lpos[jl].xyz-pi;
                    REQK.x+=REQKi.x; REQK.yz*=REQKi.yz;
                    fe += getLJQH(dp, REQK, _R2damp);
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            lpos[lid]=(float4){pi,mi};
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        float3 pi_old=pi;
        if(ok && !is_fixed){
            vi += fe.xyz*(dt*inv_mass);
            pi += vi*dt;
            lpos[lid]=(float4){pi,mi};
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for(int iter=0; iter<nInner; ++iter){
            if(ok && !is_fixed){
                float3 bi=pi*Ii; float Aii=Ii;
                for(int jj=0;jj<nneigh;++jj){
                    int j=ng_idx[jj]; if(j<0)break;
                    float2 kl=ng_KLs[jj]; int jloc=j-ia0; if(jloc<0||jloc>=natoms)continue;
                    float3 pj=lpos[jloc].xyz; float3 dij=pi-pj; float len=length(dij);
                    float inv_len=len>1e-8f?1.f/len:0.f;
                    bi += (pj + dij*(kl.y*inv_len))*kl.x; Aii+=kl.x;
                }
                pi = bi*(1.f/Aii);
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            if(ok) lpos[lid]=(float4)(pi,mi);
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if(ok && !is_fixed){
            vi=(pi-pi_old)*inv_dt;
            apos[idx]=(float4)(pi,Qi); avel[idx]=(float4)(vi,mi);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
