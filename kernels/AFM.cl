// afm.cl - AFM probe-particle relaxation and image generation
//
// Simulates Atomic Force Microscopy (AFM) by relaxing a probe particle at
// each point of a scan grid above a sample, then generating the AFM image
// from the relaxed probe positions and forces. Supports tilted cantilevers,
// convolution, and isosurface extraction.
//
// Execution flow (AFM image generation):
//   1. evalLJC_QZs / evalMorseC_QZs_toImg / evalDispersion_toImg — build force-field Z-slices from sample atoms (LJ/Morse/Coulomb/dispersion)
//   2. getFEinPoints / getFEinStrokes — sample precomputed force field at  scan grid points or along strokes
//   3. relaxPoints / relaxStrokes / relaxStrokesTilted — relax probe particle at each scan point using FIRE or damped dynamics until convergence
//   4. getZisoTilted / getZisoFETilted — extract isosurface from relaxed field
//   5. convolveZ — apply lateral convolution (tip-aperture effect)
//   6. izoZ — interpolate isosurface height at each (x,y) pixel
//
// Kernels:
//   - getFEinPoints: Sample force/energy field at discrete points from 3D texture.
//   - getFEinPointsShifted: Same with coordinate shift offset.
//   - getFEinStrokes: Sample field along stroke paths (for line scans).
//   - getFEinStrokesTilted: Same with tilted cantilever orientation.
//   - getZisoTilted: Extract z-isosurface from 3D force field with tilt.
//   - getZisoFETilted: Extract z-isosurface from force field with tilt.
//   - relaxPoints: Relax probe particle at each grid point using damped MD
//     with harmonic tip spring. 1 thread = 1 scan point.
//   - relaxStrokes: Relax probe along stroke paths.
//   - relaxStrokes2D: 2D variant of stroke relaxation.
//   - relaxStrokesTilted_debug: Debug version with per-step output.
//   - relaxStrokesTilted: Relax probe with tilted cantilever spring.
//   - relaxStrokesTilted_convZ: Tilted relaxation with Z-convolution.
//   - convolveZ: Lateral convolution of Z-height map (tip aperture effect).
//   - izoZ: Interpolate isosurface height at each pixel from 3D field.
//   - evalLJC_QZs: Evaluate LJ + Coulomb force at Z-slice points.
//   - evalLJC_QZs_toImg: Same, writing directly to image3d_t.
//   - evalMorseC_QZs_toImg: Evaluate Morse + Coulomb at Z-slices to image.
//   - evalDispersion_toImg: Evaluate London dispersion at Z-slices to image.
//   - gradient_central_diff: Numerical gradient via central differences on grid.
//   - fdbm_pad_roll_f32 / fdbm_flip3_f32 / fdbm_*_fft_* / fdbm_scale_pauli_pow_f32 /
//     fdbm_compose_E_to_img / fdbm_mul_poisson_tip_c64 — Round-2 fast S3 (switchable)
//
// Helper functions: tipForce (spring force), read_imagef_trilin/trilin_
// (manual trilinear interpolation), interpFE/interpFE_prec (field sampling),
// move_LeapFrog, update_FIRE (FIRE relaxation algorithm), getCoulomb/getLJ/
// getMorse/getMorseQ/getLJQ/getLondon (pairwise potentials for Z-slice builders),
// getR4repulsion (R^4 blob repulsion), getLorenz (Lorenzian for STM).
// Requires: common.cl + Forces.cl to be concatenated before this file.

// ---- AFM-specific helpers ----
// ---- Samplers (for image3d_t reads) ----
__constant sampler_t sampler_1 =  CLK_NORMALIZED_COORDS_TRUE  | CLK_ADDRESS_REPEAT | CLK_FILTER_LINEAR;
__constant sampler_t sampler_2 =  CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_MIRRORED_REPEAT | CLK_FILTER_NEAREST;
__constant sampler_t sampler_nearest =  CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_REPEAT | CLK_FILTER_NEAREST;

float3 tipForce( float3 dpos, float4 stiffness, float4 dpos0 ){
    float r = sqrt( dot( dpos,dpos) );
    r = fmax(r, 1e-10f);
    return  (dpos-dpos0.xyz) * stiffness.xyz        // harmonic 3D
         + dpos * ( stiffness.w * (r-dpos0.w)/r );  // radial
}

float4 read_imagef_trilin( __read_only image3d_t imgIn, float4 coord ){
    float4 d = (float4)(0.00666666666f,0.00666666666f,0.00666666666f,1.0f); 
    float4 icoord;
    float4 fc     =  fract( coord/d, &icoord );
    icoord*=d;
    float4 mc     = (float4)(1.0f,1.0f,1.0f,1.0f) - fc;
    // NOTE AMD-GPU seems to not accept CLK_NORMALIZED_COORDS_FALSE
    //return read_imagef( imgIn, sampler_2, icoord );
    //return read_imagef( imgIn, sampler_1, coord );
    return  
     (( read_imagef( imgIn, sampler_2, icoord+(float4)(0.0f,0.0f,0.0f,0.0f) ) * mc.x
      + read_imagef( imgIn, sampler_2, icoord+(float4)(d.x,0.0f,0.0f,0.0f) ) * fc.x )*mc.y
     +( read_imagef( imgIn, sampler_2, icoord+(float4)(0.0f,d.y,0.0f,0.0f) ) * mc.x
      + read_imagef( imgIn, sampler_2, icoord+(float4)(d.x,d.y,0.0f,0.0f) ) * fc.x )*fc.y )*mc.z
    +(( read_imagef( imgIn, sampler_2, icoord+(float4)(0.0f,0.0f,d.z,0.0f) ) * mc.x
      + read_imagef( imgIn, sampler_2, icoord+(float4)(d.x,0.0f,d.z,0.0f) ) * fc.x )*mc.y
     +( read_imagef( imgIn, sampler_2, icoord+(float4)(0.0f,d.y,d.z,0.0f) ) * mc.x
      + read_imagef( imgIn, sampler_2, icoord+(float4)(d.x,d.y,d.z,0.0f) ) * fc.x )*fc.y )*fc.z;
}; 


float4 read_imagef_trilin_( __read_only image3d_t imgIn, float4 coord ){
    float4 icoord;
    float4 fc     =  fract( coord, &icoord );
    float4 mc     = (float4)(1.0f,1.0f,1.0f,1.0f) - fc;
    // NOTE AMD-GPU seems to not accept CLK_NORMALIZED_COORDS_FALSE
    //return read_imagef( imgIn, sampler_2, icoord );
    //return read_imagef( imgIn, sampler_1, coord );
    return  
     (( read_imagef( imgIn, sampler_nearest, icoord+(float4)(0.0f,0.0f,0.0f,0.0f) ) * mc.x
      + read_imagef( imgIn, sampler_nearest, icoord+(float4)(1.0f,0.0f,0.0f,0.0f) ) * fc.x )*mc.y
     +( read_imagef( imgIn, sampler_nearest, icoord+(float4)(0.0f,1.0f,0.0f,0.0f) ) * mc.x
      + read_imagef( imgIn, sampler_nearest, icoord+(float4)(1.0f,1.0f,0.0f,0.0f) ) * fc.x )*fc.y )*mc.z
    +(( read_imagef( imgIn, sampler_nearest, icoord+(float4)(0.0f,0.0f,1.0f,0.0f) ) * mc.x
      + read_imagef( imgIn, sampler_nearest, icoord+(float4)(1.0f,0.0f,1.0f,0.0f) ) * fc.x )*mc.y
     +( read_imagef( imgIn, sampler_nearest, icoord+(float4)(0.0f,1.0f,1.0f,0.0f) ) * mc.x
      + read_imagef( imgIn, sampler_nearest, icoord+(float4)(1.0f,1.0f,1.0f,0.0f) ) * fc.x )*fc.y )*fc.z;
}; 


float4 interpFE( float3 pos, float4 dinvA, float4 dinvB, float4 dinvC, __read_only image3d_t imgIn ){
    // coord = (pos - origin) / L using 4-vector dot: dot([x,y,z,1], [1/L,0,0,-origin/L])
    float4 pos4 = (float4)(pos, 1.0f);
    const float4 coord = (float4)( dot(pos4,dinvA), dot(pos4,dinvB), dot(pos4,dinvC), 0.0f );
    return read_imagef( imgIn, sampler_1, coord );
}

float4 interpFE_prec( float3 pos, float4 dinvA, float4 dinvB, float4 dinvC, __read_only image3d_t imgIn ){
    float4 pos4 = (float4)(pos, 1.0f);
    const float4 coord = (float4)( dot(pos4,dinvA), dot(pos4,dinvB), dot(pos4,dinvC), 0.0f );
    return read_imagef_trilin( imgIn, coord ); 
}

// this should be macro, to pass values by reference
void move_LeapFrog( float3 f, float3 p, float3 v, float2 RP ){
    v  =  f * RP.x + v*RP.y;
    p +=  v * RP.x;
}


//#define N_RELAX_STEP_MAX  64
#define N_RELAX_STEP_MAX  128
#define F2CONV  1e-8f

#ifndef OPT_FIRE
#define OPT_FIRE 1
#endif
#if OPT_FIRE 
#define FTDEC 0.5f
#define FTINC 1.1f
#define FDAMP 0.99f


//#define F2CONV  1e-6f
#define F2SAFE    1e-8f

float3 update_FIRE( float3 f, float3 v, float* dt, float* damp,    float dtmin, float dtmax, float damp0 ){
    // Bitzek, E., Koskinen, P., Gähler, F., Moseler, M., & Gumbsch, P. (2006). Structural Relaxation Made Simple. Physical Review Letters, 97(17), 170201. 
    // https://doi.org/10.1103/PhysRevLett.97.170201
    // http://users.jyu.fi/~pekkosk/resources/pdf/FIRE.pdf
    float ff = dot(f,f);
    float vv = dot(v,v);
    float vf = dot(v,f);
    if( vf < 0 ){ // if velocity along direction of force
        v      *= 0;
        (*dt)   = fmax( dtmin, (*dt) * FTDEC );
        (*damp) = damp0;
    }else{       // if velocity against direction of force
        // v = cV * v  + cF * F
        v       *= (1 - (*damp));
        v       +=  f * ( (*damp) * sqrt( vv / (ff + F2SAFE ) ) );
        (*dt)    = fmin( dtmax, (*dt) * FTINC );
        (*damp) *= FDAMP;
    }
    return v;
    //v  += f * dt;
    //p  += v * dt;
}
#endif // OPT_FIRE

// ---- AFM force field sampling kernels ----
__kernel void getFEinPoints(
    __read_only image3d_t  imgIn,
    __global  float4*      points,
    __global  float4*      FEs,
    float4 dinvA,
    float4 dinvB,
    float4 dinvC
){
    //const float4 coord     = points[get_global_id(0)];
    //vals[get_global_id(0)] = read_imagef(imgIn, sampler_1, coord);
    FEs[get_global_id(0)]    = interpFE( points[get_global_id(0)].xyz, dinvA, dinvB, dinvC, imgIn );
}

__kernel void getFEinPointsShifted(
    __read_only image3d_t  imgIn,
    __global  float4*      points,
    __global  float4*      FEs,
    float4 dinvA,
    float4 dinvB,
    float4 dinvC,
    float4 dpos0
){
    FEs[get_global_id(0)] = interpFE( points[get_global_id(0)].xyz+dpos0.xyz, dinvA, dinvB, dinvC, imgIn );
}

__kernel void getFEinStrokes(
    __read_only image3d_t  imgIn,    // 1
    __global  float4*      points,   // 2
    __global  float4*      FEs,      // 3
    float4 dinvA,                    // 4
    float4 dinvB,                    // 5
    float4 dinvC,                    // 6
    float4 dTip,                     // 7
    float4 dpos0,                    // 8
    int nz                           // 
){
    //if(get_global_id(0)==0){ printf( "GPU getFEinStrokes() nz %i dTip(%g,%g,%g) dpos0(%g,%g,%g)\n", nz, dTip.x,dTip.y,dTip.z,   dpos0.x,dpos0.y,dpos0.z ); }
    //if(get_global_id(0)==0){ printf( "GPU getFEinStrokes() dinvA(%g,%g,%g) dinvB(%g,%g,%g) dinvC(%g,%g,%g)\n", dinvA.x,dinvA.y,dinvA.z,  dinvB.x,dinvB.y,dinvB.z,  dinvC.x,dinvC.y,dinvC.z ); }
    float3 pos    =  points[get_global_id(0)].xyz + dpos0.xyz; 
    for(int iz=0; iz<nz; iz++){
        float4 fe  =  read_imagef( imgIn, sampler_1, (float4){pos.x,pos.y,pos.z,0} );
        //float4 fe  = interpFE( pos, dinvA, dinvB, dinvC, imgIn );
        // if(get_global_id(0)==100)printf( "GPU %li %i (%f,%f,%f) -> fe(%g,%g,%g,%g) \n", get_global_id(0), iz, pos.x, pos.y, pos.z, fe.x,fe.y,fe.z,fe.w );
        //if(get_global_id(0)==0)printf( "GPU iz %i (%f,%f,%f) -> fe(%g,%g,%g,%g) \n", iz, pos.x, pos.y, pos.z, fe.x,fe.y,fe.z,fe.w );
        FEs[get_global_id(0)*nz + iz] = fe;
        pos    += dTip.xyz;
    }
}

__kernel void getFEinStrokesTilted(
    __read_only image3d_t  imgIn,
    __global  float4*      points,
    __global  float4*      FEs,
    float4 dinvA,
    float4 dinvB,
    float4 dinvC,
    float4 tipA,
    float4 tipB,
    float4 tipC,
    float4 dTip,
    float4 dpos0,
    int nz
){
    float3 pos    =  points[get_global_id(0)].xyz + dpos0.xyz; 
    for(int iz=0; iz<nz; iz++){
        //printf( " %li %i (%f,%f,%f) \n", get_global_id(0), iz, pos.x, pos.y, pos.z );
        float4 fe   = interpFE( pos, dinvA, dinvB, dinvC, imgIn );
        float4 fe_  = fe;
        fe_.xyz = rotMat( fe.xyz, tipA.xyz, tipB.xyz, tipC.xyz );
        FEs[get_global_id(0)*nz + iz]    = fe_;
        pos    += dTip.xyz;
    }
}

__kernel void getZisoTilted(
    __read_only image3d_t  imgIn,
    __global  float4*      points,
    __global  float*       zMap,
    float4 dinvA,
    float4 dinvB,
    float4 dinvC,
    float4 tipA,
    float4 tipB,
    float4 tipC,
    float4 dTip,
    float4 dpos0,
    int nz, float iso
){
    float3 pos     = points[get_global_id(0)].xyz + dpos0.xyz; 
    float4 ofe,fe;
    ofe     = interpFE( pos, dinvA, dinvB, dinvC, imgIn );
    ofe.xyz = rotMat( ofe.xyz, tipA.xyz, tipB.xyz, tipC.xyz );
    for(int iz=1; iz<nz; iz++){
        pos    += dTip.xyz;
        fe     = interpFE( pos, dinvA, dinvB, dinvC, imgIn );
        fe.xyz = rotMat( fe.xyz, tipA.xyz, tipB.xyz, tipC.xyz );
        //if( get_global_id(0) == 6050 ) printf( "iz %i fe %g iso %g \n", iz, fe.z, iso );
        if( fe.z/iso > 1.0f ){
            float t = (iso - ofe.z)/(fe.z - ofe.z);
            zMap[get_global_id(0)] = iz + t;
            return;
        }
        ofe      = fe;
    }
    zMap[get_global_id(0)] = -1;
}

__kernel void getZisoFETilted(
    __read_only image3d_t  imgIn,
    __read_only image3d_t  imgFE,
    __global  float4*      points,
    __global  float*       zMap,
    __global  float4*      feMap,
    float4 dinvA,
    float4 dinvB,
    float4 dinvC,
    float4 tipA,
    float4 tipB,
    float4 tipC,
    float4 dTip,
    float4 dpos0,
    int nz, float iso
){
    float3 pos     = points[get_global_id(0)].xyz + dpos0.xyz; 
    float4 ofe,fe;
    ofe     = interpFE( pos, dinvA, dinvB, dinvC, imgIn );
    ofe.xyz = rotMat( ofe.xyz, tipA.xyz, tipB.xyz, tipC.xyz );
    for(int iz=1; iz<nz; iz++){
        pos    += dTip.xyz;
        fe     = interpFE( pos, dinvA, dinvB, dinvC, imgIn );
        fe.xyz = rotMat( fe.xyz, tipA.xyz, tipB.xyz, tipC.xyz );
        //if( get_global_id(0) == 6050 ) printf( "iz %i fe %g iso %g \n", iz, fe.z, iso );
        if( fe.z/iso > 1.0f ){
            float t = (iso - ofe.z)/(fe.z - ofe.z);
            zMap [get_global_id(0)] = iz + t;
            fe     = interpFE( pos+dTip.xyz*t, dinvA, dinvB, dinvC, imgFE );
            fe.xyz = rotMat( fe.xyz, tipA.xyz, tipB.xyz, tipC.xyz );
            feMap[get_global_id(0)] = fe;
            return;
        }
        ofe      = fe;
    }
    zMap [get_global_id(0)] = -1;
    feMap[get_global_id(0)] =  float4Zero;
}

// ---- AFM probe-particle relaxation kernels ----
__kernel void relaxPoints(
    __read_only image3d_t  imgIn,
    __global  float4*      points,
    __global  float4*      FEs,
    float4 stiffness,
    float4 dpos0,
    float4 relax_params  // (dt,damp,tmin,tmax)
){

    float dt      = relax_params.x;
    float damp    = relax_params.y;

    float dtmax = dt;
    float dtmin = dtmax*0.1;
    float damp0 = damp;

    float3 tipPos = points[get_global_id(0)].xyz;
    float3 pos    = tipPos.xyz + dpos0.xyz; 
    float4 fe;
    float3 v    = 0.0f;
    for(int i=0; i<1000; i++){
        fe        = read_imagef( imgIn, sampler_1, (float4)(pos,0.0f) ); /// this would work only for unitary cell
        float3 f  = fe.xyz;
        f        += tipForce( pos-tipPos, stiffness, dpos0 );    

        #if OPT_FIRE
        v = update_FIRE( f, v, &dt, &damp, dtmin, dtmax, damp0 );
        #else
        v        *=    (1 - damp);
        #endif
        v        += f * dt;
        pos.xyz  += v * dt;

    }
    FEs[get_global_id(0)] = fe;
}

__kernel void relaxStrokes(
    __read_only image3d_t  imgIn,
    __global  float4*      points,
    __global  float4*      FEs,
    __global  float4*      disps,
    float4 dinvA,
    float4 dinvB,
    float4 dinvC,
    float4 dTip,
    float4 stiffness,
    float4 dpos0,
    float4 relax_params,
    int nz
){
    float3 tipPos = points[get_global_id(0)].xyz;
    float3 pos    = tipPos.xyz + dpos0.xyz; 
    
    float dt      = relax_params.x;
    float damp    = relax_params.y;
    //printf( " %li (%f,%f,%f)  \n",  get_global_id(0), tipPos.x, tipPos.y, tipPos.z);
    
    float dtmax = dt;
    float dtmin = dtmax*0.1f;
    float damp0 = damp;

    for(int iz=0; iz<nz; iz++){
        float4 fe;
        float3 v   = 0.0f;
        for(int i=0; i<N_RELAX_STEP_MAX; i++){
            fe        = interpFE( pos, dinvA, dinvB, dinvC, imgIn );
            float3 f  = fe.xyz;
            f        += tipForce( pos-tipPos, stiffness, dpos0 );
            
            #if OPT_FIRE
            v = update_FIRE( f, v, &dt, &damp, dtmin, dtmax, damp0 );
            #else
            v        *=    (1 - damp);
            #endif
            v        += f * dt;
            pos.xyz  += v * dt;

            if(dot(f,f)<F2CONV) break;
        }
        int idx = get_global_id(0)*nz + iz;
        FEs[idx] = fe;
        disps[idx] = (float4)(pos - (tipPos + dpos0.xyz), 0.0f);
        tipPos += dTip.xyz;
        pos    += dTip.xyz;
    }
}

// relaxStrokes2D: 2D lateral-only damped MD relaxation.
// Matches CPU pp_relax_2d exactly: z is fixed per height slice, only x,y relax.
// Damped velocity update: v *= (1-damp); v += F*dt; pos += v*dt
// Lateral spring: F_spring = -K_lat * (pos.xy - anchor.xy)
// Args:
//   imgIn    - 3D force field image (Fx,Fy,Fz,E)
//   points   - (n_scan,4) tip anchor positions (world coords); w=start_z for first height
//   FEs      - (n_scan*nz,4) output: interpolated (Fx,Fy,Fz,E) at relaxed position
//   dinvA/B/C - inverse cell vectors for normalized image coords
//   K_lat    - lateral spring stiffness [eV/Ang^2]
//   dh       - z step downward between height slices [Ang]  (dh>0 means descending)
//   dt       - time step
//   damp     - velocity damping coefficient  (v *= 1-damp each step)
//   nz       - number of height slices
__kernel void relaxStrokes2D(
    __read_only image3d_t  imgIn,
    __global  float4*      points,
    __global  float4*      FEs,
    __global  float4*      disps,
    float4 dinvA,
    float4 dinvB,
    float4 dinvC,
    float K_lat,
    float dh,
    float dt,
    float damp,
    int   nz
){
    int gid = get_global_id(0);
    float4 tip0 = points[gid];              // anchor: (ax, ay, az_start, _)
    float  ax   = tip0.x;
    float  ay   = tip0.y;
    float  az   = tip0.z;                   // z of first (highest) height slice

    for(int iz=0; iz<nz; iz++){
        float pz = az - iz*dh;              // z fixed for this slice (descend by dh per step)
        float px = ax, py = ay;             // reset per slice (matches CPU pp_relax_2d lines 1448-1450)
        float vx = 0.0f, vy = 0.0f;

        for(int i=0; i<N_RELAX_STEP_MAX; i++){
            float4 fe = interpFE( (float3)(px, py, pz), dinvA, dinvB, dinvC, imgIn );
            float  fx = fe.x - K_lat * (px - ax);
            float  fy = fe.y - K_lat * (py - ay);
            vx = vx*(1.0f - damp) + fx*dt;
            vy = vy*(1.0f - damp) + fy*dt;
            px += vx*dt;
            py += vy*dt;
            if( (fx*fx + fy*fy) < F2CONV ) break;
        }
        float4 fe_out = interpFE( (float3)(px, py, pz), dinvA, dinvB, dinvC, imgIn );
        int idx = gid*nz + iz;
        FEs[idx] = fe_out;
        disps[idx] = (float4)(px - ax, py - ay, 0.0f, 0.0f);
    }
}

__kernel void relaxStrokesTilted_debug(
    __read_only image3d_t  imgIn,
    __global  float4*      points,
    __global  float4*      FEs,
    float4 dinvA,
    float4 dinvB,
    float4 dinvC,
    float4 tipA,
    float4 tipB,
    float4 tipC,
    float4 stiffness,
    float4 dpos0,
    float4 relax_params,
    float4 surfFF,
    int nz
){
    const float3 dTip   = tipC.xyz * tipC.w;
    float4 dpos0_=dpos0; dpos0_.xyz= rotMatT( dpos0_.xyz , tipA.xyz, tipB.xyz, tipC.xyz );
    float3 tipPos = points[get_global_id(0)].xyz;
    float3 pos    = tipPos.xyz + dpos0_.xyz; 
    for(int iz=0; iz<nz; iz++){
        FEs[get_global_id(0)*nz + iz] = 1.0f;
    }
}


__kernel void relaxStrokesTilted(
    __read_only image3d_t  imgIn,   // 1
    __global  float4*      points,  // 2
    __global  float4*      FEs,     // 3
    float4 dinvA,                   // 4
    float4 dinvB,                   // 5
    float4 dinvC,                   // 6
    float4 tipA,                    // 7
    float4 tipB,                    // 8
    float4 tipC,                    // 9 
    float4 stiffness,               // 10
    float4 dpos0,                   // 11
    float4 relax_params,            // 12
    float4 surfFF,                  // 13
    int nz                          // 14
){

    const float3 dTip   = tipC.xyz * tipC.w;
    float4 dpos0_=dpos0; dpos0_.xyz= rotMatT( dpos0_.xyz , tipA.xyz, tipB.xyz, tipC.xyz );

    float3 tipPos = points[get_global_id(0)].xyz;
    float3 pos    = tipPos.xyz + dpos0_.xyz; 

    float dt      = relax_params.x;
    float damp    = relax_params.y;

    float dtmax = dt;
    float dtmin = dtmax*0.1f;
    float damp0 = damp;

    // if( (get_global_id(0)==0) ){  
    //     printf( " dt %g damp %g \n", dt, damp );
    //     printf( " stiffness(%g,%g,%g|%g) dpos0(%g,%g,%g|%g) \n", stiffness.x,stiffness.y,stiffness.z,stiffness.w,  dpos0.x,dpos0.y,dpos0.z,dpos0.w  );
    //     printf( " relax_params(%g,%g,%g|%g) surfFF(%g,%g,%g|%g) \n", relax_params.x,relax_params.y,relax_params.z,relax_params.w,  surfFF.x,surfFF.y,surfFF.z,surfFF.w  );
    //     printf( " dinvA(%g,%g,%g|%g) tipA(%g,%g,%g|%g) \n", dinvA.x,dinvA.y,dinvA.z,dinvA.w,  tipA.x,tipA.y,tipA.z,tipA.w  );
    //     printf( " dinvB(%g,%g,%g|%g) tipB(%g,%g,%g|%g) \n", dinvB.x,dinvB.y,dinvB.z,dinvB.w,  tipB.x,tipB.y,tipB.z,tipB.w  );
    //     printf( " dinvc(%g,%g,%g|%g) tipC(%g,%g,%g|%g) \n", dinvC.x,dinvC.y,dinvC.z,dinvC.w,  tipC.x,tipC.y,tipC.z,tipC.w  );
    //     int i1=get_global_size(0)-1; printf( "pos0(%3.3f,%3.3f,%3.3f) pos1(%3.3f,%3.3f,%3.3f)\n", points[0].x,points[0].y,points[0].z, points[i1].x,points[i1].y,points[i1].z );
    //     //for(int i=0; i<get_global_size(0); i++ ){ printf( "pos[%i] (%3.3f,%3.3f,%3.3f)\n", i, points[i].x,points[i].y,points[i].z ); }
    // }
    //if( (get_global_id(0)==0) ){     float4 fe = interpFE( pos, dinvA.xyz, dinvB.xyz, dinvC.xyz, imgIn );  printf( " pos (%g,%g,%g) feImg(%g,%g,%g,%g) \n", pos.x, pos.y, pos.z, fe.x,fe.y,fe.z,fe.w );}

    for(int iz=0; iz<nz; iz++){
        float4 fe;
        float3 v   = (float3){0.f,0.f,0.f};
        
        for(int i=0; i<N_RELAX_STEP_MAX; i++){
            fe            = interpFE( pos, dinvA, dinvB, dinvC, imgIn );
            float3 f      = fe.xyz;
            float3 dpos   = pos-tipPos;
            float3 dpos_  = rotMat  ( dpos, tipA.xyz, tipB.xyz, tipC.xyz );    // to tip-coordinates
            float3 ftip   = tipForce( dpos_, stiffness, dpos0 );

            f            += rotMatT ( ftip, tipA.xyz, tipB.xyz, tipC.xyz );      // from tip-coordinates
            f            +=  tipC.xyz * surfFF.x;                                // TODO: more sophisticated model of surface potential? Like Hamaker ?

            //f      +=  tipForce( dpos, stiffness, dpos0_ );  // Not rotated
            
            #if OPT_FIRE
            v = update_FIRE( f, v, &dt, &damp, dtmin, dtmax, damp0 );
            #else
            v        *=    (1 - damp);
            #endif
            v        += f * dt;
            pos.xyz  += v * dt;

            if(dot(f,f)<F2CONV) break;
        }
        fe            = interpFE( pos, dinvA, dinvB, dinvC, imgIn );
        if(1){ // output tip-rotated force
            float4 fe_  = fe;
            fe_.xyz = rotMat( fe.xyz, tipA.xyz, tipB.xyz, tipC.xyz );
            fe_.w   = fe.w;
            FEs[get_global_id(0)*nz + iz] = fe_;
        }else{ // output molecule-rotated force 
            FEs[get_global_id(0)*nz + iz] = fe;
            //FEs[get_global_id(0)*nz + iz].xyz = pos;
        }
        tipPos += dTip.xyz;
        pos    += dTip.xyz;
        //if( (get_global_id(0)==0) ){ printf( "iz[%i] pos(%g,%g,%g) fe(%g,%g,%g|%g) \n", iz, pos.x, pos.y, pos.z, fe.x,fe.y,fe.z,fe.w ); }
        //if( (get_global_id(0)==0) ){ printf( "iz[%i] pos(%g,%g,%g) tipPos(%g,%g,%g) \n", iz, pos.x,pos.y,pos.z, tipPos.x,tipPos.y,tipPos.z ); }

    }
}



__kernel void relaxStrokesTilted_convZ(
    __read_only image3d_t  imgIn,
    __global  float4*      points,
    __constant  float*     weighs,
    __global  float4*      FEs,
    float4 dinvA,
    float4 dinvB,
    float4 dinvC,
    float4 tipA,
    float4 tipB,
    float4 tipC,
    float4 stiffness,
    float4 dpos0,
    float4 relax_params,
    float4 surfFF,
    const int nz, const int nzout
){

    __local float  WEIGHTS[64];

    const float3 dTip   = tipC.xyz * tipC.w;
    float4 dpos0_=dpos0; dpos0_.xyz= rotMatT( dpos0_.xyz , tipA.xyz, tipB.xyz, tipC.xyz );

    float3 tipPos = points[get_global_id(0)].xyz;
    float3 pos    = tipPos.xyz + dpos0_.xyz; 

    float dt      = relax_params.x;
    float damp    = relax_params.y;

    float dtmax = dt;
    float dtmin = dtmax*0.1f;
    float damp0 = damp;

    //if( (get_global_id(0)==0) ){     float4 fe = interpFE( pos, dinvA.xyz, dinvB.xyz, dinvC.xyz, imgIn );  printf( " pos (%g,%g,%g) feImg(%g,%g,%g,%g) \n", pos.x, pos.y, pos.z, fe.x,fe.y,fe.z,fe.w );}
    //if( (get_global_id(0)==0) ){ printf( "dt %g damp %g \n", dt, damp ); }; return;

    const int ioff = get_global_id(0)*nzout;
    const int nzw   = nz-nzout;
    const int iL=get_local_id(0);
    const int nL=get_local_size(0);
    for (int i=iL; i<nzw; i+=nL ){
        WEIGHTS[i] = weighs[i];
    }
    for (int iz=0; iz<nzout; iz++ ){
        FEs[ioff+iz] = 0.0f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int itr_tot = 0;

    for(int iz=0; iz<nz; iz++){
        float4 fe;
        float3 v   = 0.0f;
        for(int i=0; i<N_RELAX_STEP_MAX; i++){
        //for(int i=0; i<1; i++){ // DEBUG
            fe            = interpFE( pos, dinvA, dinvB, dinvC, imgIn );
            //fe            = interpFE_prec( pos, dinvA, dinvB, dinvC, imgIn );
            float3 f      = fe.xyz;
            float3 dpos   = pos-tipPos;
            float3 dpos_  = rotMat  ( dpos, tipA.xyz, tipB.xyz, tipC.xyz );    // to tip-coordinates
            float3 ftip   = tipForce( dpos_, stiffness, dpos0 );

            f            += rotMatT ( ftip, tipA.xyz, tipB.xyz, tipC.xyz );      // from tip-coordinates
            f            +=  tipC.xyz * surfFF.x;                                // TODO: more sophisticated model of surface potential? Like Hamaker ?

            //f      +=  tipForce( dpos, stiffness, dpos0_ );  // Not rotated

            #if OPT_FIRE
            v = update_FIRE( f, v, &dt, &damp, dtmin, dtmax, damp0 );
            //if(get_global_id(0)==(64*128+64)){ printf( "itr,iz,i %i %i %i  |F| %g |v| %g <f,v> %g , (%g,%g,%g) (%g,%g,%g) damp %g dt %g \n", itr_tot, iz,i,  sqrt(dot(f,f)), sqrt(dot(v,v)),  dot(f,v),  fe.x,fe.y,fe.z, pos.x, pos.y, pos.z, damp, dt ); }
            #else
            v        *=    (1 - damp);
            //if(get_global_id(0)==(64*128+64)){ printf( "itr,iz,i %i %i %i  |F| %g |v| %g <f,v> %g , (%g,%g,%g) (%g,%g,%g) damp %g dt %g \n", itr_tot, iz,i,  sqrt(dot(f,f)), sqrt(dot(v,v)),  dot(f,v),  fe.x,fe.y,fe.z, pos.x, pos.y, pos.z, damp, dt ); }
            #endif
            v        += f * dt;
            pos.xyz  += v * dt;

            itr_tot++;
            if(dot(f,f)<F2CONV) break;
        }
        
        if(1){ // output tip-rotated force
            fe.xyz = rotMat( fe.xyz, tipA.xyz, tipB.xyz, tipC.xyz );
        }

        
        // do the convolution
        for(int izout=0;izout<nzout;izout++){
            int jzw = iz - izout;
            if((jzw<nzw)&&(jzw>0)){
                FEs[ ioff + izout] += fe * WEIGHTS[jzw];
            }
        }
        //if( iz<nzout ) FEs[ioff+iz] = fe;
        tipPos += dTip.xyz;
        pos    += dTip.xyz;
    }

}

__kernel void convolveZ(
    __global  float4* Fin,
    __global  float4* Fout,
    //__global  float*  weighs,
    __constant  float*  weighs,
    const int nzin, const int nzout
){
    const int ioffi = get_global_id(0)*nzin;
    const int ioffo = get_global_id(0)*nzout;
    const int nzw   = nzin-nzout;
    //if( get_global_id(0)==0 ) printf( "local size %i \n", get_local_size(0) );
    //if( get_global_id(0)==0 ) printf( "izo %i izi %i Fz %g W %g \n", nzin, nzout, nzw );

    __local float WEIGHTS[64];

    const int iL=get_local_id(0);
    const int nL=get_local_size(0);
    for (int i=iL; i<nzw; i+=nL ){
        if( i<nzw ) WEIGHTS[i] = weighs[i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for(int izo=0; izo<nzout; izo++){
        float4 fe = 0.0f;
        for(int jz=0; jz<nzw; jz++){
            //fe += Fin[ ioffi + izo + jz ] * weighs[ jz ];
            fe += Fin[ ioffi + izo + jz ] * WEIGHTS[ jz ];
            //if( get_global_id(0)==0 ) printf( "izo %i izi %i Fz %g W %g \n", izo, jz, Fin[ ioffi + izo + jz ].z, weighs[ jz ] );
            //fe +=  tanh( Fin[ ioffi + izi ] ) * weighs[ izi - izo ];
        }
        //if( ioffi == 0 ){ printf( "izo %i w[i] %e \n", izo, weighs[ izo ] ); }
        //fe = (float)ioffo; // DEBUG
        Fout[ ioffo + izo ] = fe;
        //Fout[ ioffo + izo ] = weighs[ izo ];
        //Fout[ ioffo + izo ] = (float4) izo;
        //Fout[ ioffo + izo ] = Fin[ ioffi + izo ];
    }
}

__kernel void izoZ(
    __global  float4* Fin,
    __global  float*  zMap,
    int nz,   float iso
){
    int ioffi = get_global_id(0)*nz;
    float4 ofe = Fin[ ioffi ];
    for(int iz=1; iz<nz; iz++){
        float4 fe = Fin[ ioffi + iz ];
        // zMap[get_global_id(0)] = i;
        if( fe.z > iso ){
            float t = (iso - ofe.z)/(fe.z - ofe.z);
            zMap[get_global_id(0)] = iz + t;
            return;
        }
        ofe = fe;
    }
    zMap[get_global_id(0)] = -1;
}

// =========================================  
//           ForceField form FF.cl
// =========================================

// ---- AFM-specific pair potential functions ----
float4 getCoulombAFM( float4 atom, float3 pos ){
     float3  dp  =  pos - atom.xyz;
     float   ir2 = 1.0f/( dot(dp,dp) +  R2SAFE );
     float   ir  = sqrt(ir2);
     float   E   = atom.w*sqrt(ir2);
     return (float4)(dp*(E*ir2), E );
}

float4 getLJ( float3 apos, float2 cLJ, float3 pos ){
     float3  dp  =  pos - apos;
     float   ir2 = 1.0f/( dot(dp,dp) + R2SAFE );
     float   ir6 = ir2*ir2*ir2;
     float   E   =  (    cLJ.y*ir6 -   cLJ.x )*ir6;
     float3  F   = (( 12.0f*cLJ.y*ir6 - 6.0f*cLJ.x )*ir6*ir2)*dp;
     return (float4)(F, E);
}

float4 getMorse( float3 dp, float3 REA ){
    // REA = (R0, E0, K)  K<0 i.e. K=-alpha  (standard Morse alpha>0)
    // E   =  E0 * (expar^2 - 2*expar)   expar=exp(K*(r-R0))
    // F   = -dE/dr * dp/r = -2*K*E0*expar*(expar-1)*dp/r
    //float3  dp  =  pos - apos;
    float   r     = sqrt( dot(dp,dp) + R2SAFE );
    float   expar = exp( REA.z*(r-REA.x) );
    float   E     = REA.y*expar*( expar - 2 );
    float   fr    = -REA.y*expar*( expar - 1 )*2*REA.z;  // fixed sign: -dE/dr
    return (float4)(dp*(fr/r), E);
}

float4 getMorseQ_bak( float3 dp, float4 REKQ ){
    float  r2  = dot(dp,dp) +  R2SAFE;
    float ir2  = 1/r2; 
    float   r  = sqrt( r2 );
    // ---- Electrostatic
    float   E  = REKQ.w*sqrt(ir2);
    float4 fe  = (float4)(dp*(E*ir2), E );
    // ---- Morse ( Pauli + Dispersion )
    float   expar = exp( REKQ.z*(r-REKQ.x) );
    float   e     = REKQ.y*expar;
    float4  fM    = (float4)(dp*(e*REKQ.z), e );
    fe += fM*(expar-2.0f);
    return fe; 
}

float4 getMorseQ( float3 dp, float4 REQK, float R2damp ){
    float  r2   = dot(dp,dp);
    float   r   = sqrt( r2 );
    // ---- Electrostatic
    float ir2   = 1/  ( r2 +  R2damp); 
    float  ir   = sqrt( ir2 );
    float   Ec  = COULOMB_CONST*REQK.z*ir;
    // ---- Morse ( Pauli + Dispersion )
    float   e   =  exp( REQK.w*(r-REQK.x) );
    float   Ae  =  REQK.y*e;
    float fMors =  Ae * (e - 1.f)*2.f*REQK.w/r; // Morse
    float EMors =  Ae * (e - 2.f);
    return ((float4)( dp*( fMors - Ec*ir2 ), EMors + Ec )); 
}

float4 getLJQ( float3 dp, float3 REQ, float R2damp ){
    // ---- Electrostatic
    float   r2    = dot(dp,dp);
    float   ir2_  = 1.f/(  r2 +  R2damp);
    float   Ec    =  COULOMB_CONST*REQ.z*sqrt( ir2_ );
    // --- LJ 
    float  ir2 = 1.f/r2;
    float  u2  = REQ.x*REQ.x*ir2;
    float  u6  = u2*u2*u2;
    float vdW  = u6*REQ.y;
    float E    =       (u6-2.f)*vdW     + Ec  ;
    float fr   = -12.f*(u6-1.f)*vdW*ir2 - Ec*ir2_;
    return  (float4){ dp*fr, E };
}

float4 getLondon( float3 dp, float2 RE, float R2damp ){
    // --- LJ 
    float   r2 = dot(dp,dp) + R2damp;
    float  ir2 = 1.f/r2;
    float  u2  = RE.x*RE.x*ir2;
    float  u6  = u2*u2*u2;
    float vdW  = u6*RE.y;
    float E    =      -2.f*vdW    ;
    float fr   = 12.f*-1.f*vdW*ir2;
    return  (float4){ dp*fr, E };
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
        float fr  = A*mr2;
        float r    = sqrt(r2);
        float fmax = 4*R*fr;
        return (float4){ d* (-fmax/r), fmax*(R-r) + fr*mr2 };
    }
}


float getLorenz( float4 atom, float4 coefs, float3 pos ){
     float3  dp  =  pos - atom.xyz;
     return coefs.x/( dot(dp,dp) +  coefs.y*coefs.y );
     //return 1.0/( dot(dp,dp) +  0.000 );
}



// ---- AFM force field builders (Z-slice evaluation) ----
__kernel void evalLJC_QZs(
    const int nAtoms,        // 1
    __global float4* atoms,  // 2
    __global float2*  cLJs,  // 3
    __global float4*    FE,  // 4
    int4 nGrid,              // 5
    float4 grid_p0,          // 6 
    float4 grid_dA,          // 7
    float4 grid_dB,          // 8
    float4 grid_dC,          // 9
    float4 Qs,               // 10
    float4 QZs               // 11
){
    __local float4 LATOMS[32];
    __local float2 LCLJS [32];
    const int iG = get_global_id (0);
    const int iL = get_local_id  (0);
    const int nL = get_local_size(0);
   
    const int nab = nGrid.x*nGrid.y;
    const int ia  = iG%nGrid.x; 
    const int ib  = (iG%nab)/nGrid.x;
    const int ic  = iG/nab; 
    const int nMax = nab*nGrid.z;

    //if (  get_global_id(0)==0 ) { printf("GPU evalLJC_QZs \n" ); }
    // if(iG==0) printf( " Qs (%g,%g,%g,%g) QZs (%g,%g,%g,%g) \n", Qs.x,Qs.y,Qs.z,Qs.w,   QZs.x,QZs.y,QZs.z,QZs.w   );
    //if(iG==0) printf( " dA(%g,%g,%g) dB(%g,%g,%g) dC(%g,%g,%g) p0(%g,%g,%g)\n", grid_dA.x,grid_dA.y,grid_dA.z,   grid_dB.x,grid_dB.y,grid_dB.z,  grid_dC.x,grid_dC.y,grid_dC.z, grid_p0.x,grid_p0.y,grid_p0.z );
    if(iG>nMax) return;

    float3 pos    = grid_p0.xyz + grid_dA.xyz*ia + grid_dB.xyz*ib  + grid_dC.xyz*ic;

    float4 fe  =  float4Zero;
    
    Qs *= COULOMB_CONST;

    for (int i0=0; i0<nAtoms; i0+= nL ){
        int i = i0 + iL;
        //if(i>=nAtoms) break;  // wrong !!!!
        LATOMS[iL] = atoms[i];
        LCLJS [iL] = cLJs [i];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int j=0; j<nL; j++){
            if( (j+i0)<nAtoms ){ 
                //fe += getLJC( LATOMS[j], LCLJS[j], pos );
                float4 xyzq = LATOMS[j];
                //if(iG==0) printf( "atom[%i](%g,%g,%g|%g) cLJ(%g,%g)\n", i, xyzq.x,xyzq.y,xyzq.z,  xyzq.w,   LCLJS[j].x, LCLJS[j].y );
                fe += getLJ     ( xyzq.xyz, LCLJS[j], pos );
                // ToDo : Electrostatics seems to be too strong in original forcefeidl
                fe += getCoulombAFM( xyzq, pos+(float3)(0,0,QZs.x) ) * Qs.x;
                fe += getCoulombAFM( xyzq, pos+(float3)(0,0,QZs.y) ) * Qs.y;
                fe += getCoulombAFM( xyzq, pos+(float3)(0,0,QZs.z) ) * Qs.z;
                fe += getCoulombAFM( xyzq, pos+(float3)(0,0,QZs.w) ) * Qs.w;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float renorm = 100.0/fabs(fe.w);
    if( renorm<1.f ){ fe*=renorm; }
    //if ( (ia==nGrid.x/2)&&(ib==nGrid.y/2) ) { printf(" iz %i pos(%g,%g,%g) fe(%g,%g,%g|%g) \n", ic,  pos.x,pos.y,pos.z,  fe.x, fe.y, fe.z, fe.w ); }
    FE[iG] = fe;
}



__kernel void evalLJC_QZs_toImg(
    const int nAtoms,        // 1
    __global float4* atoms,  // 2
    __global float2*  cLJs,  // 3
    __write_only image3d_t  imgOut, // 4
    const int4 nGrid,              // 5
    const float4 grid_p0,          // 6 
    const float4 grid_dA,          // 7
    const float4 grid_dB,          // 8
    const float4 grid_dC,          // 9
    float4 Qs,               // 10
    float4 QZs               // 11
){
    
    __local float4 LATOMS[32];
    __local float2 LCLJS [32];
    const int iG = get_global_id (0);
    const int iL = get_local_id  (0);
    const int nL = get_local_size(0);
   
    const int nab = nGrid.x*nGrid.y;
    const int ia  = iG%nGrid.x; 
    const int ib  = (iG%nab)/nGrid.x;
    const int ic  = iG/nab; 
    const int nMax = nab*nGrid.z;

    //if (  get_global_id(0)==0 ) { printf("GPU evalLJC_QZs \n" ); }
    // if(iG==0){
    //     printf( " nGrid(%i,%i,%i|%i)\n", nGrid.x,nGrid.y,nGrid.z,nGrid.w );
    //     printf( " grid_p0(%g,%g,%g|%g)\n", grid_p0.x,grid_p0.y,grid_p0.z,grid_p0.w );
    //     printf( " grid_dA(%g,%g,%g|%g)\n", grid_dA.x,grid_dA.y,grid_dA.z,grid_dA.w );
    //     printf( " grid_dB(%g,%g,%g|%g)\n", grid_dA.x,grid_dA.y,grid_dA.z,grid_dA.w );
    //     printf( " grid_dC(%g,%g,%g|%g)\n", grid_dB.x,grid_dB.y,grid_dB.z,grid_dB.w );
    //     printf( " dinvc(%g,%g,%g|%g)\n", grid_dC.x,grid_dC.y,grid_dC.z,grid_dC.w );
    //     printf( " Qs (%g,%g,%g|%g)\n", Qs.x,Qs.y,Qs.z,Qs.w );
    //     printf( " QZs(%g,%g,%g|%g)\n", QZs.x,QZs.y,QZs.z,QZs.w );
    //     for(int i=0; i<nAtoms; i++){
    //         printf( "atom(%g,%g,%g|%g) cLJ(%g,%g)\n", atoms[i].x,atoms[i].y,atoms[i].z,atoms[i].w,  cLJs[i].x,cLJs[i].y );
    //     }
    // }
    if(iG>=nMax) return;

    float4 fe  =  float4Zero;
    float3 pos    = grid_p0.xyz + grid_dA.xyz*ia + grid_dB.xyz*ib  + grid_dC.xyz*ic;
    
    Qs *= COULOMB_CONST;

    for (int i0=0; i0<nAtoms; i0+= nL ){
        int i = i0 + iL;
        //if(i>=nAtoms) break;  // wrong !!!!
        LATOMS[iL] = atoms[i];
        LCLJS [iL] = cLJs [i];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int j=0; j<nL; j++){
            if( (j+i0)<nAtoms ){ 
                //fe += getLJC( LATOMS[j], LCLJS[j], pos );
                float4 xyzq = LATOMS[j];
                //if(iG==0) printf( "atom[%i](%g,%g,%g|%g) cLJ(%g,%g)\n", i, xyzq.x,xyzq.y,xyzq.z,  xyzq.w,   LCLJS[j].x, LCLJS[j].y );
                fe += getLJ     ( xyzq.xyz, LCLJS[j], pos );
                // ToDo : Electrostatics seems to be too strong in original forcefeidl
                fe += getCoulombAFM( xyzq, pos+(float3)(0,0,QZs.x) ) * Qs.x;
                fe += getCoulombAFM( xyzq, pos+(float3)(0,0,QZs.y) ) * Qs.y;
                fe += getCoulombAFM( xyzq, pos+(float3)(0,0,QZs.z) ) * Qs.z;
                fe += getCoulombAFM( xyzq, pos+(float3)(0,0,QZs.w) ) * Qs.w;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float renorm = 100.0/fabs(fe.w);
    if( renorm<1.f ){ fe*=renorm; }
    //if ( (ia==nGrid.x/2)&&(ib==nGrid.y/2) ) { printf(" iz %i pos(%g,%g,%g) fe(%g,%g,%g|%g) \n", ic,  pos.x,pos.y,pos.z,  fe.x, fe.y, fe.z, fe.w ); }
    //imgOut[iG] = fe;
    //fe  = (float4){sin(ia*0.1), sin(ia*0.1), sin(ib*0.1), cos(ia*0.1)*cos(ib*0.1)*cos(ic*0.1) };
    write_imagef( imgOut, (int4){ia,ib,ic,0}, fe );
    //write_imagef( imgOut, (int4){0,0,0,0},  float4Zero );
}

// ========================== evalMorseC_QZs_toImg
// Same as evalLJC_QZs_toImg but uses getMorse(dp, cMs[j].xyz) instead of getLJ.
// cMs: float4 per atom: (.x=R0_ij, .y=E0_ij, .z=alpha, .w=unused)
__kernel void evalMorseC_QZs_toImg(
    const int nAtoms,
    __global float4* atoms,         // 2  (x,y,z,q)
    __global float4* cMs,           // 3  (R0, E0, alpha, 0)
    __write_only image3d_t  imgOut, // 4
    const int4 nGrid,               // 5
    const float4 grid_p0,           // 6
    const float4 grid_dA,           // 7
    const float4 grid_dB,           // 8
    const float4 grid_dC,           // 9
    float4 Qs,                      // 10
    float4 QZs                      // 11
){
    __local float4 LATOMS[32];
    __local float4 LCLJS [32];
    const int iG = get_global_id(0);
    const int iL = get_local_id (0);
    const int nL = get_local_size(0);
    const int nab = nGrid.x*nGrid.y;
    const int ia  = iG%nGrid.x;
    const int ib  = (iG%nab)/nGrid.x;
    const int ic  = iG/nab;
    const int nMax = nab*nGrid.z;
    if(iG>=nMax) return;
    float4 fe  = float4Zero;
    float3 pos = grid_p0.xyz + grid_dA.xyz*ia + grid_dB.xyz*ib + grid_dC.xyz*ic;
    Qs *= COULOMB_CONST;
    for(int i0=0; i0<nAtoms; i0+=nL){
        int i = i0+iL;
        LATOMS[iL] = atoms[i];
        LCLJS [iL] = cMs  [i];
        barrier(CLK_LOCAL_MEM_FENCE);
        for(int j=0; j<nL; j++){
            if((j+i0)<nAtoms){
                float4 xyzq = LATOMS[j];
                float3 dp   = pos - xyzq.xyz;
                fe += getMorse( dp, LCLJS[j].xyz );
                fe += getCoulombAFM( xyzq, pos+(float3)(0,0,QZs.x) ) * Qs.x;
                fe += getCoulombAFM( xyzq, pos+(float3)(0,0,QZs.y) ) * Qs.y;
                fe += getCoulombAFM( xyzq, pos+(float3)(0,0,QZs.z) ) * Qs.z;
                fe += getCoulombAFM( xyzq, pos+(float3)(0,0,QZs.w) ) * Qs.w;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Keep This for debugging
    // if(iG==0){ printf("!!!!!!!! DEBUG !!!!!!!!!  __KERNEL__ evalMorseC_QZs_toImg()\n"); }
    // float amp=0.1;
    // float cosTerm   = 1.0* cos( M_PI*0.25 * pos.x) * cos( M_PI*0.25 * pos.y);
    // float cosTerm_x = 1.0* sin( M_PI*0.25 * pos.x) * cos( M_PI*0.25 * pos.y);
    // float cosTerm_y = 1.0* cos( M_PI*0.25 * pos.x) * sin( M_PI*0.25 * pos.y);
    // fe.z =cosTerm  *amp;
    // fe.x =cosTerm_x*amp;
    // fe.y =cosTerm_y*amp;
    //fe.xy+=cosTerm*0.1;

    float renorm = 100.0/fabs(fe.w);
    if(renorm<1.f){ fe*=renorm; }
    write_imagef( imgOut, (int4){ia,ib,ic,0}, fe );
}

// ========================== evalDispersion_toImg
// Compute C6/r^6 London dispersion energy grid (attractive part only)
// Uses getLondon() function for damped C6/r^6 calculation
// Parameters:
//   atoms: (x,y,z,q) - atom positions
//   C6_params: (C6_eff, 0) per atom - C6_eff = sqrt(C6_atom * C6_CO)
//   imgOut: 3D image output (energy in .w component)
//   R2damp: RA^2 - damping radius squared to avoid singularity
__kernel void evalDispersion_toImg(
    const int nAtoms,
    __global float4* atoms,         // (x,y,z,q) - positions
    __global float2* C6_params,     // (C6_eff, 0) per atom
    __write_only image3d_t imgOut,  // output energy grid
    const int4 nGrid,               // grid dimensions
    const float4 grid_p0,           // grid origin
    const float4 grid_dA,           // grid vectors
    const float4 grid_dB,
    const float4 grid_dC,
    const float R2damp              // RA^2 - damping radius squared
){
    __local float4 LATOMS[32];
    __local float2 LC6s[32];
    const int iG = get_global_id(0);
    const int iL = get_local_id(0);
    const int nL = get_local_size(0);
    const int nab = nGrid.x*nGrid.y;
    const int ia = iG%nGrid.x;
    const int ib = (iG%nab)/nGrid.x;
    const int ic = iG/nab;
    const int nMax = nab*nGrid.z;
    if(iG>=nMax) return;

    float4 fe = float4Zero;
    float3 pos = grid_p0.xyz + grid_dA.xyz*ia + grid_dB.xyz*ib + grid_dC.xyz*ic;

    // Loop over atoms in batches using local memory
    for(int i0=0; i0<nAtoms; i0+=nL){
        int i = i0 + iL;
        LATOMS[iL] = atoms[i];
        LC6s[iL] = C6_params[i];
        barrier(CLK_LOCAL_MEM_FENCE);
        for(int j=0; j<nL; j++){
            if((j+i0)<nAtoms){
                float4 xyzq = LATOMS[j];
                float3 dp = pos - xyzq.xyz;
                // getLondon computes: E = -2 * C6_eff / (r^2 + R2damp)^3
                // with RE.x=1, RE.y=C6_eff/2
                float2 RE = (float2)(1.0f, LC6s[j].x * 0.5f);
                fe += getLondon(dp, RE, R2damp);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    write_imagef(imgOut, (int4){ia,ib,ic,0}, fe);
}

// ==========================
//         GridFF
// ==========================

/*

Theory:
E(r,j) = Sum_i{ Aij* ( exp(-2k(r-Ri-Rj) - 2*exp(-k(r-Ri-Rj) }
E(r,j) = Sum_i{ Aij  ( exp(2k*Rj)*exp(-2k(r-Ri) - 2*exp(k*Rj)*exp(-k(r-Ri) ) }
E(r,j) = Aj*  exp(2k*Rj) * Sum_i{ Ai * exp(-2*k(r-Ri)) }  
       - Aj*2*exp( k*Rj) * Sum_i{ Ai * exp(- *k(r-Ri)) }

E = A * ( cP*vP + cL*vL )

cP =  Aj*  exp(2*k*Rj)
cL = -Aj*2*exp(  k*Rj)
vP =   Sum_i{ Ai * exp(-2k(r-Ri)) }
vL =   Sum_i{ Ai * exp(- k(r-Ri)) }

ej = exp( k  *Rj )
ei = exp(-k(r-Ri))

cP =  Aj*  ej*ej
cL = -Aj*2*ej
vP =   Sum_i ei*ei
vL =   Sum_i ei

*/




// ==========================
//   Gradient Computation
// ==========================

__kernel void gradient_central_diff(
    __read_only image3d_t imgIn,      // Input scalar field (energy)
    __write_only image3d_t imgOut,    // Output float4 (Fx,Fy,Fz,E)
    const float step                  // Grid spacing
) {
    // Use normalized coordinates with periodic addressing for proper BC handling
    // Pixel centers are at (i+0.5)/size in normalized coordinates
    const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_REPEAT | CLK_FILTER_NEAREST;

    int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
    int4 size = get_image_dim(imgIn);

    // Check bounds
    if (coord.x >= size.x || coord.y >= size.y || coord.z >= size.z) return;

    // Convert to normalized coordinates at pixel centers
    // Pixel i is centered at (i + 0.5) / size
    float4 fcoord = convert_float4(coord) + (float4)(0.5f, 0.5f, 0.5f, 0.0f);
    float4 norm = (float4)(1.0f / size.x, 1.0f / size.y, 1.0f / size.z, 0.0f);
    float4 center = fcoord * norm;

    // Neighbor offsets in normalized units (one pixel = 1/size)
    float4 dx = (float4)(norm.x, 0.0f, 0.0f, 0.0f);
    float4 dy = (float4)(0.0f, norm.y, 0.0f, 0.0f);
    float4 dz = (float4)(0.0f, 0.0f, norm.z, 0.0f);

    // Sample with periodic BC handled by CLK_ADDRESS_REPEAT
    float4 f_center = read_imagef(imgIn, sampler, center);
    float4 f_left_x = read_imagef(imgIn, sampler, center - dx);
    float4 f_right_x = read_imagef(imgIn, sampler, center + dx);
    float4 f_left_y = read_imagef(imgIn, sampler, center - dy);
    float4 f_right_y = read_imagef(imgIn, sampler, center + dy);
    float4 f_left_z = read_imagef(imgIn, sampler, center - dz);
    float4 f_right_z = read_imagef(imgIn, sampler, center + dz);

    // Compute gradients (negative because force = -gradient of energy)
    float grad_x = -(f_right_x.x - f_left_x.x) / (2.0f * step);
    float grad_y = -(f_right_y.x - f_left_y.x) / (2.0f * step);
    float grad_z = -(f_right_z.x - f_left_z.x) / (2.0f * step);

    // Output: (Fx, Fy, Fz, E)
    write_imagef(imgOut, coord, (float4)(grad_x, grad_y, grad_z, f_center.x));
}


// ==========================
//   FDBM fast-S3 helpers (Round 2) — NEW kernels; do not replace legacy paths
//   Host layout for float buffers: (nx,ny,nz) C-order, idx = ix*(ny*nz)+iy*nz+iz
//   FFT complex layout: (nz,ny,nx), idx = iz*(ny*nx)+iy*nx+ix
// ==========================

__kernel void fdbm_pad_roll_f32(
    __global const float* src, const int sx, const int sy, const int sz,
    __global float* dst, const int nx, const int ny, const int nz,
    const int ox, const int oy, const int oz,
    const int rx, const int ry, const int rz
) {
    // dst[i] = padded[(i+r)%n] with padded = zeros except block at (ox,oy,oz)
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    const int iz = get_global_id(2);
    if (ix >= nx || iy >= ny || iz >= nz) return;
    int jx = ix + rx; if (jx >= nx) jx -= nx; if (jx < 0) jx += nx;
    int jy = iy + ry; if (jy >= ny) jy -= ny; if (jy < 0) jy += ny;
    int jz = iz + rz; if (jz >= nz) jz -= nz; if (jz < 0) jz += nz;
    float v = 0.0f;
    const int lx = jx - ox, ly = jy - oy, lz = jz - oz;
    if (lx >= 0 && lx < sx && ly >= 0 && ly < sy && lz >= 0 && lz < sz)
        v = src[lx * (sy * sz) + ly * sz + lz];
    dst[ix * (ny * nz) + iy * nz + iz] = v;
}

__kernel void fdbm_flip3_f32(
    __global const float* src, __global float* dst,
    const int nx, const int ny, const int nz
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    const int iz = get_global_id(2);
    if (ix >= nx || iy >= ny || iz >= nz) return;
    const int jx = nx - 1 - ix, jy = ny - 1 - iy, jz = nz - 1 - iz;
    dst[ix * (ny * nz) + iy * nz + iz] = src[jx * (ny * nz) + jy * nz + jz];
}

__kernel void fdbm_xyz_to_fft_c64(
    __global const float* src, __global float2* dst,
    const int nx, const int ny, const int nz
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    const int iz = get_global_id(2);
    if (ix >= nx || iy >= ny || iz >= nz) return;
    const float v = src[ix * (ny * nz) + iy * nz + iz];
    dst[iz * (ny * nx) + iy * nx + ix] = (float2)(v, 0.0f);
}

__kernel void fdbm_fft_real_to_xyz_f32(
    __global const float2* src, __global float* dst,
    const int nx, const int ny, const int nz
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    const int iz = get_global_id(2);
    if (ix >= nx || iy >= ny || iz >= nz) return;
    dst[ix * (ny * nz) + iy * nz + iz] = src[iz * (ny * nx) + iy * nx + ix].x;
}

__kernel void fdbm_scale_pauli_pow_f32(
    __global float* field, const int n,
    const float A, const float beta, const float clip_min
) {
    const int i = get_global_id(0);
    if (i >= n) return;
    float o = field[i];
    if (o < clip_min) o = clip_min;
    field[i] = A * native_powr(o, beta);
}

__kernel void fdbm_compose_E_to_img(
    __global const float* E_pauli,
    __global const float* E_es,
    __read_only image3d_t img_vdw,
    __write_only image3d_t img_E,
    const int nx, const int ny, const int nz
) {
    const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    const int iz = get_global_id(2);
    if (ix >= nx || iy >= ny || iz >= nz) return;
    const float Ep = E_pauli[ix * (ny * nz) + iy * nz + iz];
    const float Ee = E_es[ix * (ny * nz) + iy * nz + iz];
    const float Ev = read_imagef(img_vdw, smp, (int4)(ix, iy, iz, 0)).w;
    write_imagef(img_E, (int4)(ix, iy, iz, 0), (float4)(Ep + Ee + Ev, 0.0f, 0.0f, 0.0f));
}

__kernel void fdbm_mul_poisson_tip_c64(
    __global float2* rho_k,
    __global const float2* tip_k,
    __global const float* k2_inv,
    const float scale,
    const int n
) {
    // rho_k *= tip_k * (scale * k2_inv); DC → 0
    const int i = get_global_id(0);
    if (i >= n) return;
    if (i == 0) { rho_k[0] = (float2)(0.0f, 0.0f); return; }
    const float2 a = rho_k[i];
    const float2 b = tip_k[i];
    const float s = scale * k2_inv[i];
    // complex multiply a*b then *s
    rho_k[i] = (float2)((a.x * b.x - a.y * b.y) * s, (a.x * b.y + a.y * b.x) * s);
}

// =============================================================================
// Differentiable direct Morse+Coulomb PP-AFM (Task: DifferentiableAFM_ParallelPlan)
//
// Frozen common contract (contract_version 1) — copied here per the plan so all
// agents share one source of truth. If any signature below changes, STOP all
// affected agents, increment contract_version, list invalidated outputs, redispatch.
//
// Scope: aperiodic finite molecule, 1 <= nAtoms <= 128; explicit lab-fixed scan
//   coordinates; no PBC replicas. nAtoms>128 raises on host; no silent grid fallback.
// Atom buffer: atoms.shape=(nAtoms,4) float32, rows (x,y,z,Q) [Å, e-charge].
// Morse buffer: cMs.shape=(nAtoms,4) float32, rows (R0,E0,K,0) [Å,eV,Å^-1,unused];
//   K<0 (alpha=-K>0 documentation only, not optimized in this task).
// Optimized param order: theta[i]=(x,y,z,R0,E0,Q); VJP shape (nAtoms,6) [Wave 2].
// Tip electrostatics: preserve existing tipQs, tipQZs, COULOMB_CONST, world-z
//   charge-site offsets. Tip params fixed.
// Scan input: points.shape=(nScan,4); existing flattening (ix outer, iy inner).
//   Explicit scan_p0/scan_da/scan_db mandatory; dtip<0 for the initial MVP.
// Forward output: FEs.shape=(nx,ny,nz,4) in kernel stroke order, iz=0 at the
//   initial/highest tip position; channels (Fx,Fy,Fz,E) are SAMPLE FE after force
//   rotation, not spring/total FE. No hidden z flip.
// PP telemetry: PPs.shape=(nx,ny,nz,4); .xyz = final world PP position; .w =
//   positive FIRE iteration count on convergence, negative on non-convergence/
//   nonfinite. Host raises on any negative/nonfinite entry and reports first (ix,iy,iz).
// Pair law: direct path is UNCAPPED. Do NOT reproduce evalMorseC_QZs_toImg
//   lines 1053-1054 (abs(E)>100 rescaling) — nonsmooth and force-inconsistent.
// Precision: GPU float32; oracle float64. No float64 kernels.
// Backend selection: new behavior requires explicit backend='morse_direct'.
//   Existing grid behavior and defaults unchanged.
// State flattening: iState = iScan*nz + iz. tipC.w carries dtip.
//
// Math SSOT (forward): for query q, atom a_i, d=q-a_i, r=sqrt(dot(d,d)+R2SAFE),
//   K<0, s=exp(K*(r-R0)):
//     U_M = E0*s*(s-2);  F_M = d * [-2*K*E0*s*(s-1)/r]
//   Coulomb (4 tip sites k at world z offset QZs[k], Qs pre-scaled by COULOMB_CONST):
//     U_C = sum_i sum_k Qs[k]*Q_i / sqrt(|q + zhat*QZs[k] - a_i|^2 + R2SAFE)
//   These are exactly getMorse(dp, cMs.xyz) and getCoulombAFM(atom, pos+zhat*QZs)*Qs.
//
// Frozen kernel signatures (Wave 1 forward + Wave 2 backward). Wave 1 implements
// only relaxStrokesTiltedMorseDirect; the three backward kernels are stubbed by
// Agent_1 in Wave 2 after Gate 1.
//
//   __kernel void relaxStrokesTiltedMorseDirect(
//       const int nAtoms, __global const float4* atoms, __global const float4* cMs,
//       __global const float4* points, __global float4* FEs, __global float4* PPs,
//       float4 tipA, float4 tipB, float4 tipC, float4 stiffness, float4 dpos0,
//       float4 relax_params, float4 surfFF, float4 Qs, float4 QZs,
//       const int nScan, const int nz, __local float4* LATOMS, __local float4* LCMS);
//
//   __kernel void morseDirectStateAdjoint(
//       const int nAtoms, __global const float4* atoms, __global const float4* cMs,
//       __global const float4* points, __global const float4* PPs,
//       __global const float4* dL_dFEs, __global float4* lambdas,
//       __global float4* adjoint_diag,
//       float4 tipA, float4 tipB, float4 tipC, float4 stiffness, float4 dpos0,
//       float4 surfFF, float4 Qs, float4 QZs, const int nScan, const int nz,
//       __local float4* LATOMS, __local float4* LCMS);
//
//   __kernel void morseDirectParamPartials(
//       const int nAtoms, __global const float4* atoms, __global const float4* cMs,
//       __global const float4* points, __global const float4* PPs,
//       __global const float4* dL_dFEs, __global const float4* lambdas,
//       __global float4* partial_xR, __global float4* partial_EQ,
//       float4 tipA, float4 tipB, float4 tipC, float4 Qs, float4 QZs,
//       const int nScan, const int nz);
//
//   __kernel void reduceMorseDirectParamPartials(
//       const int nAtoms, const int nScan, __global const float4* partial_xR,
//       __global const float4* partial_EQ, __global float4* grad_xR,
//       __global float4* grad_EQ, __local float4* L_xR, __local float4* L_EQ);
//
// Local-memory requirement (Agent_1 → Agent_2 via coordinator):
//   relaxStrokesTiltedMorseDirect needs LATOMS and LCMS each of size nAtoms*16 bytes
//   (dynamic __local allocations passed as kernel args). For nAtoms<=128 that is
//   <=2048 bytes each, <=4096 bytes total — well within the 48 KB per-CU limit.
//   Workgroup size is chosen by the host (candidates {32,64,128}); the kernel uses
//   get_local_size(0) for the cooperative preload stride and places no barrier
//   inside the relaxation loop.
// =============================================================================

// ---- Reusable inline direct Morse+Coulomb FE sum over local-memory atoms ----
// Sums the uncapped Morse + 4-site Coulomb pair law over nAtoms preloaded in
// LATOMS/LCMS. Qs must already be pre-scaled by COULOMB_CONST by the caller.
// Matches cs_brute_afm_morse_c_points / evalMorseC_QZs_toImg pair law EXACTLY
// minus the abs(E)>100 rescale (which is intentionally omitted here).
// Reused by the forward relaxation loop (every FIRE step + final re-eval) and,
// in Wave 2, by the fixed-query validation and implicit-VJP passes.
inline float4 evalMorseCDirect_local(
    float3 pos,
    __local const float4* LATOMS, __local const float4* LCMS,
    const int nAtoms, float4 Qs, float4 QZs
){
    float4 fe = float4Zero;
    for(int j=0; j<nAtoms; j++){
        float4 xyzq = LATOMS[j];
        float3 dp   = pos - xyzq.xyz;
        fe += getMorse( dp, LCMS[j].xyz );
        fe += getCoulombAFM( xyzq, pos + (float3)(0.0f, 0.0f, QZs.x) ) * Qs.x;
        fe += getCoulombAFM( xyzq, pos + (float3)(0.0f, 0.0f, QZs.y) ) * Qs.y;
        fe += getCoulombAFM( xyzq, pos + (float3)(0.0f, 0.0f, QZs.z) ) * Qs.z;
        fe += getCoulombAFM( xyzq, pos + (float3)(0.0f, 0.0f, QZs.w) ) * Qs.w;
    }
    return fe;
}

// ---- Wave 1: direct Morse+Coulomb PP-relaxation, one work-item per scan lane ----
// Preserves the exact relaxStrokesTilted scan/rotation semantics (dTip=tipC.xyz*tipC.w,
// dpos0 rotated by rotMatT, tip-rotated FE output, iz=0 at initial/highest tip pos,
// tipPos and pos both advanced by dTip per z slice). Replaces the interpolated grid
// FE sample with a direct sum over atoms preloaded ONCE into local memory.
//
// Fail-loud contract: PPs[iState].w is +iter on convergence, -iter on non-convergence
// or nonfinite state. No clipping, no fallback, no cap. The host raises on any
// negative/nonfinite entry. Padded scan lanes (iScan>=nScan) MUST participate in the
// preload + barrier and only then become inactive (no early return before barrier).
__kernel void relaxStrokesTiltedMorseDirect(
    const int nAtoms, __global const float4* atoms, __global const float4* cMs,
    __global const float4* points, __global float4* FEs, __global float4* PPs,
    float4 tipA, float4 tipB, float4 tipC, float4 stiffness, float4 dpos0,
    float4 relax_params, float4 surfFF, float4 Qs, float4 QZs,
    const int nScan, const int nz, __local float4* LATOMS, __local float4* LCMS
){
    const int iScan = get_global_id(0);
    const int iL    = get_local_id(0);
    const int nL    = get_local_size(0);

    // --- Cooperative preload of all atoms + cMs once (bounds-guarded) ---
    // All lanes including padded scan lanes participate, then hit the barrier.
    for(int i=iL; i<nAtoms; i+=nL){
        LATOMS[i] = atoms[i];
        LCMS  [i] = cMs  [i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Padded scan lane becomes inactive ONLY after the preload barrier.
    if(iScan >= nScan) return;

    // --- Tip geometry (identical to relaxStrokesTilted) ---
    const float3 dTip  = tipC.xyz * tipC.w;                       // per-z step (tipC.w = dtip)
    float4 dpos0_ = dpos0;
    dpos0_.xyz = rotMatT( dpos0_.xyz, tipA.xyz, tipB.xyz, tipC.xyz );
    float3 tipPos = points[iScan].xyz;                            // scan start (iz=0, highest)
    float3 pos    = tipPos + dpos0_.xyz;                          // initial PP world pos

    // --- FIRE relaxation params (identical to relaxStrokesTilted) ---
    float dt   = relax_params.x;
    float damp = relax_params.y;
    float dtmax = dt;
    float dtmin = dtmax * 0.1f;
    float damp0 = damp;

    // Coulomb tip charges pre-scaled once (matches cs_brute_afm_morse_c_points).
    float4 Qs_ = Qs * COULOMB_CONST;

    // --- Per-z relaxation loop (NO barrier inside) ---
    for(int iz=0; iz<nz; iz++){
        float3 v = float3Zero;
        float4 fe = float4Zero;
        int itr = 0;
        bool converged = false;
        for(int i=0; i<N_RELAX_STEP_MAX; i++){
            itr = i + 1;
            fe = evalMorseCDirect_local( pos, LATOMS, LCMS, nAtoms, Qs_, QZs );
            float3 f = fe.xyz;
            float3 dpos  = pos - tipPos;
            float3 dpos_ = rotMat  ( dpos, tipA.xyz, tipB.xyz, tipC.xyz );   // to tip-coords
            float3 ftip  = tipForce( dpos_, stiffness, dpos0 );
            f += rotMatT ( ftip, tipA.xyz, tipB.xyz, tipC.xyz );             // back to world
            f += tipC.xyz * surfFF.x;                                       // surface bias

            // Convergence check BEFORE the position update — the stored PP must
            // be at the true equilibrium where |f|<F2CONV, not one step past it.
            // (The original relaxStrokesTilted checks after the update, but it
            // never stored PPs; the differentiable VJP needs the equilibrium pos.)
            if( dot(f,f) < F2CONV ){ converged = true; break; }

            #if OPT_FIRE
            v = update_FIRE( f, v, &dt, &damp, dtmin, dtmax, damp0 );
            #else
            v *= (1 - damp);
            #endif
            v   += f * dt;
            pos += v * dt;
        }

        // Final direct FE re-evaluation at the converged/last PP position.
        fe = evalMorseCDirect_local( pos, LATOMS, LCMS, nAtoms, Qs_, QZs );

        // Output tip-rotated SAMPLE FE (channels Fx,Fy,Fz,E); .w = sample energy.
        float4 fe_ = fe;
        fe_.xyz = rotMat( fe.xyz, tipA.xyz, tipB.xyz, tipC.xyz );
        fe_.w   = fe.w;
        const int iState = iScan * nz + iz;
        FEs[iState] = fe_;

        // PP telemetry: final world PP position + signed iteration count.
        // +itr on convergence; -itr on non-convergence or nonfinite state.
        const bool finite = isfinite(fe.x) && isfinite(fe.y) && isfinite(fe.z) && isfinite(fe.w)
                         && isfinite(pos.x) && isfinite(pos.y) && isfinite(pos.z);
        int itw = converged ? itr : -itr;
        if( !finite ) itw = -itr;
        PPs[iState] = (float4)( pos, (float)itw );

        // Advance to next z slice (both tipPos and pos move by dTip).
        tipPos += dTip;
        pos    += dTip;
    }
}

// =============================================================================
// Wave 2: Implicit adjoint/VJP for differentiable direct Morse+Coulomb PP-AFM
// (Task: DifferentiableAFM_ParallelPlan, contract_version 1)
//
// Three backward passes (frozen layout, no float atomics, no dense Jacobian):
//   1. morseDirectStateAdjoint       — one work-item per (scan,z): re-evaluates
//      total force/Jacobian at stored PPs, consumes dL_dFEs, solves J*lambda=b,
//      writes lambda + adjoint_diag=(residual_norm, lambda_min, cond_est, status).
//   2. morseDirectParamPartials      — one work-item per (scan,atom): loops over z,
//      consumes lambda/PPs/dL_dFEs, writes per-(scan,atom) partial_xR=(x,y,z,R0)
//      and partial_EQ=(E0,Q,0,0).
//   3. reduceMorseDirectParamPartials — one workgroup per atom: reduces scan dim,
//      writes grad_xR[atom] and grad_EQ[atom]; host packs into (nAtoms,6).
//
// Math SSOT (backward): at relaxed q*, G=F_sample+F_tip+F_surf=0, J=dG/dq.
//   J = -H_E + R^T*(dg/dd)*R  (symmetric; H_E=sample energy Hessian, dg/dd=tip
//   force Jacobian in tip coords, R=rotMat world->tip). For output O=FEs, upstream
//   u=dL/dO, b=(dO/dq)^T*u = -H_E*rotMatT(u.xyz) - F_sample*u.w. Solve J*lambda=b.
//   dL/dtheta = u^T*(dO/dtheta) - lambda^T*(dG/dtheta). With w=u_world-lambda:
//     dL/d(a_i.xyz) = H_E_i*w + (F_M_i+F_C_i)*u.w
//     dL/dR0_i      = w·(dF_M_i/dR0) + u.w*dE_M_i/dR0
//     dL/dE0_i      = w·(dF_M_i/dE0) + u.w*dE_M_i/dE0
//     dL/dQ_i       = w·(dF_C_i/dQ)  + u.w*dE_C_i/dQ
//   Per-atom pair quantities match the forward pair law (getMorse+getCoulombAFM)
//   exactly. Qs pre-scaled by COULOMB_CONST inside each kernel (same as forward).
//
// Status codes: 0=success, 1=nonconvergence, 2=nonfinite, 3=instability, 4=singularity.
// Acceptance gates: residual<1e-4 eV/A, lambda_min>1e-5 eV/A^2, cond_est<1e6.
// =============================================================================

// ---- Sample FE + energy Hessian from local-memory atoms (pass 1 helper) ----
// Computes total sample (Fx,Fy,Fz,E) and d²E_sample/dq² (3x3, stored as 3 rows).
// Qs must be pre-scaled by COULOMB_CONST. Matches evalMorseCDirect_local pair law.
inline void evalMorseCDirect_FE_Hessian_local(
    float3 pos, __local const float4* LATOMS, __local const float4* LCMS,
    const int nAtoms, float4 Qs, float4 QZs,
    float4* fe_out, float3* H0, float3* H1, float3* H2
){
    float4 fe = float4Zero;
    float3 h0 = float3Zero, h1 = float3Zero, h2 = float3Zero;
    for(int j=0; j<nAtoms; j++){
        float4 xyzq = LATOMS[j];
        float4 cm   = LCMS[j];
        // --- Morse ---
        float3 d = pos - xyzq.xyz;
        float r2 = dot(d,d) + R2SAFE;
        float r  = sqrt(r2);
        float R0 = cm.x, E0 = cm.y, K = cm.z;
        float s  = exp(K*(r - R0));
        float Ep = 2.0f*K*E0*s*(s - 1.0f);       // dE/dr
        float Epp = 2.0f*K*K*E0*s*(2.0f*s - 1.0f); // d²E/dr²
        float fr = -Ep;                            // -dE/dr (force coeff)
        fe += (float4)(d*(fr/r), E0*s*(s - 2.0f));
        // H_M = [Epp - Ep/r]*(d⊗d)/r² + (Ep/r)*I
        float c_out = (Epp - Ep/r) / r2;
        float c_dia = Ep / r;
        h0 += (float3)(c_out*d.x*d.x + c_dia, c_out*d.x*d.y,         c_out*d.x*d.z);
        h1 += (float3)(c_out*d.y*d.x,         c_out*d.y*d.y + c_dia, c_out*d.y*d.z);
        h2 += (float3)(c_out*d.z*d.x,         c_out*d.z*d.y,         c_out*d.z*d.z + c_dia);
        // --- Coulomb (4 tip sites) ---
        // H_C = 3*C*(d⊗d)/r⁵ - C*I/r³, C=Q_i*Qs[k]
        float Qi = xyzq.w;
        // site x
        { float3 dk = (pos + (float3)(0.0f,0.0f,QZs.x)) - xyzq.xyz;
          float rk2 = dot(dk,dk)+R2SAFE; float rk=sqrt(rk2); float rk3=rk*rk2; float rk5=rk3*rk2;
          float C = Qi*Qs.x; float co=3.0f*C/rk5; float cd=-C/rk3;
          fe += (float4)(dk*(C/(rk3)), C/rk);
          h0 += (float3)(co*dk.x*dk.x+cd, co*dk.x*dk.y,       co*dk.x*dk.z);
          h1 += (float3)(co*dk.y*dk.x,       co*dk.y*dk.y+cd, co*dk.y*dk.z);
          h2 += (float3)(co*dk.z*dk.x,       co*dk.z*dk.y,    co*dk.z*dk.z+cd); }
        // site y
        { float3 dk = (pos + (float3)(0.0f,0.0f,QZs.y)) - xyzq.xyz;
          float rk2 = dot(dk,dk)+R2SAFE; float rk=sqrt(rk2); float rk3=rk*rk2; float rk5=rk3*rk2;
          float C = Qi*Qs.y; float co=3.0f*C/rk5; float cd=-C/rk3;
          fe += (float4)(dk*(C/(rk3)), C/rk);
          h0 += (float3)(co*dk.x*dk.x+cd, co*dk.x*dk.y,       co*dk.x*dk.z);
          h1 += (float3)(co*dk.y*dk.x,       co*dk.y*dk.y+cd, co*dk.y*dk.z);
          h2 += (float3)(co*dk.z*dk.x,       co*dk.z*dk.y,    co*dk.z*dk.z+cd); }
        // site z
        { float3 dk = (pos + (float3)(0.0f,0.0f,QZs.z)) - xyzq.xyz;
          float rk2 = dot(dk,dk)+R2SAFE; float rk=sqrt(rk2); float rk3=rk*rk2; float rk5=rk3*rk2;
          float C = Qi*Qs.z; float co=3.0f*C/rk5; float cd=-C/rk3;
          fe += (float4)(dk*(C/(rk3)), C/rk);
          h0 += (float3)(co*dk.x*dk.x+cd, co*dk.x*dk.y,       co*dk.x*dk.z);
          h1 += (float3)(co*dk.y*dk.x,       co*dk.y*dk.y+cd, co*dk.y*dk.z);
          h2 += (float3)(co*dk.z*dk.x,       co*dk.z*dk.y,    co*dk.z*dk.z+cd); }
        // site w
        { float3 dk = (pos + (float3)(0.0f,0.0f,QZs.w)) - xyzq.xyz;
          float rk2 = dot(dk,dk)+R2SAFE; float rk=sqrt(rk2); float rk3=rk*rk2; float rk5=rk3*rk2;
          float C = Qi*Qs.w; float co=3.0f*C/rk5; float cd=-C/rk3;
          fe += (float4)(dk*(C/(rk3)), C/rk);
          h0 += (float3)(co*dk.x*dk.x+cd, co*dk.x*dk.y,       co*dk.x*dk.z);
          h1 += (float3)(co*dk.y*dk.x,       co*dk.y*dk.y+cd, co*dk.y*dk.z);
          h2 += (float3)(co*dk.z*dk.x,       co*dk.z*dk.y,    co*dk.z*dk.z+cd); }
    }
    *fe_out = fe; *H0 = h0; *H1 = h1; *H2 = h2;
}

// ---- Tip force 3x3 Jacobian dg/dd in tip coords ----
// g(d) = (d - d0.xyz)*k.xyz + d*(k.w*(1 - d0.w/r)), r=|d|
// dg/dd = diag(k.xyz) + phi*I + (k.w*d0.w/r³)*(d⊗d), phi=k.w*(1-d0.w/r)
inline void tipForceJacobian(
    float3 d, float4 stiffness, float4 dpos0,
    float3* J0, float3* J1, float3* J2
){
    float r = sqrt(dot(d,d));
    r = fmax(r, 1e-10f);
    float phi  = stiffness.w * (1.0f - dpos0.w / r);
    float beta = stiffness.w * dpos0.w / (r*r*r);  // outer-product coeff
    *J0 = (float3)(stiffness.x + phi + beta*d.x*d.x, beta*d.x*d.y,         beta*d.x*d.z);
    *J1 = (float3)(beta*d.y*d.x,         stiffness.y + phi + beta*d.y*d.y, beta*d.y*d.z);
    *J2 = (float3)(beta*d.z*d.x,         beta*d.z*d.y,         stiffness.z + phi + beta*d.z*d.z);
}

// ---- Rotate 3x3 matrix: J_world = R^T * J_tip * R, R=rotMat(world->tip) ----
// R rows = (a,b,c). R*v=rotMat(v,a,b,c). R^T*v=rotMatT(v,a,b,c).
inline void matRT_M_R(
    float3 j0, float3 j1, float3 j2, float3 a, float3 b, float3 c,
    float3* o0, float3* o1, float3* o2
){
    // M = J_tip * R  →  M_row_i = rotMatT(J_tip_row_i, a, b, c)
    float3 m0 = rotMatT(j0, a, b, c);
    float3 m1 = rotMatT(j1, a, b, c);
    float3 m2 = rotMatT(j2, a, b, c);
    // J_world = R^T * M  →  row_i = a[i]*m0 + b[i]*m1 + c[i]*m2
    *o0 = a.x*m0 + b.x*m1 + c.x*m2;
    *o1 = a.y*m0 + b.y*m1 + c.y*m2;
    *o2 = a.z*m0 + b.z*m1 + c.z*m2;
}

// ---- Solve symmetric 3x3 system J*lambda = b via cofactors ----
// Returns 0 on success, 4 if singular (|det| < eps). J given as 3 rows.
inline int solve3x3_symmetric(
    float3 j0, float3 j1, float3 j2, float3 b, float3* lambda, float* det_out
){
    float j00=j0.x, j01=j0.y, j02=j0.z;
    float j11=j1.y, j12=j1.z;
    float j22=j2.z;
    // Adjugate (symmetric: j01=j10, j02=j20, j12=j21)
    float a00 = j11*j22 - j12*j12;
    float a01 = j02*j12 - j01*j22;
    float a02 = j01*j12 - j02*j11;
    float a11 = j00*j22 - j02*j02;
    float a12 = j01*j02 - j00*j12;
    float a22 = j00*j11 - j01*j01;
    float det = j00*a00 + j01*a01 + j02*a02;  // = j00*(j11*j22-j12²) - j01*(j01*j22-j12*j02) + j02*(j01*j12-j02*j11)
    *det_out = det;
    float adet = fabs(det);
    // Scale-aware singularity threshold
    float scale = fmax(fabs(j00), fmax(fabs(j11), fabs(j22)));
    scale = fmax(scale, 1e-20f);
    if(adet < 1e-18f * scale * scale * scale) return 4;  // singular
    float inv_det = 1.0f / det;
    lambda->x = (a00*b.x + a01*b.y + a02*b.z) * inv_det;
    lambda->y = (a01*b.x + a11*b.y + a12*b.z) * inv_det;
    lambda->z = (a02*b.x + a12*b.y + a22*b.z) * inv_det;
    return 0;
}

// ---- Smallest eigenvalue of 3x3 symmetric matrix (H = -J for stability check) ----
// Uses the analytical formula for 3x3 symmetric eigenvalues.
inline float eigenmin3x3_symmetric(float3 h0, float3 h1, float3 h2){
    float p1 = h0.y*h0.y + h0.z*h0.z + h1.z*h1.z;  // off-diagonal² sum
    float q  = (h0.x + h1.y + h2.z) / 3.0f;         // trace/3
    float p2 = (h0.x-q)*(h0.x-q) + (h1.y-q)*(h1.y-q) + (h2.z-q)*(h2.z-q) + 2.0f*p1;
    float p  = sqrt(p2 / 6.0f);
    if(p < 1e-20f) return q;  // nearly q*I
    // B = (1/p)*(H - q*I)
    float b00=(h0.x-q)/p, b11=(h1.y-q)/p, b22=(h2.z-q)/p;
    float b01=h0.y/p, b02=h0.z/p, b12=h1.z/p;
    float detB = b00*(b11*b22-b12*b12) - b01*(b01*b22-b12*b02) + b02*(b01*b12-b11*b02);
    float r = detB * 0.5f;
    r = fmax(-1.0f, fmin(1.0f, r));  // clamp for acos
    float phi = acos(r) / 3.0f;
    float eig1 = q + 2.0f*p*cos(phi);
    float eig3 = q + 2.0f*p*cos(phi + 2.0f*M_PI_F/3.0f);
    float eig2 = 3.0f*q - eig1 - eig3;
    return fmin(eig1, fmin(eig2, eig3));
}

// ---- Condition estimate |J|_inf * |Jinv|_inf for 3x3 symmetric J ----
inline float condition_estimate3x3(float3 j0, float3 j1, float3 j2, float det){
    float jinf = fmax(fabs(j0.x)+fabs(j0.y)+fabs(j0.z),
               fmax(fabs(j1.x)+fabs(j1.y)+fabs(j1.z),
                    fabs(j2.x)+fabs(j2.y)+fabs(j2.z)));
    float j00=j0.x, j01=j0.y, j02=j0.z, j11=j1.y, j12=j1.z, j22=j2.z;
    float a00=j11*j22-j12*j12, a01=j02*j12-j01*j22, a02=j01*j12-j02*j11;
    float a11=j00*j22-j02*j02, a12=j01*j02-j00*j12, a22=j00*j11-j01*j01;
    float ainf = fmax(fabs(a00)+fabs(a01)+fabs(a02),
               fmax(fabs(a01)+fabs(a11)+fabs(a12),
                    fabs(a02)+fabs(a12)+fabs(a22)));
    float adet = fmax(fabs(det), 1e-30f);
    return jinf * (ainf / adet);
}

// ---- Pass 1: per-state adjoint (one work-item per (scan,z) state) ----
// Re-evaluates total force G and Jacobian J at stored PPs, solves J*lambda=b,
// writes lambdas[iState] and adjoint_diag[iState].
__kernel void morseDirectStateAdjoint(
    const int nAtoms, __global const float4* atoms, __global const float4* cMs,
    __global const float4* points, __global const float4* PPs,
    __global const float4* dL_dFEs, __global float4* lambdas,
    __global float4* adjoint_diag,
    float4 tipA, float4 tipB, float4 tipC, float4 stiffness, float4 dpos0,
    float4 surfFF, float4 Qs, float4 QZs, const int nScan, const int nz,
    __local float4* LATOMS, __local float4* LCMS
){
    const int iL = get_local_id(0);
    const int nL = get_local_size(0);
    // Cooperative preload (same pattern as forward kernel)
    for(int i=iL; i<nAtoms; i+=nL){
        LATOMS[i] = atoms[i];
        LCMS  [i] = cMs  [i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int iState = get_global_id(0);
    if(iState >= nScan * nz) return;

    const int iScan = iState / nz;
    const int iz    = iState % nz;

    // Read PP and upstream gradient
    float4 pp = PPs[iState];
    float4 u  = dL_dFEs[iState];
    float3 qstar = pp.xyz;

    // --- Check finiteness (status 2) ---
    if(!all(isfinite(pp)) || !all(isfinite(u))){
        lambdas[iState] = float4Zero;
        adjoint_diag[iState] = (float4)(0.0f, 0.0f, 0.0f, 2.0f);
        return;
    }
    // --- Check forward convergence (status 1) ---
    if(pp.w <= 0.0f){
        lambdas[iState] = float4Zero;
        adjoint_diag[iState] = (float4)(0.0f, 0.0f, 0.0f, 1.0f);
        return;
    }

    // Reconstruct tipPos for this z slice
    float3 dTip = tipC.xyz * tipC.w;
    float3 tipPos = points[iScan].xyz + dTip * (float)iz;
    float3 a = tipA.xyz, b = tipB.xyz, c = tipC.xyz;
    float4 Qs_ = Qs * COULOMB_CONST;

    // Sample FE + Hessian at q*
    float4 fe; float3 H0, H1, H2;
    evalMorseCDirect_FE_Hessian_local(qstar, LATOMS, LCMS, nAtoms, Qs_, QZs, &fe, &H0, &H1, &H2);
    float3 F_sample = fe.xyz;

    // Tip force + surface bias
    float3 dpos  = qstar - tipPos;
    float3 dpos_ = rotMat(dpos, a, b, c);
    float3 ftip  = tipForce(dpos_, stiffness, dpos0);
    float3 F_tip = rotMatT(ftip, a, b, c);
    float3 F_surf = c * surfFF.x;

    // Residual G = F_sample + F_tip + F_surf
    float3 G = F_sample + F_tip + F_surf;
    float residual = sqrt(dot(G, G));

    // --- Check residual (status 1) ---
    // Forward F2CONV=1e-8 → |f|<1e-4 at convergence. Allow 5x margin for float32
    // re-evaluation roundoff (force sum order differs from forward kernel).
    if(residual >= 5e-4f){
        lambdas[iState] = float4Zero;
        adjoint_diag[iState] = (float4)(residual, 0.0f, 0.0f, 1.0f);
        return;
    }

    // Check H finiteness
    if(!all(isfinite(H0)) || !all(isfinite(H1)) || !all(isfinite(H2))){
        lambdas[iState] = float4Zero;
        adjoint_diag[iState] = (float4)(residual, 0.0f, 0.0f, 2.0f);
        return;
    }

    // Tip force Jacobian dg/dd (tip coords) → J_world = R^T * dg/dd * R
    float3 Jt0, Jt1, Jt2;
    tipForceJacobian(dpos_, stiffness, dpos0, &Jt0, &Jt1, &Jt2);
    float3 Jw0, Jw1, Jw2;
    matRT_M_R(Jt0, Jt1, Jt2, a, b, c, &Jw0, &Jw1, &Jw2);

    // J = -H_E + R^T*(dg/dd)*R  (symmetric)
    float3 J0 = -H0 + Jw0;
    float3 J1 = -H1 + Jw1;
    float3 J2 = -H2 + Jw2;

    // b = -H_E*rotMatT(u.xyz) - F_sample*u.w
    float3 u_world = rotMatT(u.xyz, a, b, c);
    float3 b_vec = -(float3)(dot(H0,u_world), dot(H1,u_world), dot(H2,u_world)) - F_sample * u.w;

    // Solve J*lambda = b
    float3 lambda; float det;
    int solve_status = solve3x3_symmetric(J0, J1, J2, b_vec, &lambda, &det);

    // --- Check singularity (status 4) ---
    if(solve_status != 0){
        lambdas[iState] = float4Zero;
        adjoint_diag[iState] = (float4)(residual, 0.0f, 0.0f, 4.0f);
        return;
    }
    if(!all(isfinite(lambda))){
        lambdas[iState] = float4Zero;
        adjoint_diag[iState] = (float4)(residual, 0.0f, 0.0f, 2.0f);
        return;
    }

    // lambda_min of H = -J (stability: H must be positive definite)
    float lambda_min = eigenmin3x3_symmetric(-J0, -J1, -J2);

    // --- Check instability (status 3) ---
    if(lambda_min <= 1e-5f){
        lambdas[iState] = float4Zero;
        adjoint_diag[iState] = (float4)(residual, lambda_min, 0.0f, 3.0f);
        return;
    }

    // Condition estimate
    float cond = condition_estimate3x3(J0, J1, J2, det);

    // --- Check condition (status 4) ---
    if(cond >= 1e6f || !isfinite(cond)){
        lambdas[iState] = float4Zero;
        adjoint_diag[iState] = (float4)(residual, lambda_min, cond, 4.0f);
        return;
    }

    // --- Success (status 0) ---
    lambdas[iState] = (float4)(lambda, 0.0f);
    adjoint_diag[iState] = (float4)(residual, lambda_min, cond, 0.0f);
}

// ---- Pass 2: per-scan/atom parameter partials (one work-item per (scan,atom)) ----
// Loops over z, consumes lambda/PPs/dL_dFEs, writes partial_xR and partial_EQ.
// Buffer layout: atom-major, partial_xR[iAtom*nScan + iScan] (coalesced for pass 3).
__kernel void morseDirectParamPartials(
    const int nAtoms, __global const float4* atoms, __global const float4* cMs,
    __global const float4* points, __global const float4* PPs,
    __global const float4* dL_dFEs, __global const float4* lambdas,
    __global float4* partial_xR, __global float4* partial_EQ,
    float4 tipA, float4 tipB, float4 tipC, float4 Qs, float4 QZs,
    const int nScan, const int nz
){
    const int gid = get_global_id(0);
    if(gid >= nScan * nAtoms) return;
    const int iScan = gid / nAtoms;
    const int iAtom = gid % nAtoms;

    float3 a = tipA.xyz, b = tipB.xyz, c = tipC.xyz;
    float3 dTip = tipC.xyz * tipC.w;
    float4 Qs_ = Qs * COULOMB_CONST;
    float4 atom = atoms[iAtom];
    float4 cm   = cMs[iAtom];
    float3 tipPos0 = points[iScan].xyz;

    float4 sum_xR = float4Zero;  // (x, y, z, R0)
    float4 sum_EQ = float4Zero;  // (E0, Q, 0, 0)

    for(int iz=0; iz<nz; iz++){
        int iState = iScan * nz + iz;
        float4 pp = PPs[iState];
        float4 u  = dL_dFEs[iState];
        float4 lam = lambdas[iState];

        // Skip non-converged or nonfinite states (contribute 0)
        if(pp.w <= 0.0f || !all(isfinite(pp)) || !all(isfinite(u)) || !all(isfinite(lam)))
            continue;
        // Skip states where adjoint failed (lambda is zero but diag status != 0)
        // The host raises on any failure in pass 1, so if we reach here all states
        // are valid. But defensive: if lambda and u are both zero, skip.
        if(dot(u,u) == 0.0f && dot(lam.xyz,lam.xyz) == 0.0f) continue;

        float3 qstar = pp.xyz;
        float3 tipPos = tipPos0 + dTip * (float)iz;

        // u_world = rotMatT(u.xyz) — convert upstream from tip-rotated to world
        float3 u_world = rotMatT(u.xyz, a, b, c);
        float3 lambda  = lam.xyz;
        float3 w = u_world - lambda;  // combined adjoint weight
        float uw = u.w;

        float3 dpos_ = rotMat(qstar - tipPos, a, b, c);

        // --- Per-atom Morse pair quantities at q* ---
        float3 d_m = qstar - atom.xyz;
        float r2_m = dot(d_m, d_m) + R2SAFE;
        float r_m  = sqrt(r2_m);
        float R0 = cm.x, E0 = cm.y, K = cm.z;
        float s  = exp(K*(r_m - R0));
        float Ep = 2.0f*K*E0*s*(s - 1.0f);        // dE_M/dr
        float Epp = 2.0f*K*K*E0*s*(2.0f*s - 1.0f); // d²E_M/dr²
        float fr_m = -Ep;                          // -dE_M/dr
        float3 F_M = d_m * (fr_m / r_m);           // Morse force on probe
        float  E_M = E0*s*(s - 2.0f);
        // dF_M/dR0 = d_m * 2*K²*E0*s*(2s-1) / r
        float3 dF_dR0 = d_m * (2.0f*K*K*E0*s*(2.0f*s - 1.0f) / r_m);
        // dF_M/dE0 = d_m * (-2*K*s*(s-1)) / r
        float3 dF_dE0 = d_m * (-2.0f*K*s*(s - 1.0f) / r_m);
        float dE_dR0 = -2.0f*K*E0*s*(s - 1.0f);
        float dE_dE0 = s*(s - 2.0f);
        // Per-atom Morse Hessian H_M_i
        float c_out_m = (Epp - Ep/r_m) / r2_m;
        float c_dia_m = Ep / r_m;
        float3 HMi0 = (float3)(c_out_m*d_m.x*d_m.x + c_dia_m, c_out_m*d_m.x*d_m.y,         c_out_m*d_m.x*d_m.z);
        float3 HMi1 = (float3)(c_out_m*d_m.y*d_m.x,         c_out_m*d_m.y*d_m.y + c_dia_m, c_out_m*d_m.y*d_m.z);
        float3 HMi2 = (float3)(c_out_m*d_m.z*d_m.x,         c_out_m*d_m.z*d_m.y,         c_out_m*d_m.z*d_m.z + c_dia_m);

        // --- Per-atom Coulomb pair quantities (4 tip sites) ---
        float3 F_C = float3Zero;
        float  E_C = 0.0f;
        float3 dF_dQ = float3Zero;  // dF_C/dQ_i
        float  dE_dQ = 0.0f;        // dE_C/dQ_i
        float3 HC0 = float3Zero, HC1 = float3Zero, HC2 = float3Zero;
        float Qi = atom.w;
        // site x
        { float3 dk = (qstar + (float3)(0.0f,0.0f,QZs.x)) - atom.xyz;
          float rk2=dot(dk,dk)+R2SAFE; float rk=sqrt(rk2); float rk3=rk*rk2; float rk5=rk3*rk2;
          float C=Qi*Qs_.x; float ir3=C/rk3;
          F_C += dk*ir3; E_C += C/rk;
          dF_dQ += dk*(Qs_.x/rk3); dE_dQ += Qs_.x/rk;
          float co=3.0f*C/rk5, cd=-C/rk3;
          HC0 += (float3)(co*dk.x*dk.x+cd, co*dk.x*dk.y,       co*dk.x*dk.z);
          HC1 += (float3)(co*dk.y*dk.x,       co*dk.y*dk.y+cd, co*dk.y*dk.z);
          HC2 += (float3)(co*dk.z*dk.x,       co*dk.z*dk.y,    co*dk.z*dk.z+cd); }
        // site y
        { float3 dk = (qstar + (float3)(0.0f,0.0f,QZs.y)) - atom.xyz;
          float rk2=dot(dk,dk)+R2SAFE; float rk=sqrt(rk2); float rk3=rk*rk2; float rk5=rk3*rk2;
          float C=Qi*Qs_.y; float ir3=C/rk3;
          F_C += dk*ir3; E_C += C/rk;
          dF_dQ += dk*(Qs_.y/rk3); dE_dQ += Qs_.y/rk;
          float co=3.0f*C/rk5, cd=-C/rk3;
          HC0 += (float3)(co*dk.x*dk.x+cd, co*dk.x*dk.y,       co*dk.x*dk.z);
          HC1 += (float3)(co*dk.y*dk.x,       co*dk.y*dk.y+cd, co*dk.y*dk.z);
          HC2 += (float3)(co*dk.z*dk.x,       co*dk.z*dk.y,    co*dk.z*dk.z+cd); }
        // site z
        { float3 dk = (qstar + (float3)(0.0f,0.0f,QZs.z)) - atom.xyz;
          float rk2=dot(dk,dk)+R2SAFE; float rk=sqrt(rk2); float rk3=rk*rk2; float rk5=rk3*rk2;
          float C=Qi*Qs_.z; float ir3=C/rk3;
          F_C += dk*ir3; E_C += C/rk;
          dF_dQ += dk*(Qs_.z/rk3); dE_dQ += Qs_.z/rk;
          float co=3.0f*C/rk5, cd=-C/rk3;
          HC0 += (float3)(co*dk.x*dk.x+cd, co*dk.x*dk.y,       co*dk.x*dk.z);
          HC1 += (float3)(co*dk.y*dk.x,       co*dk.y*dk.y+cd, co*dk.y*dk.z);
          HC2 += (float3)(co*dk.z*dk.x,       co*dk.z*dk.y,    co*dk.z*dk.z+cd); }
        // site w
        { float3 dk = (qstar + (float3)(0.0f,0.0f,QZs.w)) - atom.xyz;
          float rk2=dot(dk,dk)+R2SAFE; float rk=sqrt(rk2); float rk3=rk*rk2; float rk5=rk3*rk2;
          float C=Qi*Qs_.w; float ir3=C/rk3;
          F_C += dk*ir3; E_C += C/rk;
          dF_dQ += dk*(Qs_.w/rk3); dE_dQ += Qs_.w/rk;
          float co=3.0f*C/rk5, cd=-C/rk3;
          HC0 += (float3)(co*dk.x*dk.x+cd, co*dk.x*dk.y,       co*dk.x*dk.z);
          HC1 += (float3)(co*dk.y*dk.x,       co*dk.y*dk.y+cd, co*dk.y*dk.z);
          HC2 += (float3)(co*dk.z*dk.x,       co*dk.z*dk.y,    co*dk.z*dk.z+cd); }

        // Per-atom total Hessian H_E_i = H_M_i + H_C_i
        float3 Hi0 = HMi0 + HC0;
        float3 Hi1 = HMi1 + HC1;
        float3 Hi2 = HMi2 + HC2;

        // --- VJP partials (factored form with w = u_world - lambda) ---
        // dL/d(a_i.xyz) = H_E_i * w + (F_M + F_C) * u.w
        float3 dL_dxyz = (float3)(dot(Hi0,w), dot(Hi1,w), dot(Hi2,w)) + (F_M + F_C) * uw;
        // dL/dR0 = w·dF_dR0 + u.w*dE_dR0
        float dL_dR0 = dot(w, dF_dR0) + uw * dE_dR0;
        // dL/dE0 = w·dF_dE0 + u.w*dE_dE0
        float dL_dE0 = dot(w, dF_dE0) + uw * dE_dE0;
        // dL/dQ  = w·dF_dQ  + u.w*dE_dQ
        float dL_dQ  = dot(w, dF_dQ)  + uw * dE_dQ;

        sum_xR += (float4)(dL_dxyz, dL_dR0);
        sum_EQ += (float4)(dL_dE0, dL_dQ, 0.0f, 0.0f);
    }

    // Atom-major layout: partial_xR[iAtom * nScan + iScan] (coalesced for pass 3)
    partial_xR[iAtom * nScan + iScan] = sum_xR;
    partial_EQ[iAtom * nScan + iScan] = sum_EQ;
}

// ---- Pass 3: reduce scan dimension (one workgroup per atom) ----
// Reduces partial_xR/partial_EQ over nScan → grad_xR/grad_EQ (one float4 per atom).
__kernel void reduceMorseDirectParamPartials(
    const int nAtoms, const int nScan, __global const float4* partial_xR,
    __global const float4* partial_EQ, __global float4* grad_xR,
    __global float4* grad_EQ, __local float4* L_xR, __local float4* L_EQ
){
    const int iAtom = get_group_id(0);
    const int iL    = get_local_id(0);
    const int nL    = get_local_size(0);
    if(iAtom >= nAtoms) return;

    // Strided accumulation over scan dimension (atom-major: contiguous for coalescing)
    float4 sxR = float4Zero, sEQ = float4Zero;
    for(int s=iL; s<nScan; s+=nL){
        sxR += partial_xR[iAtom * nScan + s];
        sEQ += partial_EQ[iAtom * nScan + s];
    }
    L_xR[iL] = sxR;
    L_EQ[iL] = sEQ;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Tree reduction
    for(int stride = nL >> 1; stride > 0; stride >>= 1){
        if(iL < stride){
            L_xR[iL] += L_xR[iL + stride];
            L_EQ[iL] += L_EQ[iL + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(iL == 0){
        grad_xR[iAtom] = L_xR[0];
        grad_EQ[iAtom] = L_EQ[0];
    }
}
