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
