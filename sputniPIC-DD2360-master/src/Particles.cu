#include "Particles.h"
#include "Alloc.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include "Timing.h"
#include <cuda_fp16.h>
#define TPB 128

__global__ void mover_kernel(int n_sub_cycles, int NiterMover, long nop, half qom, struct grid grd, struct parameters param, 
                struct d_particles d_parts, struct d_EMfield d_fld, struct d_grid d_grd) {

    printf("Test\n");
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= nop / 2) {/*printf("return condition in thread: %d \n", i);*/ return;}

    // auxiliary variables
    /*
    FPpart dt_sub_cycling = (FPpart) param->dt/((double) part->n_sub_cycles);
    FPpart dto2 = .5*dt_sub_cycling, qomdt2 = part->qom*dto2/param->c;*/
    
    /*half2 dt_sub_cycling = __half2half2((half) (param.dt/((double) n_sub_cycles)));
    */
    half2 dt_sub_cycling = __half2half2((half) (param.dt/((double) n_sub_cycles)));
    half2 dto2 = __half2half2(__hmul_rn((half)0.5, dt_sub_cycling.x));
    
    half2 qomdt2 = __half2half2(__hdiv(__hmul_rn(qom, dto2.x), (half)param.c));
    
    half2 omdtsq, denom, ut, vt, wt, udotb;
    
    half2 Exl, Eyl, Ezl, Bxl, Byl, Bzl; 
    // local (to the particle) electric and magnetic field
    Exl.x=(half)0.0, Eyl.x=(half)0.0, Ezl.x=(half)0.0, Bxl.x=(half)0.0, Byl.x=(half)0.0, Bzl.x=(half)0.0;
    Exl.y=(half)0.0, Eyl.y=(half)0.0, Ezl.y=(half)0.0, Bxl.y=(half)0.0, Byl.y=(half)0.0, Bzl.y=(half)0.0;
    
    // interpolation densities
    half2 ix,iy,iz;
    half2 weight[2][2][2];
    half2 xi[2], eta[2], zeta[2];
    
    // intermediate particle position and velocity
    half2 xptilde, yptilde, zptilde, uptilde, vptilde, wptilde;

    if(i == 1) { printf("Before first cycle:\n x: %f, y: %f, z: %f, u: %f, v: %f, w: %f\n", __half2float(d_parts.x[i].x), __half2float(d_parts.y[i].x), __half2float(d_parts.z[i].x), __half2float(d_parts.u[i].x), __half2float(d_parts.v[i].x), __half2float(d_parts.w[i].x));}
    
    // calculate the average velocity iteratively
    for(int i_sub = 0; i_sub < n_sub_cycles; i_sub++) {
        xptilde = d_parts.x[i]; 
        yptilde = d_parts.y[i];
        zptilde = d_parts.z[i];
        for(int innter=0; innter < NiterMover; innter++){
            // interpolation G-->P
            /*ix = 2 +  int((d_parts.x[i] - (half)grd.xStart)*(half)grd.invdx);
            iy = 2 +  int((d_parts.y[i] - (half)grd.yStart)*(half)grd.invdy);
            iz = 2 +  int((d_parts.z[i] - (half)grd.zStart)*(half)grd.invdz);*/
            ix = __hadd2(__half2half2((half)2), (__hmul2_rn((__hsub2(d_parts.x[i], d_grd.xStart)), d_grd.invdx)));
            iy = __hadd2(__half2half2((half)2), (__hmul2_rn((__hsub2(d_parts.y[i], d_grd.yStart)), d_grd.invdy)));
            iz = __hadd2(__half2half2((half)2), (__hmul2_rn((__hsub2(d_parts.z[i], d_grd.zStart)), d_grd.invdz)));

            //if(i == 1) { printf("AFTER ix iy iz\n");}

            
            // calculate weights
            /*xi[0]   = d_parts.x[i] - d_grd.XN_flat[get_idx(ix-1, iy, iz, grd.nyn, grd.nzn)];
            eta[0]  = d_parts.y[i] - d_grd.YN_flat[get_idx(ix, iy-1, iz, grd.nyn, grd.nzn)];
            zeta[0] = d_parts.z[i] - d_grd.ZN_flat[get_idx(ix, iy, iz-1, grd.nyn, grd.nzn)];*/

            xi[0].x = __hsub_rn(d_parts.x[i].x, d_grd.XN_flat[get_idx((float)ix.x-1, (float)iy.x, (float)iz.x, grd.nyn, grd.nzn)]);
            xi[0].y = __hsub_rn(d_parts.x[i].y, d_grd.XN_flat[get_idx((float)ix.y-1, (float)iy.y, (float)iz.y, grd.nyn, grd.nzn)]);
            eta[0].x = __hsub_rn(d_parts.y[i].x, d_grd.YN_flat[get_idx((float)ix.x, (float)iy.x-1, (float)iz.x, grd.nyn, grd.nzn)]);
            eta[0].y = __hsub_rn(d_parts.y[i].y, d_grd.YN_flat[get_idx((float)ix.y, (float)iy.y-1, (float)iz.y, grd.nyn, grd.nzn)]);
            zeta[0].x = __hsub_rn(d_parts.z[i].x, d_grd.ZN_flat[get_idx((float)ix.x, (float)iy.x, (float)iz.x-1, grd.nyn, grd.nzn)]);
            zeta[0].y = __hsub_rn(d_parts.z[i].y, d_grd.ZN_flat[get_idx((float)ix.y, (float)iy.y, (float)iz.y-1, grd.nyn, grd.nzn)]);
 
            //if(i == 1) { printf("AFTER xi eta zeta\n");}
            /*xi[1]   = d_grd.XN_flat[get_idx(ix, iy, iz, grd.nyn, grd.nzn)] - d_parts.x[i];
            eta[1]  = d_grd.YN_flat[get_idx(ix, iy, iz, grd.nyn, grd.nzn)] - d_parts.y[i];
            zeta[1] = d_grd.ZN_flat[get_idx(ix, iy, iz, grd.nyn, grd.nzn)] - d_parts.z[i];*/

            xi[1].x = __hsub_rn(d_grd.XN_flat[get_idx((float)ix.x, (float)iy.x, (float)iz.x, grd.nyn, grd.nzn)], d_parts.x[i].x);
            xi[1].y = __hsub_rn(d_grd.XN_flat[get_idx((float)ix.y, (float)iy.y, (float)iz.y, grd.nyn, grd.nzn)], d_parts.x[i].y);
            eta[1].x = __hsub_rn(d_grd.YN_flat[get_idx((float)ix.x, (float)iy.x, (float)iz.x, grd.nyn, grd.nzn)], d_parts.y[i].x);
            eta[1].y = __hsub_rn(d_grd.YN_flat[get_idx((float)ix.y, (float)iy.y, (float)iz.y, grd.nyn, grd.nzn)], d_parts.y[i].y);
            zeta[1].x = __hsub_rn(d_grd.ZN_flat[get_idx((float)ix.x, (float)iy.x, (float)iz.x, grd.nyn, grd.nzn)], d_parts.z[i].x);
            zeta[1].y = __hsub_rn(d_grd.ZN_flat[get_idx((float)ix.y, (float)iy.y, (float)iz.y, grd.nyn, grd.nzn)], d_parts.z[i].y);

            //if(i == 1) { printf("AFTER xi eta zeta 2\n");}
            /*
            for (int ii = 0; ii < 2; ii++)
                for (int jj = 0; jj < 2; jj++)
                    for (int kk = 0; kk < 2; kk++)
                        weight[ii][jj][kk] = xi[ii] * eta[jj] * zeta[kk] * (half)grd.invVOL;
            Same loop as above but using half2s.
            */
            for (int ii = 0; ii < 2; ii++){
                for (int jj = 0; jj < 2; jj++) {
                    for (int kk = 0; kk < 2; kk++) {
                        weight[ii][jj][kk] = __hmul2_rn(__hmul2_rn(xi[ii], eta[jj]) , __hmul2_rn(zeta[kk], d_grd.invVOL));
                    }
                }
            }
            //if(i == 1) { printf("AFTER weight loop\n");}
            // set to zero local electric and magnetic field
            Exl.x=(half)0.0, Eyl.x=(half)0.0, Ezl.x=(half)0.0, Bxl.x=(half)0.0, Byl.x=(half)0.0, Bzl.x=(half)0.0;
            Exl.y=(half)0.0, Eyl.y=(half)0.0, Ezl.y=(half)0.0, Bxl.y=(half)0.0, Byl.y=(half)0.0, Bzl.y=(half)0.0;
            
            //if(i == 1) { printf("AFTER exl\n");}
            
            for (int ii=0; ii < 2; ii++)
                for (int jj=0; jj < 2; jj++)
                    for(int kk=0; kk < 2; kk++){ //SEG FAULT HERE `PERHAPS MAYBE?'TUNE IN NEXT TIME AND FIND OUT
                        Exl = __hadd2(Exl, __hmul2_rn(weight[ii][jj][kk], 
                            __halves2half2(d_fld.Ex_flat[get_idx((float)ix.x-ii, (float)iy.x-jj, (float)iz.x-kk, grd.nyn, grd.nzn)], 
                                           d_fld.Ex_flat[get_idx((float)ix.y-ii, (float)iy.y-jj, (float)iz.y-kk, grd.nyn, grd.nzn)])));
                        Eyl = __hadd2(Eyl, __hmul2_rn(weight[ii][jj][kk],
                            __halves2half2(d_fld.Ey_flat[get_idx((float)ix.x-ii, (float)iy.x-jj, (float)iz.x-kk, grd.nyn, grd.nzn)],
                                           d_fld.Ey_flat[get_idx((float)ix.y-ii, (float)iy.y-jj, (float)iz.y-kk, grd.nyn, grd.nzn)])));
                        Ezl = __hadd2(Ezl, __hmul2_rn(weight[ii][jj][kk],
                            __halves2half2(d_fld.Ez_flat[get_idx((float)ix.x-ii, (float)iy.x-jj, (float)iz.x-kk, grd.nyn, grd.nzn)],
                                           d_fld.Ez_flat[get_idx((float)ix.y-ii, (float)iy.y-jj, (float)iz.y-kk, grd.nyn, grd.nzn)])));
                        Bxl = __hadd2(Bxl, __hmul2_rn(weight[ii][jj][kk],
                            __halves2half2(d_fld.Bxn_flat[get_idx((float)ix.x-ii, (float)iy.x-jj, (float)iz.x-kk, grd.nyn, grd.nzn)],
                                           d_fld.Bxn_flat[get_idx((float)ix.y-ii, (float)iy.y-jj, (float)iz.y-kk, grd.nyn, grd.nzn)])));
                        Byl = __hadd2(Byl, __hmul2_rn(weight[ii][jj][kk],
                            __halves2half2(d_fld.Byn_flat[get_idx((float)ix.x-ii, (float)iy.x-jj, (float)iz.x-kk, grd.nyn, grd.nzn)],
                                           d_fld.Byn_flat[get_idx((float)ix.y-ii, (float)iy.y-jj, (float)iz.y-kk, grd.nyn, grd.nzn)])));
                        Bzl = __hadd2(Bzl, __hmul2_rn(weight[ii][jj][kk],
                            __halves2half2(d_fld.Bzn_flat[get_idx((float)ix.x-ii, (float)iy.x-jj, (float)iz.x-kk, grd.nyn, grd.nzn)],
                                           d_fld.Bzn_flat[get_idx((float)ix.y-ii, (float)iy.y-jj, (float)iz.y-kk, grd.nyn, grd.nzn)])));
                                           /*    
                        Exl += weight[ii][jj][kk]*d_fld.Ex_flat[get_idx(ix-ii, iy-jj, iz-kk, grd.nyn, grd.nzn)];
                        Eyl += weight[ii][jj][kk]*d_fld.Ey_flat[get_idx(ix-ii, iy-jj, iz-kk, grd.nyn, grd.nzn)];
                        Ezl += weight[ii][jj][kk]*d_fld.Ez_flat[get_idx(ix-ii, iy-jj, iz-kk, grd.nyn, grd.nzn)];
                        Bxl += weight[ii][jj][kk]*d_fld.Bxn_flat[get_idx(ix-ii, iy-jj, iz-kk, grd.nyn, grd.nzn)];
                        Byl += weight[ii][jj][kk]*d_fld.Byn_flat[get_idx(ix-ii, iy-jj, iz-kk, grd.nyn, grd.nzn)];
                        Bzl += weight[ii][jj][kk]*d_fld.Bzn_flat[get_idx(ix-ii, iy-jj, iz-kk, grd.nyn, grd.nzn)];*/
                    }
            //if(i == 1) { printf("AFTER big ass loop\n");}
                        
            // end interpolation
            /*omdtsq = qomdt2*qomdt2*(Bxl*Bxl+Byl*Byl+Bzl*Bzl);
            denom = (half)1.0/((half)1.0 + omdtsq);*/
            omdtsq = __hmul2_rn(qomdt2, __hmul2_rn(qomdt2, __hadd2(__hmul2_rn(Bxl, Bxl), __hadd2(__hmul2_rn(Byl, Byl), __hmul2_rn(Bzl, Bzl)))));
            denom = __h2div(__half2half2(1.0), __hadd2(__half2half2(1.0), omdtsq));

            // solve the position equation
            /*ut= d_parts.u[i] + qomdt2*Exl;
            vt= d_parts.v[i] + qomdt2*Eyl;
            wt= d_parts.w[i] + qomdt2*Ezl;
            udotb = ut*Bxl + vt*Byl + wt*Bzl;*/
            ut= __hadd2(d_parts.u[i], __hmul2_rn(qomdt2, Exl));
            vt= __hadd2(d_parts.v[i], __hmul2_rn(qomdt2, Eyl));
            wt= __hadd2(d_parts.w[i], __hmul2_rn(qomdt2, Ezl));
            udotb = __hadd2(__hadd2(__hmul2_rn(ut, Bxl), __hmul2_rn(vt, Byl)), __hmul2_rn(wt, Bzl));

            //if(i == 1) { printf("AFTER position equation\n");}
            
            // solve the velocity equation
            /*uptilde = (ut+qomdt2*(vt*Bzl -wt*Byl + qomdt2*udotb*Bxl))*denom;
            vptilde = (vt+qomdt2*(wt*Bxl -ut*Bzl + qomdt2*udotb*Byl))*denom;
            wptilde = (wt+qomdt2*(ut*Byl -vt*Bxl + qomdt2*udotb*Bzl))*denom;*/
            uptilde = __hmul2_rn(__hadd2(ut, __hmul2_rn(qomdt2, __hadd2(__hmul2_rn(vt, Bzl), __hadd2(__hmul2_rn(__hneg2(wt), Byl), __hmul2_rn(qomdt2, __hmul2_rn(udotb, Bxl)))))), denom);
            vptilde = __hmul2_rn(__hadd2(vt, __hmul2_rn(qomdt2, __hadd2(__hmul2_rn(wt, Bxl), __hadd2(__hmul2_rn(__hneg2(ut), Bzl), __hmul2_rn(qomdt2, __hmul2_rn(udotb, Byl)))))), denom);
            wptilde = __hmul2_rn(__hadd2(wt, __hmul2_rn(qomdt2, __hadd2(__hmul2_rn(ut, Byl), __hadd2(__hmul2_rn(__hneg2(vt), Bxl), __hmul2_rn(qomdt2, __hmul2_rn(udotb, Bzl)))))), denom);
            
            //if(i == 1) { printf("AFTER velocity equation\n");}

            // update position
            /*d_parts.x[i] = xptilde + uptilde*dto2;
            d_parts.y[i] = yptilde + vptilde*dto2;
            d_parts.z[i] = zptilde + wptilde*dto2;
            */
            d_parts.x[i] = __hadd2(xptilde, __hmul2_rn(uptilde, dto2));
            d_parts.y[i] = __hadd2(yptilde, __hmul2_rn(vptilde, dto2));
            d_parts.z[i] = __hadd2(zptilde, __hmul2_rn(wptilde, dto2));

            //if(i == 1) { printf("AFTER Update position\n");}
            
        } // end of iteration
        // update the final position and velocity
        /*
        d_parts.u[i]= (half)2.0*uptilde - d_parts.u[i];
        d_parts.v[i]= (half)2.0*vptilde - d_parts.v[i];
        d_parts.w[i]= (half)2.0*wptilde - d_parts.w[i];
        d_parts.x[i] = xptilde + uptilde*dt_sub_cycling;
        d_parts.y[i] = yptilde + vptilde*dt_sub_cycling;
        d_parts.z[i] = zptilde + wptilde*dt_sub_cycling;*/
        d_parts.u[i]= __hsub2(__hmul2_rn(__half2half2(2.0), uptilde), d_parts.u[i]);
        d_parts.v[i]= __hsub2(__hmul2_rn(__half2half2(2.0), vptilde), d_parts.v[i]);
        d_parts.w[i]= __hsub2(__hmul2_rn(__half2half2(2.0), wptilde), d_parts.w[i]);
        d_parts.x[i] = __hadd2(xptilde, __hmul2_rn(uptilde, dt_sub_cycling));
        d_parts.y[i] = __hadd2(yptilde, __hmul2_rn(vptilde, dt_sub_cycling));
        d_parts.z[i] = __hadd2(zptilde, __hmul2_rn(wptilde, dt_sub_cycling));
        
        //if(i == 1) { printf("AFTER  final position and velocity\n");}
        //////////
        //////////
        ////////// BC
        // X-DIRECTION: BC particles
        /*if (d_parts.x[i] > (half)grd.Lx){
            if (param.PERIODICX==true){ // PERIODIC
                d_parts.x[i] = d_parts.x[i] - (half)grd.Lx;
            } else { // REFLECTING BC
                d_parts.u[i] = -d_parts.u[i];
                d_parts.x[i] = (half)2*(half)grd.Lx - d_parts.x[i];
            }
        }*/
        if (__hgt(d_parts.x[i].x, d_grd.Lx.x)){
            if (param.PERIODICX==true){ // PERIODIC
                d_parts.x[i].x = __hsub(d_parts.x[i].x, d_grd.Lx.x);
            } else { // REFLECTING BC
                d_parts.u[i].x = __hneg(d_parts.u[i].x);
                d_parts.x[i].x = __hsub(__hmul_rn((half)2, d_grd.Lx.x), d_parts.x[i].x);
            }
        }
        
        if (__hgt(d_parts.x[i].y, d_grd.Lx.y)){
            if (param.PERIODICX==true){ // PERIODIC
                d_parts.x[i].y = __hsub(d_parts.x[i].y, d_grd.Lx.y);
            } else { // REFLECTING BC
                d_parts.u[i].y = __hneg(d_parts.u[i].y);
                d_parts.x[i].y = __hsub(__hmul_rn((half)2, d_grd.Lx.y), d_parts.x[i].y);
            }
        }

        //if(i == 1) { printf("AFTER first mini loop\n");}

        /*
        if (d_parts.x[i] < (half)0){
            if (param.PERIODICX==true){ // PERIODIC
                d_parts.x[i] = d_parts.x[i] + (half)grd.Lx;
            } else { // REFLECTING BC
                d_parts.u[i] = -d_parts.u[i];
                d_parts.x[i] = -d_parts.x[i];
            }
        }*/
        if (__hlt(d_parts.x[i].x, (half)0)){
            if (param.PERIODICX==true){ // PERIODIC
                if(i == 1) printf("(param.PERIODICX==true): x: %f, Lx: %f\n",(float)d_parts.x[i].x, (float)d_grd.Lx.x);
                d_parts.x[i].x = __hadd(d_parts.x[i].x, d_grd.Lx.x);
                if(i == 1) printf("(AFTER:::param.PERIODICX==true): x: %f, Lx: %f\n",(float)d_parts.x[i].x, (float)d_grd.Lx.x);
                
            } else { // REFLECTING BC
                if(i == 1) printf("(param.PERIODICX==false): x: %f\n", d_parts.x[i].x);
                d_parts.u[i].x = __hneg(d_parts.u[i].x);
                d_parts.x[i].x = __hneg(d_parts.x[i].x);
                if(i == 1) printf("(AFTER:::param.PERIODICX==false): x: %f\n",(float)d_parts.x[i].x);
            }
        }

        if (__hlt(d_parts.x[i].y, (half)0)){
            if (param.PERIODICX==true){ // PERIODIC
                d_parts.x[i].y = __hadd(d_parts.x[i].y, d_grd.Lx.y);
            } else { // REFLECTING BC
                d_parts.u[i].y = __hneg(d_parts.u[i].y);
                d_parts.x[i].y = __hneg(d_parts.x[i].y);
            }
        }

        //if(i == 1) { printf("AFTER second mini loop\n");}
            
        // Y-DIRECTION: BC particles
        /*if (d_parts.y[i] > (half)grd.Ly){
            if (param.PERIODICY==true){ // PERIODIC
                d_parts.y[i] = d_parts.y[i] - (half)grd.Ly;
            } else { // REFLECTING BC
                d_parts.v[i] = -d_parts.v[i];
                d_parts.y[i] = (half)2*(half)grd.Ly - d_parts.y[i];
            }
        }*/
        if (__hgt(d_parts.y[i].x, d_grd.Ly.x)){
            if (param.PERIODICY==true){ // PERIODIC
                d_parts.y[i].x = __hsub(d_parts.y[i].x, d_grd.Ly.x);
            } else { // REFLECTING BC
                d_parts.v[i].x = __hneg(d_parts.v[i].x);
                d_parts.y[i].x = __hsub(__hmul_rn((half)2, d_grd.Ly.x), d_parts.y[i].x);
            }
        }

        if (__hgt(d_parts.y[i].y, d_grd.Ly.y)){
            if (param.PERIODICY==true){ // PERIODIC
                d_parts.y[i].y = __hsub(d_parts.y[i].y, d_grd.Ly.y);
            } else { // REFLECTING BC
                d_parts.v[i].y = __hneg(d_parts.v[i].y);
                d_parts.y[i].y = __hsub(__hmul_rn((half)2, d_grd.Ly.y), d_parts.y[i].y);
            }
        }

        //if(i == 1) { printf("AFTER third mini loop\n");}

                                                                    
        /*if (d_parts.y[i] < (half) 0){
            if (param.PERIODICY==true){ // PERIODIC
                d_parts.y[i] = d_parts.y[i] + (half)grd.Ly;
            } else { // REFLECTING BC
                d_parts.v[i] = -d_parts.v[i];
                d_parts.y[i] = -d_parts.y[i];
            }
        }*/
        if (__hlt(d_parts.y[i].x, (half)0)){
            if (param.PERIODICY==true){ // PERIODIC
                d_parts.y[i].x = __hadd(d_parts.y[i].x, d_grd.Ly.x);
            } else { // REFLECTING BC
                d_parts.v[i].x = __hneg(d_parts.v[i].x);
                d_parts.y[i].x = __hneg(d_parts.y[i].x);
            }
        }

        if (__hlt(d_parts.y[i].y, (half)0)){
            if (param.PERIODICY==true){ // PERIODIC
                d_parts.y[i].y = __hadd(d_parts.y[i].y, d_grd.Ly.y);
            } else { // REFLECTING BC
                d_parts.v[i].y = __hneg(d_parts.v[i].y);
                d_parts.y[i].y = __hneg(d_parts.y[i].y);
            }
        }

        //if(i == 1) { printf("AFTER forth mini loop\n");}
                                                                    
        // Z-DIRECTION: BC particles
        /*if (d_parts.z[i] > (half)grd.Lz){
            if (param.PERIODICZ==true){ // PERIODIC
                d_parts.z[i] = d_parts.z[i] - (half)grd.Lz;
            } else { // REFLECTING BC
                d_parts.w[i] = -d_parts.w[i];
                d_parts.z[i] = (half)2*(half)grd.Lz - d_parts.z[i];
            }
        }*/
        
        if (__hgt(d_parts.z[i].x, d_grd.Lz.x)){
            if (param.PERIODICZ==true){ // PERIODIC
                d_parts.z[i].x = __hsub(d_parts.z[i].x, d_grd.Lz.x);
            } else { // REFLECTING BC
                d_parts.w[i].x = __hneg(d_parts.w[i].x);
                d_parts.z[i].x = __hsub(__hmul_rn((half)2, d_grd.Lz.x), d_parts.z[i].x);
            }
        }

        if (__hgt(d_parts.z[i].y, d_grd.Lz.y)){
            if (param.PERIODICZ==true){ // PERIODIC
                d_parts.z[i].y = __hsub(d_parts.z[i].y, d_grd.Lz.y);
            } else { // REFLECTING BC
                d_parts.w[i].y = __hneg(d_parts.w[i].y);
                d_parts.z[i].y = __hsub(__hmul_rn((half)2, d_grd.Lz.y), d_parts.z[i].y);
            }
        }

        //if(i == 1) { printf("AFTER fifth mini loop\n");}
            
                                                                    
        /*if (d_parts.z[i] < (half)0){
            if (param.PERIODICZ==true){ // PERIODIC
                d_parts.z[i] = d_parts.z[i] + (half)grd.Lz;
            } else { // REFLECTING BC
                d_parts.w[i] = -d_parts.w[i];
                d_parts.z[i] = -d_parts.z[i];
            }
        }*/

        if (__hlt(d_parts.z[i].x, (half)0)){
            if (param.PERIODICZ==true){ // PERIODIC
                d_parts.z[i].x = __hadd(d_parts.z[i].x, d_grd.Lz.x);
            } else { // REFLECTING BC
                d_parts.w[i].x = __hneg(d_parts.w[i].x);
                d_parts.z[i].x = __hneg(d_parts.z[i].x);
            }
        }

        if (__hlt(d_parts.z[i].y, (half)0)){
            if (param.PERIODICZ==true){ // PERIODIC
                d_parts.z[i].y = __hadd(d_parts.z[i].y, d_grd.Lz.y);
            } else { // REFLECTING BC
                d_parts.w[i].y = __hneg(d_parts.w[i].y);
                d_parts.z[i].y = __hneg(d_parts.z[i].y);
            }
        }

        //if(i == 1) { printf("AFTER last mini loop\n");}
    }
    if(i == 1) {
        printf("End of mover:\n x: %f, y: %f, z: %f, u: %f, v: %f, w: %f\n",
            __half2float(d_parts.x[i].x), __half2float(d_parts.y[i].x), __half2float(d_parts.z[i].x),
            __half2float(d_parts.u[i].x), __half2float(d_parts.v[i].x), __half2float(d_parts.w[i].x));
    }
    //if(i > 2040990) printf("%d\n", i);
    //seg fault at 2040991 + 1
    //2047999 + 1
}

void convert_to_fp16(struct particles* parts, struct d_particles *d_parts, int num,
                    struct grid* grd, struct d_grid* d_grd,
                    struct EMfield* field, struct d_EMfield* d_fld, half2** temp_parts) {
    for(int p = 0; p < num; p++) {
        struct particles part = parts[p];
        struct d_particles d_part = d_parts[p];
        
        half2 * d_prt[6];
        for(int i = 0; i < 6; i++) {
            cudaMalloc(&d_prt[i], part.npmax/2 * sizeof(half2) + 1);
        }

        d_part.x = d_prt[0];
        d_part.y = d_prt[1];
        d_part.z = d_prt[2];
        d_part.u = d_prt[3];
        d_part.v = d_prt[4];
        d_part.w = d_prt[5];


        for(int i = 0; i < part.npmax; i += 2) {
            temp_parts[0][i/2].x = __float2half(part.x[i]);
            temp_parts[0][i/2].y = __float2half(part.x[i + 1]);
            temp_parts[1][i/2].x = __float2half(part.y[i]);
            temp_parts[1][i/2].y = __float2half(part.y[i + 1]);
            temp_parts[2][i/2].x = __float2half(part.z[i]);
            temp_parts[2][i/2].y = __float2half(part.z[i + 1]);
            temp_parts[3][i/2].x = __float2half(part.u[i]);
            temp_parts[3][i/2].y = __float2half(part.u[i + 1]);
            temp_parts[4][i/2].x = __float2half(part.v[i]);
            temp_parts[4][i/2].y = __float2half(part.v[i + 1]);
            temp_parts[5][i/2].x = __float2half(part.w[i]);
            temp_parts[5][i/2].y = __float2half(part.w[i + 1]);
        }

        cudaMemcpy((d_part.x), temp_parts[0], part.npmax /2 *sizeof(half2) + 1, cudaMemcpyHostToDevice);
        cudaMemcpy((d_part.y), temp_parts[1], part.npmax /2 *sizeof(half2) + 1, cudaMemcpyHostToDevice);
        cudaMemcpy((d_part.z), temp_parts[2], part.npmax /2 *sizeof(half2) + 1, cudaMemcpyHostToDevice);
        cudaMemcpy((d_part.u), temp_parts[3], part.npmax /2 *sizeof(half2) + 1, cudaMemcpyHostToDevice);
        cudaMemcpy((d_part.v), temp_parts[4], part.npmax /2 *sizeof(half2) + 1, cudaMemcpyHostToDevice);
        cudaMemcpy((d_part.w), temp_parts[5], part.npmax /2 *sizeof(half2) + 1, cudaMemcpyHostToDevice);
    }
    // Grid GPU Allocation
    FPfield * d_cnodes[3];
    for (int i = 0; i < 3; i++) {
        cudaMalloc(&d_cnodes[i], grd->nxn*grd->nyn*grd->nzn*sizeof(FPfield));
    }
    d_grd->XN_flat = d_cnodes[0];
    d_grd->YN_flat = d_cnodes[1];
    d_grd->ZN_flat = d_cnodes[2];

    cudaMemcpy((d_grd->XN_flat), grd->XN_flat, grd->nxn*grd->nyn*grd->nzn*sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy((d_grd->YN_flat), grd->YN_flat, grd->nxn*grd->nyn*grd->nzn*sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy((d_grd->ZN_flat), grd->ZN_flat, grd->nxn*grd->nyn*grd->nzn*sizeof(FPfield), cudaMemcpyHostToDevice);
    
    d_grd->xStart.x = __double2half(grd->xStart);
    d_grd->xStart.y = __double2half(grd->xStart);
    d_grd->xEnd.x = __double2half(grd->xEnd);
    d_grd->xEnd.y = __double2half(grd->xEnd);
    d_grd->yStart.x = __double2half(grd->yStart);
    d_grd->yStart.y = __double2half(grd->yStart);
    d_grd->yEnd.x = __double2half(grd->yEnd);
    d_grd->yEnd.y = __double2half(grd->yEnd);
    d_grd->zStart.x = __double2half(grd->zStart);
    d_grd->zStart.y = __double2half(grd->zStart);
    d_grd->zEnd.x = __double2half(grd->zEnd);
    d_grd->zEnd.y = __double2half(grd->zEnd);
    d_grd->Lx.x = __double2half(grd->Lx);
    d_grd->Lx.y = __double2half(grd->Lx);
    d_grd->Ly.x = __double2half(grd->Ly);
    d_grd->Ly.y = __double2half(grd->Ly);
    d_grd->Lz.x = __double2half(grd->Lz);
    d_grd->Lz.y = __double2half(grd->Lz);
    d_grd->invdx.x = __double2half(grd->invdx);
    d_grd->invdx.y = __double2half(grd->invdx);
    d_grd->invdy.x = __double2half(grd->invdy);
    d_grd->invdy.y = __double2half(grd->invdy);
    d_grd->invdz.x = __double2half(grd->invdz);
    d_grd->invdz.y = __double2half(grd->invdz);
    d_grd->invVOL.x = __double2half(grd->invVOL);
    d_grd->invVOL.y = __double2half(grd->invVOL);
    
    // Field GPU Allocation
    FPfield * d_enodes[6];
    for(int i = 0; i < 6; i++) {
        cudaMalloc(&d_enodes[i], grd->nxn*grd->nyn*grd->nzn*sizeof(FPfield));
    }

    d_fld->Ex_flat = d_enodes[0];
    d_fld->Ey_flat = d_enodes[1];
    d_fld->Ez_flat = d_enodes[2];
    d_fld->Bxn_flat = d_enodes[3];
    d_fld->Byn_flat = d_enodes[4];
    d_fld->Bzn_flat = d_enodes[5];

    cudaMemcpy((d_fld->Ex_flat), field->Ex_flat, grd->nxn*grd->nyn*grd->nzn*sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy((d_fld->Ey_flat), field->Ey_flat, grd->nxn*grd->nyn*grd->nzn*sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy((d_fld->Ez_flat), field->Ez_flat, grd->nxn*grd->nyn*grd->nzn*sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy((d_fld->Bxn_flat), field->Bxn_flat, grd->nxn*grd->nyn*grd->nzn*sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy((d_fld->Byn_flat), field->Byn_flat, grd->nxn*grd->nyn*grd->nzn*sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy((d_fld->Bzn_flat), field->Bzn_flat, grd->nxn*grd->nyn*grd->nzn*sizeof(FPfield), cudaMemcpyHostToDevice);
}

void convert_to_float(struct particles* parts, struct d_particles *d_parts, int num,
                    struct grid* grd, struct d_grid* d_grd,
                    struct EMfield* field, struct d_EMfield* d_fld, half2** temp_parts) {
    for(int p = 0; p < num; p++) {
        struct particles* part = &parts[p];
        struct d_particles d_part = d_parts[p];
        cudaMemcpy(temp_parts[0], d_part.x, part->npmax/2 * sizeof(half2) + 1, cudaMemcpyDeviceToHost);
        cudaMemcpy(temp_parts[1], d_part.y, part->npmax/2 * sizeof(half2) + 1, cudaMemcpyDeviceToHost);
        cudaMemcpy(temp_parts[2], d_part.z, part->npmax/2 * sizeof(half2) + 1, cudaMemcpyDeviceToHost);
        cudaMemcpy(temp_parts[3], d_part.u, part->npmax/2 * sizeof(half2) + 1, cudaMemcpyDeviceToHost);
        cudaMemcpy(temp_parts[4], d_part.v, part->npmax/2 * sizeof(half2) + 1, cudaMemcpyDeviceToHost);
        cudaMemcpy(temp_parts[5], d_part.w, part->npmax/2 * sizeof(half2) + 1, cudaMemcpyDeviceToHost);


        for(int i = 0; i < part->nop; i+=2) {
            part->x[i] = (FPpart)__half2float(temp_parts[0][i/2].x);
            part->x[i + 1] = (FPpart)__half2float(temp_parts[0][i/2].y);
            part->y[i] = (FPpart)__half2float(temp_parts[1][i/2].x);
            part->y[i + 1] = (FPpart)__half2float(temp_parts[1][i/2].y);
            part->z[i] = (FPpart)__half2float(temp_parts[2][i/2].x);
            part->z[i + 1] = (FPpart)__half2float(temp_parts[2][i/2].y);
            part->u[i] = (FPpart)__half2float(temp_parts[3][i/2].x);
            part->u[i + 1] = (FPpart)__half2float(temp_parts[3][i/2].y);
            part->v[i] = (FPpart)__half2float(temp_parts[4][i/2].x);
            part->v[i + 1] = (FPpart)__half2float(temp_parts[4][i/2].y);
            part->w[i] = (FPpart)__half2float(temp_parts[5][i/2].x);
            part->w[i + 1] = (FPpart)__half2float(temp_parts[5][i/2].y);
        }

        cudaFree(d_part.x);
        cudaFree(d_part.y);
        cudaFree(d_part.z);
        cudaFree(d_part.u);
        cudaFree(d_part.v);
        cudaFree(d_part.w);
    }

    cudaFree(d_grd->XN_flat);
    cudaFree(d_grd->YN_flat);
    cudaFree(d_grd->ZN_flat);
    
    cudaFree(d_fld->Ex_flat);
    cudaFree(d_fld->Ey_flat);
    cudaFree(d_fld->Ez_flat);
    cudaFree(d_fld->Bxn_flat);
    cudaFree(d_fld->Byn_flat);
    cudaFree(d_fld->Bzn_flat);
}

int mover_PC_gpu(struct particles* part, struct d_particles d_part, struct d_EMfield d_fld, struct grid* grd, struct d_grid d_grd, struct parameters* param) {
    // print species and subcycling
    std::cout << "***  MOVER with SUBCYCLYING "<< param->n_sub_cycles << " - species " << part->species_ID << " ***" << std::endl;

    // Particle GPU allocation

    
    double startTime = cpuSecond();
    int Db = TPB;
    int Dg = (part->nop / 2 + TPB - 1) / TPB;
    std::cout << "Block: " << Db << " Group: " << Dg << "\n" << std::endl;
    mover_kernel2<<<Dg, Db>>>(part->n_sub_cycles, part->NiterMover, part->nop, __float2half(part->qom), *grd, *param, d_part, d_fld, d_grd);
    double endTime = cpuSecond() -startTime;
    //I FEAR NO MAN, BUT THAT THING, SEGMENTATION FUALT(core dumped), IT SCARES ME. 
    std::cout << "End time: " << endTime << "\n\n" << std::endl;
    cudaDeviceSynchronize();

    //std::cout << "x: " << part->x[0] << std::endl;

    return(0); // exit succcesfully
}

/** allocate particle arrays */
void particle_allocate(struct parameters* param, struct particles* part, int is)
{
    
    // set species ID
    part->species_ID = is;
    // number of particles
    part->nop = param->np[is];
    // maximum number of particles
    part->npmax = param->npMax[is];
    
    // choose a different number of mover iterations for ions and electrons
    if (param->qom[is] < 0){  //electrons
        part->NiterMover = param->NiterMover;
        part->n_sub_cycles = param->n_sub_cycles;
    } else {                  // ions: only one iteration
        part->NiterMover = 1;
        part->n_sub_cycles = 1;
    }
    
    // particles per cell
    part->npcelx = param->npcelx[is];
    part->npcely = param->npcely[is];
    part->npcelz = param->npcelz[is];
    part->npcel = part->npcelx*part->npcely*part->npcelz;
    
    // cast it to required precision
    part->qom = (FPpart) param->qom[is];
    
    long npmax = part->npmax;
    
    // initialize drift and thermal velocities
    // drift
    part->u0 = (FPpart) param->u0[is];
    part->v0 = (FPpart) param->v0[is];
    part->w0 = (FPpart) param->w0[is];
    // thermal
    part->uth = (FPpart) param->uth[is];
    part->vth = (FPpart) param->vth[is];
    part->wth = (FPpart) param->wth[is];
    
    
    //////////////////////////////
    /// ALLOCATION PARTICLE ARRAYS
    //////////////////////////////
    part->x = new FPpart[npmax];
    part->y = new FPpart[npmax];
    part->z = new FPpart[npmax];
    // allocate velocity
    part->u = new FPpart[npmax];
    part->v = new FPpart[npmax];
    part->w = new FPpart[npmax];
    // allocate charge = q * statistical weight
    part->q = new FPinterp[npmax];
    
}
/** deallocate */
void particle_deallocate(struct particles* part)
{
    // deallocate particle variables
    delete[] part->x;
    delete[] part->y;
    delete[] part->z;
    delete[] part->u;
    delete[] part->v;
    delete[] part->w;
    delete[] part->q;
}

/** particle mover */
int mover_PC(struct particles* part, struct EMfield* field, struct grid* grd, struct parameters* param)
{
    // print species and subcycling
    std::cout << "***  MOVER with SUBCYCLYING "<< param->n_sub_cycles << " - species " << part->species_ID << " ***" << std::endl;
 
    // auxiliary variables
    FPpart dt_sub_cycling = (FPpart) param->dt/((double) part->n_sub_cycles);
    FPpart dto2 = .5*dt_sub_cycling, qomdt2 = part->qom*dto2/param->c;
    FPpart omdtsq, denom, ut, vt, wt, udotb;
    
    // local (to the particle) electric and magnetic field
    FPfield Exl=0.0, Eyl=0.0, Ezl=0.0, Bxl=0.0, Byl=0.0, Bzl=0.0;
    
    // interpolation densities
    int ix,iy,iz;
    FPfield weight[2][2][2];
    FPfield xi[2], eta[2], zeta[2];
    
    // intermediate particle position and velocity
    FPpart xptilde, yptilde, zptilde, uptilde, vptilde, wptilde;
    
    // start subcycling
    for (int i_sub=0; i_sub <  part->n_sub_cycles; i_sub++){
        // move each particle with new fields
        for (int i=0; i <  part->nop; i++){
            xptilde = part->x[i];
            yptilde = part->y[i];
            zptilde = part->z[i];
            // calculate the average velocity iteratively
            for(int innter=0; innter < part->NiterMover; innter++){
                // interpolation G-->P
                ix = 2 +  int((part->x[i] - grd->xStart)*grd->invdx);
                iy = 2 +  int((part->y[i] - grd->yStart)*grd->invdy);
                iz = 2 +  int((part->z[i] - grd->zStart)*grd->invdz);
                
                // calculate weights
                xi[0]   = part->x[i] - grd->XN[ix - 1][iy][iz];
                eta[0]  = part->y[i] - grd->YN[ix][iy - 1][iz];
                zeta[0] = part->z[i] - grd->ZN[ix][iy][iz - 1];

                xi[1]   = grd->XN[ix][iy][iz] - part->x[i];
                eta[1]  = grd->YN[ix][iy][iz] - part->y[i];
                zeta[1] = grd->ZN[ix][iy][iz] - part->z[i];

                for (int ii = 0; ii < 2; ii++)
                    for (int jj = 0; jj < 2; jj++)
                        for (int kk = 0; kk < 2; kk++)
                            weight[ii][jj][kk] = xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;
                
                // set to zero local electric and magnetic field
                Exl=0.0, Eyl = 0.0, Ezl = 0.0, Bxl = 0.0, Byl = 0.0, Bzl = 0.0;
                
                for (int ii=0; ii < 2; ii++)
                    for (int jj=0; jj < 2; jj++)
                        for(int kk=0; kk < 2; kk++){
                            Exl += weight[ii][jj][kk]*field->Ex[ix- ii][iy -jj][iz- kk ];
                            Eyl += weight[ii][jj][kk]*field->Ey[ix- ii][iy -jj][iz- kk ];
                            Ezl += weight[ii][jj][kk]*field->Ez[ix- ii][iy -jj][iz -kk ];
                            Bxl += weight[ii][jj][kk]*field->Bxn[ix- ii][iy -jj][iz -kk ];
                            Byl += weight[ii][jj][kk]*field->Byn[ix- ii][iy -jj][iz -kk ];
                            Bzl += weight[ii][jj][kk]*field->Bzn[ix- ii][iy -jj][iz -kk ];
                        }
                
                // end interpolation
                omdtsq = qomdt2*qomdt2*(Bxl*Bxl+Byl*Byl+Bzl*Bzl);
                denom = 1.0/(1.0 + omdtsq);

                // solve the position equation
                ut= part->u[i] + qomdt2*Exl;
                vt= part->v[i] + qomdt2*Eyl;
                wt= part->w[i] + qomdt2*Ezl;
                udotb = ut*Bxl + vt*Byl + wt*Bzl;
                // solve the velocity equation
                uptilde = (ut+qomdt2*(vt*Bzl -wt*Byl + qomdt2*udotb*Bxl))*denom;
                vptilde = (vt+qomdt2*(wt*Bxl -ut*Bzl + qomdt2*udotb*Byl))*denom;
                wptilde = (wt+qomdt2*(ut*Byl -vt*Bxl + qomdt2*udotb*Bzl))*denom;
                // update position
                part->x[i] = xptilde + uptilde*dto2;
                part->y[i] = yptilde + vptilde*dto2;
                part->z[i] = zptilde + wptilde*dto2;
                
                
            } // end of iteration
            // update the final position and velocity
            part->u[i]= 2.0*uptilde - part->u[i];
            part->v[i]= 2.0*vptilde - part->v[i];
            part->w[i]= 2.0*wptilde - part->w[i];
            part->x[i] = xptilde + uptilde*dt_sub_cycling;
            part->y[i] = yptilde + vptilde*dt_sub_cycling;
            part->z[i] = zptilde + wptilde*dt_sub_cycling;
            
            //////////
            //////////
            ////////// BC
                                        
            // X-DIRECTION: BC particles
            if (part->x[i] > grd->Lx){
                if (param->PERIODICX==true){ // PERIODIC
                    part->x[i] = part->x[i] - grd->Lx;
                } else { // REFLECTING BC
                    part->u[i] = -part->u[i];
                    part->x[i] = 2*grd->Lx - part->x[i];
                }
            }
                                                                        
            if (part->x[i] < 0){
                if (param->PERIODICX==true){ // PERIODIC
                   part->x[i] = part->x[i] + grd->Lx;
                } else { // REFLECTING BC
                    part->u[i] = -part->u[i];
                    part->x[i] = -part->x[i];
                }
            }
                
            
            // Y-DIRECTION: BC particles
            if (part->y[i] > grd->Ly){
                if (param->PERIODICY==true){ // PERIODIC
                    part->y[i] = part->y[i] - grd->Ly;
                } else { // REFLECTING BC
                    part->v[i] = -part->v[i];
                    part->y[i] = 2*grd->Ly - part->y[i];
                }
            }
                                                                        
            if (part->y[i] < 0){
                if (param->PERIODICY==true){ // PERIODIC
                    part->y[i] = part->y[i] + grd->Ly;
                } else { // REFLECTING BC
                    part->v[i] = -part->v[i];
                    part->y[i] = -part->y[i];
                }
            }
                                                                        
            // Z-DIRECTION: BC particles
            if (part->z[i] > grd->Lz){
                if (param->PERIODICZ==true){ // PERIODIC
                    part->z[i] = part->z[i] - grd->Lz;
                } else { // REFLECTING BC
                    part->w[i] = -part->w[i];
                    part->z[i] = 2*grd->Lz - part->z[i];
                }
            }
                                                                        
            if (part->z[i] < 0){
                if (param->PERIODICZ==true){ // PERIODIC
                    part->z[i] = part->z[i] + grd->Lz;
                } else { // REFLECTING BC
                    part->w[i] = -part->w[i];
                    part->z[i] = -part->z[i];
                }
            }
                                                                        
            
            
        }  // end of subcycling
    } // end of one particle
                                                                        
    return(0); // exit succcesfully
} // end of the mover



/** Interpolation Particle --> Grid: This is for species */
void interpP2G(struct particles* part, struct interpDensSpecies* ids, struct grid* grd)
{
    
    // arrays needed for interpolation
    FPpart weight[2][2][2];
    FPpart temp[2][2][2];
    FPpart xi[2], eta[2], zeta[2];
    
    // index of the cell
    int ix, iy, iz;
    
    
    for (register long long i = 0; i < part->nop; i++) {
        
        // determine cell: can we change to int()? is it faster?
        ix = 2 + int (floor((part->x[i] - grd->xStart) * grd->invdx));
        iy = 2 + int (floor((part->y[i] - grd->yStart) * grd->invdy));
        iz = 2 + int (floor((part->z[i] - grd->zStart) * grd->invdz));
        
        // distances from node
        xi[0]   = part->x[i] - grd->XN[ix - 1][iy][iz];
        eta[0]  = part->y[i] - grd->YN[ix][iy - 1][iz];
        zeta[0] = part->z[i] - grd->ZN[ix][iy][iz - 1];
        xi[1]   = grd->XN[ix][iy][iz] - part->x[i];
        eta[1]  = grd->YN[ix][iy][iz] - part->y[i];
        zeta[1] = grd->ZN[ix][iy][iz] - part->z[i];
        
        // calculate the weights for different nodes
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    weight[ii][jj][kk] = part->q[i] * xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;
        
        //////////////////////////
        // add charge density
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->rhon[ix - ii][iy - jj][iz - kk] += weight[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add current density - Jx
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * weight[ii][jj][kk];
        
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jx[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add current density - Jy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        
        ////////////////////////////
        // add current density - Jz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add pressure pxx
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->u[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxx[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add pressure pxy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        
        /////////////////////////////
        // add pressure pxz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pyy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pyy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pyz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pyz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pzz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->w[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii=0; ii < 2; ii++)
            for (int jj=0; jj < 2; jj++)
                for(int kk=0; kk < 2; kk++)
                    ids->pzz[ix -ii][iy -jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
    
    }
   
}
