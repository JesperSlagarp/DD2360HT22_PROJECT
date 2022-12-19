#include "Particles.h"
#include "Alloc.h"
#include <cuda.h>
#include <cuda_runtime.h>
#define TPB 128

__global__ void mover_kernel(int n_sub_cycles, int NiterMover, long nop, FPpart qom, struct grid grd, struct parameters param, 
                  struct d_particles d_parts, struct d_EMfield d_fld, struct d_grid d_grd) 
{
 
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i >= nop) return;

    // auxiliary variables
    FPpart dt_sub_cycling = (FPpart) param.dt/((double) n_sub_cycles);
    FPpart dto2 = .5*dt_sub_cycling, qomdt2 = qom*dto2/param.c;
    FPpart omdtsq, denom, ut, vt, wt, udotb;
    
    // local (to the particle) electric and magnetic field
    FPfield Exl=0.0, Eyl=0.0, Ezl=0.0, Bxl=0.0, Byl=0.0, Bzl=0.0;
    
    // interpolation densities
    int ix,iy,iz;
    FPfield weight[2][2][2];
    FPfield xi[2], eta[2], zeta[2];
    
    // intermediate particle position and velocity
    FPpart xptilde, yptilde, zptilde, uptilde, vptilde, wptilde;

    //if(i == 1) { printf("Before first cycle:\n x: %f, y: %f, z: %f, u: %f, v: %f, w: %f\n", d_parts.x[i], d_parts.y[i], d_parts.z[i], d_parts.u[i], d_parts.v[i], d_parts.w[i]);}
    
    // calculate the average velocity iteratively
    for(int i_sub = 0; i_sub < n_sub_cycles; i_sub++) {
        xptilde = d_parts.x[i]; 
        yptilde = d_parts.y[i];
        zptilde = d_parts.z[i];
        for(int innter=0; innter < NiterMover; innter++){
            // interpolation G-->P
            ix = 2 +  int((d_parts.x[i] - grd.xStart)*grd.invdx);
            iy = 2 +  int((d_parts.y[i] - grd.yStart)*grd.invdy);
            iz = 2 +  int((d_parts.z[i] - grd.zStart)*grd.invdz);
            
            // calculate weights
            xi[0]   = d_parts.x[i] - d_grd.XN_flat[get_idx(ix-1, iy, iz, grd.nyn, grd.nzn)];
            eta[0]  = d_parts.y[i] - d_grd.YN_flat[get_idx(ix, iy-1, iz, grd.nyn, grd.nzn)];
            zeta[0] = d_parts.z[i] - d_grd.ZN_flat[get_idx(ix, iy, iz-1, grd.nyn, grd.nzn)];

            xi[1]   = d_grd.XN_flat[get_idx(ix, iy, iz, grd.nyn, grd.nzn)] - d_parts.x[i];
            eta[1]  = d_grd.YN_flat[get_idx(ix, iy, iz, grd.nyn, grd.nzn)] - d_parts.y[i];
            zeta[1] = d_grd.ZN_flat[get_idx(ix, iy, iz, grd.nyn, grd.nzn)] - d_parts.z[i];

            for (int ii = 0; ii < 2; ii++)
                for (int jj = 0; jj < 2; jj++)
                    for (int kk = 0; kk < 2; kk++)
                        weight[ii][jj][kk] = xi[ii] * eta[jj] * zeta[kk] * grd.invVOL;
            
            // set to zero local electric and magnetic field
            Exl=0.0, Eyl = 0.0, Ezl = 0.0, Bxl = 0.0, Byl = 0.0, Bzl = 0.0;
            
            for (int ii=0; ii < 2; ii++)
                for (int jj=0; jj < 2; jj++)
                    for(int kk=0; kk < 2; kk++){
                        Exl += weight[ii][jj][kk]*d_fld.Ex_flat[get_idx(ix-ii, iy-jj, iz-kk, grd.nyn, grd.nzn)];
                        Eyl += weight[ii][jj][kk]*d_fld.Ey_flat[get_idx(ix-ii, iy-jj, iz-kk, grd.nyn, grd.nzn)];
                        Ezl += weight[ii][jj][kk]*d_fld.Ez_flat[get_idx(ix-ii, iy-jj, iz-kk, grd.nyn, grd.nzn)];
                        Bxl += weight[ii][jj][kk]*d_fld.Bxn_flat[get_idx(ix-ii, iy-jj, iz-kk, grd.nyn, grd.nzn)];
                        Byl += weight[ii][jj][kk]*d_fld.Byn_flat[get_idx(ix-ii, iy-jj, iz-kk, grd.nyn, grd.nzn)];
                        Bzl += weight[ii][jj][kk]*d_fld.Bzn_flat[get_idx(ix-ii, iy-jj, iz-kk, grd.nyn, grd.nzn)];
                    }
            
            // end interpolation
            omdtsq = qomdt2*qomdt2*(Bxl*Bxl+Byl*Byl+Bzl*Bzl);
            denom = 1.0/(1.0 + omdtsq);

            // solve the position equation
            ut= d_parts.u[i] + qomdt2*Exl;
            vt= d_parts.v[i] + qomdt2*Eyl;
            wt= d_parts.w[i] + qomdt2*Ezl;
            udotb = ut*Bxl + vt*Byl + wt*Bzl;
            // solve the velocity equation
            uptilde = (ut+qomdt2*(vt*Bzl -wt*Byl + qomdt2*udotb*Bxl))*denom;
            vptilde = (vt+qomdt2*(wt*Bxl -ut*Bzl + qomdt2*udotb*Byl))*denom;
            wptilde = (wt+qomdt2*(ut*Byl -vt*Bxl + qomdt2*udotb*Bzl))*denom;
            // update position
            d_parts.x[i] = xptilde + uptilde*dto2;
            d_parts.y[i] = yptilde + vptilde*dto2;
            d_parts.z[i] = zptilde + wptilde*dto2;
            
            
        } // end of iteration
        // update the final position and velocity
        d_parts.u[i]= 2.0*uptilde - d_parts.u[i];
        d_parts.v[i]= 2.0*vptilde - d_parts.v[i];
        d_parts.w[i]= 2.0*wptilde - d_parts.w[i];
        d_parts.x[i] = xptilde + uptilde*dt_sub_cycling;
        d_parts.y[i] = yptilde + vptilde*dt_sub_cycling;
        d_parts.z[i] = zptilde + wptilde*dt_sub_cycling;
                
        //////////
        //////////
        ////////// BC
                                    
        // X-DIRECTION: BC particles
        if (d_parts.x[i] > grd.Lx){
            if (param.PERIODICX==true){ // PERIODIC
                d_parts.x[i] = d_parts.x[i] - grd.Lx;
            } else { // REFLECTING BC
                d_parts.u[i] = -d_parts.u[i];
                d_parts.x[i] = 2*grd.Lx - d_parts.x[i];
            }
        }
                                                                    
        if (d_parts.x[i] < 0){
            if (param.PERIODICX==true){ // PERIODIC
                d_parts.x[i] = d_parts.x[i] + grd.Lx;
            } else { // REFLECTING BC
                d_parts.u[i] = -d_parts.u[i];
                d_parts.x[i] = -d_parts.x[i];
            }
        }
            
        // Y-DIRECTION: BC particles
        if (d_parts.y[i] > grd.Ly){
            if (param.PERIODICY==true){ // PERIODIC
                d_parts.y[i] = d_parts.y[i] - grd.Ly;
            } else { // REFLECTING BC
                d_parts.v[i] = -d_parts.v[i];
                d_parts.y[i] = 2*grd.Ly - d_parts.y[i];
            }
        }
                                                                    
        if (d_parts.y[i] < 0){
            if (param.PERIODICY==true){ // PERIODIC
                d_parts.y[i] = d_parts.y[i] + grd.Ly;
            } else { // REFLECTING BC
                d_parts.v[i] = -d_parts.v[i];
                d_parts.y[i] = -d_parts.y[i];
            }
        }
                                                                    
        // Z-DIRECTION: BC particles
        if (d_parts.z[i] > grd.Lz){
            if (param.PERIODICZ==true){ // PERIODIC
                d_parts.z[i] = d_parts.z[i] - grd.Lz;
            } else { // REFLECTING BC
                d_parts.w[i] = -d_parts.w[i];
                d_parts.z[i] = 2*grd.Lz - d_parts.z[i];
            }
        }
                                                                    
        if (d_parts.z[i] < 0){
            if (param.PERIODICZ==true){ // PERIODIC
                d_parts.z[i] = d_parts.z[i] + grd.Lz;
            } else { // REFLECTING BC
                d_parts.w[i] = -d_parts.w[i];
                d_parts.z[i] = -d_parts.z[i];
            }
        }
    }
    //if(i == 1) { printf("End of mover:\n x: %f, y: %f, z: %f, u: %f, v: %f, w: %f\n", d_parts.x[i], d_parts.y[i], d_parts.z[i], d_parts.u[i], d_parts.v[i], d_parts.w[i]);}
}

int mover_PC_gpu(struct particles* part, struct EMfield* field, struct grid* grd, struct parameters* param) {
    // print species and subcycling
    std::cout << "***  MOVER with SUBCYCLYING "<< param->n_sub_cycles << " - species " << part->species_ID << " ***" << std::endl;

    // Particle GPU allocation
    d_particles d_parts;
    FPpart * d_prt[6];
    for(int i = 0; i < 6; i++) {
        cudaMalloc(&d_prt[i], part->npmax * sizeof(FPpart));
    }
    d_parts.x = d_prt[0];
    d_parts.y = d_prt[1];
    d_parts.z = d_prt[2];
    d_parts.u = d_prt[3];
    d_parts.v = d_prt[4];
    d_parts.w = d_prt[5];

    cudaMemcpy((d_parts.x), (part->x), part->npmax*sizeof(FPpart), cudaMemcpyHostToDevice);
    cudaMemcpy((d_parts.y), (part->y), part->npmax*sizeof(FPpart), cudaMemcpyHostToDevice);
    cudaMemcpy((d_parts.z), (part->z), part->npmax*sizeof(FPpart), cudaMemcpyHostToDevice);
    cudaMemcpy((d_parts.u), (part->u), part->npmax*sizeof(FPpart), cudaMemcpyHostToDevice);
    cudaMemcpy((d_parts.v), (part->v), part->npmax*sizeof(FPpart), cudaMemcpyHostToDevice);
    cudaMemcpy((d_parts.w), (part->w), part->npmax*sizeof(FPpart), cudaMemcpyHostToDevice);

    // Grid GPU Allocation
    d_grid d_grd;
    FPfield * d_cnodes[3];
    for (int i = 0; i < 3; i++) {
        cudaMalloc(&d_cnodes[i], grd->nxn*grd->nyn*grd->nzn*sizeof(FPfield));
    }
    d_grd.XN_flat = d_cnodes[0];
    d_grd.YN_flat = d_cnodes[1];
    d_grd.ZN_flat = d_cnodes[2];

    cudaMemcpy((d_grd.XN_flat), (grd->XN_flat), grd->nxn*grd->nyn*grd->nzn*sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy((d_grd.YN_flat), (grd->YN_flat), grd->nxn*grd->nyn*grd->nzn*sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy((d_grd.ZN_flat), (grd->ZN_flat), grd->nxn*grd->nyn*grd->nzn*sizeof(FPfield), cudaMemcpyHostToDevice);

    // Field GPU Allocation
    d_EMfield d_fld;
    FPfield * d_enodes[6];
    for(int i = 0; i < 6; i++) {
        cudaMalloc(&d_enodes[i], grd->nxn*grd->nyn*grd->nzn*sizeof(FPfield));
    }

    d_fld.Ex_flat = d_enodes[0];
    d_fld.Ey_flat = d_enodes[1];
    d_fld.Ez_flat = d_enodes[2];
    d_fld.Bxn_flat = d_enodes[3];
    d_fld.Byn_flat = d_enodes[4];
    d_fld.Bzn_flat = d_enodes[5];

    cudaMemcpy((d_fld.Ex_flat), (field->Ex_flat), grd->nxn*grd->nyn*grd->nzn*sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy((d_fld.Ey_flat), (field->Ey_flat), grd->nxn*grd->nyn*grd->nzn*sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy((d_fld.Ez_flat), (field->Ez_flat), grd->nxn*grd->nyn*grd->nzn*sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy((d_fld.Bxn_flat), (field->Bxn_flat), grd->nxn*grd->nyn*grd->nzn*sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy((d_fld.Byn_flat), (field->Byn_flat), grd->nxn*grd->nyn*grd->nzn*sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy((d_fld.Bzn_flat), (field->Bzn_flat), grd->nxn*grd->nyn*grd->nzn*sizeof(FPfield), cudaMemcpyHostToDevice);
    
    mover_kernel<<<(part->nop + TPB - 1) / TPB, TPB>>>(part->n_sub_cycles, part->NiterMover, part->nop, part->qom, *grd, *param, d_parts, d_fld, d_grd);
    
    cudaDeviceSynchronize();

    cudaMemcpy(part->x, d_parts.x, part->npmax * sizeof(FPpart), cudaMemcpyDeviceToHost);
    cudaMemcpy(part->y, d_parts.y, part->npmax * sizeof(FPpart), cudaMemcpyDeviceToHost);
    cudaMemcpy(part->z, d_parts.z, part->npmax * sizeof(FPpart), cudaMemcpyDeviceToHost);
    cudaMemcpy(part->u, d_parts.u, part->npmax * sizeof(FPpart), cudaMemcpyDeviceToHost);
    cudaMemcpy(part->v, d_parts.v, part->npmax * sizeof(FPpart), cudaMemcpyDeviceToHost);
    cudaMemcpy(part->w, d_parts.w, part->npmax * sizeof(FPpart), cudaMemcpyDeviceToHost);

    cudaFree(d_parts.x);
    cudaFree(d_parts.y);
    cudaFree(d_parts.z);
    cudaFree(d_parts.u);
    cudaFree(d_parts.v);
    cudaFree(d_parts.w);

    cudaFree(d_grd.XN_flat);
    cudaFree(d_grd.YN_flat);
    cudaFree(d_grd.ZN_flat);
    
    cudaFree(d_fld.Ex_flat);
    cudaFree(d_fld.Ey_flat);
    cudaFree(d_fld.Ez_flat);
    cudaFree(d_fld.Bxn_flat);
    cudaFree(d_fld.Byn_flat);
    cudaFree(d_fld.Bzn_flat);
                                                                  
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
