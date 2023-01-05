#ifndef PARTICLES_H
#define PARTICLES_H

#include <math.h>

#include "Alloc.h"
#include "Parameters.h"
#include "PrecisionTypes.h"
#include "Grid.h"
#include "EMfield.h"
#include "InterpDensSpecies.h"
#include <cuda_fp16.h>

struct particles {
    
    /** species ID: 0, 1, 2 , ... */
    int species_ID;
    
    /** maximum number of particles of this species on this domain. used for memory allocation */
    long npmax;
    /** number of particles of this species on this domain */
    long nop;
    
    /** Electron and ions have different number of iterations: ions moves slower than ions */
    int NiterMover;
    /** number of particle of subcycles in the mover */
    int n_sub_cycles;
    
    
    /** number of particles per cell */
    int npcel;
    /** number of particles per cell - X direction */
    int npcelx;
    /** number of particles per cell - Y direction */
    int npcely;
    /** number of particles per cell - Z direction */
    int npcelz;
    
    
    /** charge over mass ratio */
    FPpart qom;
    
    /* drift and thermal velocities for this species */
    FPpart u0, v0, w0;
    FPpart uth, vth, wth;
    
    /** particle arrays: 1D arrays[npmax] */
    FPpart* x; FPpart*  y; FPpart* z; FPpart* u; FPpart* v; FPpart* w;
    /** q must have precision of interpolated quantities: typically double. Not used in mover */
    FPinterp* q;
};

struct d_particles {
    half2* x; half2* y; half2* z;
    half2* u; half2* v; half2* w;
    float * temp_parts[6];
};

/** allocate particle arrays */
void particle_allocate(struct parameters*, struct particles*, int);

/** deallocate */
void particle_deallocate(struct particles*);

/** particle mover */
int mover_PC(struct particles*, struct EMfield*, struct grid*, struct parameters*);

/** Interpolation Particle --> Grid: This is for species */
void interpP2G(struct particles*, struct interpDensSpecies*, struct grid*);

int mover_PC_gpu(struct particles* part, struct EMfield* field, struct grid* grd, struct parameters* param, struct d_particles* d_parts, struct d_grid *d_grd, struct d_EMfield * d_fld, double& kernelTotTime);

void grid_to_fp16(struct grid*, struct d_grid*);
void free_d_grid(struct d_grid* d_grd);

void field_to_fp16(struct grid* grd, struct EMfield *field, struct d_EMfield *d_fld);
void free_d_field(struct d_EMfield*);

void parts_to_fp16(struct particles* parts, struct d_particles *d_parts);
void parts_to_float(struct particles* parts, struct d_particles *d_parts);
#endif
