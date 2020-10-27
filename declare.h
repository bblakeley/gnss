#ifndef DECLARE
#define DECLARE
#include "struct_def.h"

// Declare variables and structs to be used in the program
gpudata gpu;
fftdata fft;

// Declare statistics variables on GPU
statistics *stats;

// Declare struct for storing averaged profiles of variables
profile Yprof;      // X,Z averaged profiles of important variables

// Wavenumber vector
griddata h_grid;
griddata grid;

// Velocity and scalar fields
fielddata h_vel;        // Primitive variables resident on the host
fielddata vel;          // Primitive variables resident on the GPU
fielddata rhs;          // 'Temporary' variable storing RHS of equations for timestepping
fielddata rhs_old;      // Storing previous RHS for Adams-Bashforth
fielddata temp;         // Vector of 3D temporary arrays (primarily used for colloids)

#endif
