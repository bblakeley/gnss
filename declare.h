#ifndef DECLARE
#define DECLARE
#include "struct_def.h"

// Declare variables and structs to be used in the program
gpuinfo gpu;
fftinfo fft;
// Declare statistics variables on GPU
statistics *stats;
// Declare struct for storing averaged profiles of variables
profile Yprofile;      // X,Z averaged profiles of important variables

// Wavenumber vector
double **k;

// Velocity and scalar fields
fielddata h_vel;        // Primitive variables resident on the host
fielddata vel;          // Primitive variables resident on the GPU
fielddata rhs;          // 'Temporary' variable storing RHS of equations for timestepping
fielddata rhs_old;      // Storing previous RHS for Adams-Bashforth

// Temporary array for computing advective derivative
cufftDoubleComplex **temp_advective;
#endif
