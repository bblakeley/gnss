#ifndef DECLARE
#define DECLARE
#include "struct_def.h"

// Declare variables and structs to be used in the program
gpuinfo gpu;
fftinfo fft;
// Declare struct for holding statistics on the host
statistics h_stats;
// Declare statistics variables on GPU
statistics *stats;

// Wavenumber vector
double **k;

// Velocity and scalar fields
fielddata h_vel;
fielddata vel;
fielddata rhs;
fielddata rhs_old;

// Temporary array for computing advective derivative
cufftDoubleComplex **temp_advective;
#endif
