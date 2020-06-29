#ifndef SOLVER_PS
#define SOLVER_PS
#include "struct_def.h"

void solver_ps(const int euler, fftdata fft, gpudata gpu, griddata grid, fielddata vel, fielddata rhs, fielddata rhs_old, fielddata temp);

void deAlias(gpudata gpu, griddata grid, fielddata vel);

void takeDerivative(int dir, gpudata gpu, griddata grid, cufftDoubleComplex **f, cufftDoubleComplex **fIk);

void vorticity(gpudata gpu, griddata grid, fielddata vel, fielddata rhs);

#endif
