#ifndef SOLVER_PS
#define SOLVER_PS
#include "struct_def.h"

void solver_ps(const int euler, fftinfo fft, gpuinfo gpu, fielddata vel, fielddata rhs, fielddata rhs_old, double **k, cufftDoubleComplex **temp_advective);

void deAlias(gpuinfo gpu, double **k, fielddata vel);

void takeDerivative(int dir, gpuinfo gpu, double **waveNum, cufftDoubleComplex **f, cufftDoubleComplex **fIk);

#endif