#ifndef INIT_H
#define INIT_H
#include "struct_def.h"

void init_unit_test(gpudata gpu, fftdata fft, fielddata vel);

void initializeVelocity(gpudata gpu, fielddata vel);

void initializeScalar(gpudata gpu, fielddata vel);

void initializeWaveNumbers(gpudata gpu, griddata grid);

void initializeJet(fftdata fft, gpudata gpu, griddata grid, fielddata h_vel, fielddata vel, fielddata rhs);

void initializeTaylorGreen(gpudata gpu, fielddata vel);

#endif
