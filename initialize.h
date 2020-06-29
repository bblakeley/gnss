#ifndef INIT_H
#define INIT_H
#include "struct_def.h"

void initializeData(gpudata gpu, fftdata fft, fielddata vel);

void initializeVelocity(gpudata gpu, fielddata vel);

void initializeScalar(gpudata gpu, fielddata vel);

void initializeWaveNumbers(gpudata gpu, griddata grid);

void initializeJet_Convolution(fftdata fft, gpudata gpu, fielddata h_vel, fielddata vel, fielddata rhs);

void initializeJet_Superposition(fftdata fft, gpudata gpu, griddata grid, fielddata h_vel, fielddata vel, fielddata rhs);

void initializeTaylorGreen(gpudata gpu, fielddata vel);

#endif
