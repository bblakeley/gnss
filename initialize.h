#ifndef INIT_H
#define INIT_H
#include "struct_def.h"

void initializeData(gpuinfo gpu, fftinfo fft, fielddata vel);

void initializeVelocity(gpuinfo gpu, fielddata vel);

void initializeScalar(gpuinfo gpu, fielddata vel);

void initializeWaveNumbers(gpuinfo gpu, double **waveNum);

void initializeJet_Convolution(fftinfo fft, gpuinfo gpu, fielddata h_vel, fielddata vel, fielddata rhs);

void initializeJet_Superposition(fftinfo fft, gpuinfo gpu, double **wave, fielddata h_vel, fielddata vel, fielddata rhs);

void initializeTaylorGreen(gpuinfo gpu, fielddata vel);

#endif
