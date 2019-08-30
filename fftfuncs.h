#ifndef FFT_H
#define FFT_H
#include "struct_def.h"

void plan1dFFT(int nGPUs, fftinfo ftt);

void plan2dFFT(gpuinfo gpu, fftinfo fft);

void forwardTransform(fftinfo fft, gpuinfo gpu, cufftDoubleReal **f );

void inverseTransform(fftinfo fft, gpuinfo gpu, cufftDoubleComplex **f);

#endif