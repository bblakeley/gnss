#ifndef FFT_H
#define FFT_H
#include "struct_def.h"

void plan1dFFT(int nGPUs, fftdata ftt);

void plan2dFFT(gpudata gpu, fftdata fft);

void plan3dFFT(fftdata fft);

void forwardTransform(fftdata fft, gpudata gpu, cufftDoubleReal **f );

void inverseTransform(fftdata fft, gpudata gpu, cufftDoubleComplex **f);

#endif
