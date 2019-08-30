#ifndef CUFUNCS_H
#define CUFUNCS_H

int divUp(int a, int b);

extern "C" __device__ int idxClip(int idx, int idxMax);

extern "C" __device__ int flatten(int col, int row, int stack, int width, int height, int depth);

extern "C" __global__ void scaleKernel_mgpu(int start_x, cufftDoubleReal *f);

void synchronizeGPUs(int nGPUs);

#endif