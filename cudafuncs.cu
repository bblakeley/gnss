#include <cuda_runtime.h>
#include <cufft.h>
#include <cuComplex.h>
#include "dnsparams.h"

int divUp(int a, int b) { return (a + b - 1) / b; }

extern "C" __device__
int idxClip(int idx, int idxMax){
	return idx > (idxMax - 1) ? (idxMax - 1) : (idx < 0 ? 0 : idx);
}

extern "C" __device__
int flatten(int col, int row, int stack, int width, int height, int depth){
	return idxClip(stack, depth) + idxClip(row, height)*depth + idxClip(col, width)*depth*height;
	// Note: using column-major indexing format
}

extern "C" __global__
void scaleKernel_mgpu(int start_x, cufftDoubleReal *f)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int j = blockIdx.y * blockDim.y + threadIdx.y;
	const int k = blockIdx.z * blockDim.z + threadIdx.z;
	if (( (i + start_x) >= NX) || (j >= NY) || (k >= NZ)) return;
	const int idx = flatten( i, j, k, NX, NY, 2*NZ2);

	f[idx] = f[idx] / ( (double)NX*NY*NZ );

	return;
}

void synchronizeGPUs(int nGPUs){
	// Synchronize all GPUs on the node
	int n;

	for(n=0; n<nGPUs; ++n){
		cudaSetDevice(n);
		cudaDeviceSynchronize();
	}

	return;
}