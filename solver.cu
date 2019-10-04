#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cufft.h>
#include <cuComplex.h>
#include <helper_functions.h>
#include <helper_cuda.h>

// include parameters for DNS
#include "dnsparams.h"
#include "cudafuncs.h"
#include "fftfuncs.h"
#include "struct_def.h"

__global__
void deAliasKernel_mgpu(int start_y, double *waveNum, cufftDoubleComplex *fhat){

	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int j = blockIdx.y * blockDim.y + threadIdx.y;
	const int k = blockIdx.z * blockDim.z + threadIdx.z;
	if (( i >= NX) || ((j+start_y) >= NY) || (k >= NZ2)) return;
	const int idx = flatten( j, i, k, NY, NX, NZ2);

	double k_sq = waveNum[i]*waveNum[i] + waveNum[(j+start_y)]*waveNum[(j+start_y)] + waveNum[k]*waveNum[k];

	if( k_sq > (k_max*k_max) )
	{
		fhat[idx].x = 0.0;
		fhat[idx].y = 0.0;
	}

	return;
}

void deAlias(gpuinfo gpu, double **k, fielddata vel)
{	// Truncate data for de-aliasing

	int n;
	for(n = 0; n<gpu.nGPUs; ++n){
		cudaSetDevice(n);

		// Set thread and block dimensions for kernal calls
		const dim3 blockSize(TX, TY, TZ);
		const dim3 gridSize(divUp(NX, TX), divUp(gpu.ny[n], TY), divUp(NZ2, TZ));

		// Call the kernel
		deAliasKernel_mgpu<<<gridSize, blockSize>>>(gpu.start_y[n], k[n], vel.uh[n]);
		deAliasKernel_mgpu<<<gridSize, blockSize>>>(gpu.start_y[n], k[n], vel.vh[n]);
		deAliasKernel_mgpu<<<gridSize, blockSize>>>(gpu.start_y[n], k[n], vel.wh[n]);
		deAliasKernel_mgpu<<<gridSize, blockSize>>>(gpu.start_y[n], k[n], vel.sh[n]);
	}
	
	return;
}

__global__
void calcOmega1Kernel_mgpu(int start_y, double *waveNum, cufftDoubleComplex *u2hat, cufftDoubleComplex *u3hat, cufftDoubleComplex *omega1){

	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int j = blockIdx.y * blockDim.y + threadIdx.y;
	const int k = blockIdx.z * blockDim.z + threadIdx.z;
	if ((i >= NX) || ((j + start_y) >= NY) || (k >= NZ2)) return;
	const int idx = flatten( j, i, k, NY, NX, NZ2);

	omega1[idx].x = -waveNum[(j + start_y)]*u3hat[idx].y + waveNum[k]*u2hat[idx].y;
	omega1[idx].y = waveNum[(j + start_y)]*u3hat[idx].x - waveNum[k]*u2hat[idx].x;

	return;
}

__global__
void calcOmega2Kernel_mgpu(int start_y, double *waveNum, cufftDoubleComplex *u1hat, cufftDoubleComplex *u3hat, cufftDoubleComplex *omega2){

	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int j = blockIdx.y * blockDim.y + threadIdx.y;
	const int k = blockIdx.z * blockDim.z + threadIdx.z;
	if ((i >= NX) || ((j + start_y) >= NY) || (k >= NZ2)) return;
	const int idx = flatten( j, i, k, NY, NX, NZ2);

	omega2[idx].x = waveNum[i]*u3hat[idx].y - waveNum[k]*u1hat[idx].y;
	omega2[idx].y = -waveNum[i]*u3hat[idx].x + waveNum[k]*u1hat[idx].x;

	return;
}

__global__
void calcOmega3Kernel_mgpu(int start_y, double *waveNum, cufftDoubleComplex *u1hat, cufftDoubleComplex *u2hat, cufftDoubleComplex *omega3){

	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int j = blockIdx.y * blockDim.y + threadIdx.y;
	const int k = blockIdx.z * blockDim.z + threadIdx.z;
	if ((i >= NX) || ((j + start_y) >= NY) || (k >= NZ2)) return;
	const int idx = flatten( j, i, k, NY, NX, NZ2);

	omega3[idx].x = -waveNum[i]*u2hat[idx].y + waveNum[(j + start_y)]*u1hat[idx].y;
	omega3[idx].y = waveNum[i]*u2hat[idx].x - waveNum[(j + start_y)]*u1hat[idx].x;

	return;
}

void calcVorticity(gpuinfo gpu, double **waveNum, fielddata vel, fielddata rhs){
	// Function to calculate the vorticity in Fourier Space and transform to physical space
	
	int n;
	for(n=0; n<gpu.nGPUs; ++n){
		cudaSetDevice(n);

		// Set thread and block dimensions for kernal calls
		const dim3 blockSize(TX, TY, TZ);
		const dim3 gridSize(divUp(NX, TX), divUp(gpu.ny[n], TY), divUp(NZ2, TZ));

		// Call kernels to calculate vorticity
		calcOmega1Kernel_mgpu<<<gridSize, blockSize>>>(gpu.start_y[n], waveNum[n], vel.vh[n], vel.wh[n], rhs.uh[n]);
		calcOmega2Kernel_mgpu<<<gridSize, blockSize>>>(gpu.start_y[n], waveNum[n], vel.uh[n], vel.wh[n], rhs.vh[n]);
		calcOmega3Kernel_mgpu<<<gridSize, blockSize>>>(gpu.start_y[n], waveNum[n], vel.uh[n], vel.vh[n], rhs.wh[n]);
		// Kernel calls include scaling for post-FFT
	}

	// printf("Vorticity calculated in fourier space...\n");

	return;
}

__global__
void CrossProductKernel_mgpu(int start_x, cufftDoubleReal *u1, cufftDoubleReal *u2, cufftDoubleReal *u3, cufftDoubleReal *omega1, cufftDoubleReal *omega2, cufftDoubleReal *omega3){

	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int j = blockIdx.y * blockDim.y + threadIdx.y;
	const int k = blockIdx.z * blockDim.z + threadIdx.z;
	if (((i + start_x) >= NX) || (j >= NY) || (k >= NZ)) return;
	const int idx = flatten(i, j, k, NX, NY, 2*NZ2);

	// Load values into register memory (would be overwritten if not loaded into memory)
	double w1 = omega1[idx];
	double w2 = omega2[idx];
	double w3 = omega3[idx];

	__syncthreads();

	// Direction 1
	omega1[idx] = w2*u3[idx] - w3*u2[idx];
	// Direction 2
	omega2[idx] = -w1*u3[idx] + w3*u1[idx];
	// Direction 3
	omega3[idx] = w1*u2[idx] - w2*u1[idx];

	return;
}

void formCrossProduct(gpuinfo gpu, fielddata vel, fielddata rhs){
// Function to evaluate omega x u in physical space and then transform the result to Fourier Space

	int n;
	for (n = 0; n<gpu.nGPUs; ++n){
		cudaSetDevice(n);

		const dim3 blockSize(TX, TY, TZ);
		const dim3 gridSize(divUp(gpu.nx[n], TX), divUp(NY, TY), divUp(NZ, TZ));

		// Call kernel to calculate vorticity
		CrossProductKernel_mgpu<<<gridSize, blockSize>>>(gpu.start_x[n], vel.u[n], vel.v[n], vel.w[n], rhs.u[n], rhs.v[n], rhs.w[n]);

		cudaDeviceSynchronize();
	}

	// printf("Cross Product calculated!\n");

	return;
}

__global__
void multIkKernel_mgpu(const int dir, int start_y, double *waveNum, cufftDoubleComplex *f, cufftDoubleComplex *fIk)
{   // Multiples an input array by ik 
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int j = blockIdx.y * blockDim.y + threadIdx.y;
	const int k = blockIdx.z * blockDim.z + threadIdx.z;
	if (( i >= NX) || ((j+start_y) >= NY) || (k >= NZ2)) return;
	const int idx = flatten( j, i, k, NY, NX, NZ2);

	if(dir == 1){
		fIk[idx].x = -waveNum[i]*f[idx].y;     // Scaling results for when the inverse FFT is taken
		fIk[idx].y = waveNum[i]*f[idx].x;
	}

	if(dir == 2){
		fIk[idx].x = -waveNum[j+start_y]*f[idx].y;     // Scaling results for when the inverse FFT is taken
		fIk[idx].y = waveNum[j+start_y]*f[idx].x;
	}

	if(dir == 3){
		fIk[idx].x = -waveNum[k]*f[idx].y;     // Scaling results for when the inverse FFT is taken
		fIk[idx].y = waveNum[k]*f[idx].x;
	}

	return;
}

void takeDerivative(int dir, gpuinfo gpu, double **waveNum, cufftDoubleComplex **f, cufftDoubleComplex **fIk)
{
	// Loop through GPUs and multiply by iK
	// Note: Data assumed to be transposed during 3D FFt process; k + NZ*i + NZ*NX*j
	int n;
	for(n = 0; n<gpu.nGPUs; ++n){
		cudaSetDevice(n);
		
		const dim3 blockSize(TX, TY, TZ);
		const dim3 gridSize(divUp(NX, TX), divUp(gpu.ny[n], TY), divUp(NZ2, TZ));

		// Take derivative (dir = 1 => x-direction, 2 => y-direction, 3 => z-direction)
		multIkKernel_mgpu<<<gridSize, blockSize>>>(dir, gpu.start_y[n], waveNum[n], f[n], fIk[n]);
	}

	return;  
}

__global__
void multAndAddKernel_mgpu(int start_x, cufftDoubleReal *f1, cufftDoubleReal *f2, cufftDoubleReal *f3)
{	// Function to compute the non-linear terms on the RHS

	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int j = blockIdx.y * blockDim.y + threadIdx.y;
	const int k = blockIdx.z * blockDim.z + threadIdx.z;
	if (((i + start_x) >= NX) || (j >= NY) || (k >= NZ)) return;
	const int idx = flatten(i, j, k, NX, NY, 2*NZ2);

	f3[idx] = f3[idx] + f1[idx] * f2[idx];
		
	return;
}

void multAndAdd( gpuinfo gpu, cufftDoubleReal **f1, cufftDoubleReal **f2, cufftDoubleReal **f3)
{
	// Loop through GPUs and perform operation: f1*f2 + f3 = f3
	// Note: Data assumed to be transposed during 3D FFt process; k + NZ*i + NZ*NX*j
	int n;
	for(n = 0; n<gpu.nGPUs; ++n){
		cudaSetDevice(n);
		
		const dim3 blockSize(TX, TY, TZ);
		const dim3 gridSize(divUp(gpu.nx[n], TX), divUp(NY, TY), divUp(NZ, TZ));

		// Take derivative (dir = 1 => x-direction, 2 => y-direction, 3 => z-direction)
		multAndAddKernel_mgpu<<<gridSize, blockSize>>>(gpu.start_x[n], f1[n], f2[n], f3[n]);
	}

	return;  
}

void formScalarAdvection(fftinfo fft, gpuinfo gpu, cufftDoubleComplex **temp_advective, double **k, fielddata vel, fielddata rhs)
{	// Compute the advection term in the scalar equation

	// Zero out right hand side term before beginning calculation
	int n;
	for(n = 0; n<gpu.nGPUs; ++n){
		cudaSetDevice(n);
		checkCudaErrors( cudaMemset(rhs.s[n], 0, sizeof(cufftDoubleComplex)*NZ2*NX*gpu.ny[n]) );
	}

	//===============================================================
	// ( u \dot grad ) z = u * dZ/dx + v * dZ/dy + w * dZ/dz
	//===============================================================

	// Calculate u*dZdx and add it to RHS
	// Find du/dz in Fourier space
	takeDerivative(1, gpu, k, vel.sh, temp_advective);
	// Transform du/dz to physical space
	inverseTransform(fft, gpu, temp_advective);
	// Form term and add to RHS
	multAndAdd(gpu, vel.u, (cufftDoubleReal **)temp_advective, rhs.s);


	// Calculate v*dZdy and add it to RHS
	// Find du/dz in Fourier space
	takeDerivative(2, gpu, k, vel.sh, temp_advective);
	// Transform du/dz to physical space
	inverseTransform(fft, gpu, temp_advective);
	// Form term and add to RHS
	multAndAdd(gpu, vel.v, (cufftDoubleReal **)temp_advective, rhs.s);


	// Calculate w*dZdz and add it to RHS
	// Find du/dz in Fourier space
	takeDerivative(3, gpu, k, vel.sh, temp_advective);
	// Transform du/dz to physical space
	inverseTransform(fft, gpu, temp_advective);
	// Form term and add to RHS
	multAndAdd(gpu, vel.w, (cufftDoubleReal **)temp_advective, rhs.s);

	// rhs_z now holds the advective terms of the scalar equation (in physical domain). 
	// printf("Scalar advection terms formed...\n");

	return;
}

__global__
void computeRHSKernel_mgpu(int start_y, double *k1, cufftDoubleComplex *rhs_u1, cufftDoubleComplex *rhs_u2, cufftDoubleComplex *rhs_u3, cufftDoubleComplex *rhs_Z)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int j = blockIdx.y * blockDim.y + threadIdx.y;
	const int k = blockIdx.z * blockDim.z + threadIdx.z;
	if (( i >= NX) || ((j+start_y) >= NY) || (k >= NZ2)) return;
	const int idx = flatten( j, i, k, NY, NX, NZ2);

	// if(i == 0 && j == 0 && k ==0){printf("Calling computeRHS kernel\n");}

	// Move RHS into register memory (otherwise the values would be overwritten)
	double temp1_r = rhs_u1[idx].x;
	double temp1_c = rhs_u1[idx].y;

	double temp2_r = rhs_u2[idx].x;
	double temp2_c = rhs_u2[idx].y;

	double temp3_r = rhs_u3[idx].x;
	double temp3_c = rhs_u3[idx].y;

	// Calculate k^2 for each index
	double k_sq = k1[i]*k1[i] + k1[(j+start_y)]*k1[(j+start_y)] + k1[k]*k1[k];

	// Form RHS
	if( i == 0 && (j+start_y) == 0 && k == 0){
		rhs_u1[idx].x = 0.0;
		rhs_u1[idx].y = 0.0;

		rhs_u2[idx].x = 0.0;
		rhs_u2[idx].y = 0.0;

		rhs_u3[idx].x = 0.0;
		rhs_u3[idx].y = 0.0;

		rhs_Z[idx].x = 0.0;
		rhs_Z[idx].y = 0.0;
	}
	else {
		rhs_u1[idx].x = (k1[i]*k1[i] / k_sq - 1.0)*temp1_r + (k1[i]*k1[(j+start_y)] / k_sq)*temp2_r + (k1[i]*k1[k] / k_sq)*temp3_r;
		rhs_u1[idx].y = (k1[i]*k1[i] / k_sq - 1.0)*temp1_c + (k1[i]*k1[(j+start_y)] / k_sq)*temp2_c + (k1[i]*k1[k] / k_sq)*temp3_c;

		rhs_u2[idx].x = (k1[(j+start_y)]*k1[i] / k_sq)*temp1_r + (k1[(j+start_y)]*k1[(j+start_y)] / k_sq - 1.0)*temp2_r + (k1[(j+start_y)]*k1[k] / k_sq)*temp3_r;
		rhs_u2[idx].y = (k1[(j+start_y)]*k1[i] / k_sq)*temp1_c + (k1[(j+start_y)]*k1[(j+start_y)] / k_sq - 1.0)*temp2_c + (k1[(j+start_y)]*k1[k] / k_sq)*temp3_c;

		rhs_u3[idx].x = (k1[k]*k1[i] / k_sq)*temp1_r + (k1[k]*k1[(j+start_y)] / k_sq)*temp2_r + (k1[k]*k1[k] / k_sq - 1.0)*temp3_r;
		rhs_u3[idx].y = (k1[k]*k1[i] / k_sq)*temp1_c + (k1[k]*k1[(j+start_y)] / k_sq)*temp2_c + (k1[k]*k1[k] / k_sq - 1.0)*temp3_c;

		rhs_Z[idx].x = -rhs_Z[idx].x;
		rhs_Z[idx].y = -rhs_Z[idx].y;
	}

	return;
}

void makeRHS(gpuinfo gpu, double **waveNum, fielddata rhs)
{	// Function to create the rhs of the N-S equations in Fourier Space

	int n;
	for(n=0; n<gpu.nGPUs; ++n){
		cudaSetDevice(n);

		// Set thread and block dimensions for kernal calls
		const dim3 blockSize(TX, TY, TZ);
		const dim3 gridSize(divUp(NX, TX), divUp(gpu.ny[n], TY), divUp(NZ2, TZ));

		// Call the kernel
		computeRHSKernel_mgpu<<<gridSize, blockSize>>>(gpu.start_y[n], waveNum[n], rhs.uh[n], rhs.vh[n], rhs.wh[n], rhs.sh[n]);
		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess) 
	    printf("Error: %s\n", cudaGetErrorString(err));	
	}

	// printf("Right hand side of equations formed!\n");

	return;
}


__global__
void eulerKernel_mgpu(double num, int start_y, double *waveNum, cufftDoubleComplex *fhat,  cufftDoubleComplex *rhs_f)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int j = blockIdx.y * blockDim.y + threadIdx.y;
	const int k = blockIdx.z * blockDim.z + threadIdx.z;
	if (( i >= NX) || ((j+start_y) >= NY) || (k >= NZ2)) return;
	const int idx = flatten( j, i, k, NY, NX, NZ2);

	// Calculate k^2 for each index
	double k_sq = waveNum[i]*waveNum[i] + waveNum[(j+start_y)]*waveNum[(j+start_y)] + waveNum[k]*waveNum[k];

	// Timestep in X-direction
	fhat[idx].x = ( (1.0 - dt/2.0*k_sq/num)*fhat[idx].x + dt * rhs_f[idx].x ) / (1.0 + dt/2.0*k_sq/num);
	fhat[idx].y = ( (1.0 - dt/2.0*k_sq/num)*fhat[idx].y + dt * rhs_f[idx].y ) / (1.0 + dt/2.0*k_sq/num);

	return;
}

__global__
void adamsBashforthKernel_mgpu(double num, int start_y, double *waveNum, cufftDoubleComplex *fhat, cufftDoubleComplex *rhs_f, cufftDoubleComplex *rhs_f_old)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int j = blockIdx.y * blockDim.y + threadIdx.y;
	const int k = blockIdx.z * blockDim.z + threadIdx.z;
	if (( i >= NX) || ((j+start_y) >= NY) || (k >= NZ2)) return;
	const int idx = flatten( j, i, k, NY, NX, NZ2);

	// Calculate k^2 for each index
	double k_sq = waveNum[i]*waveNum[i] + waveNum[(j+start_y)]*waveNum[(j+start_y)] + waveNum[k]*waveNum[k];

	// Timestep in X-direction
	fhat[idx].x = ( (1.0 - dt/2.0*k_sq/num)*fhat[idx].x + dt * (1.5*rhs_f[idx].x - 0.5*rhs_f_old[idx].x) ) / (1.0 + dt/2.0*k_sq/num);
	fhat[idx].y = ( (1.0 - dt/2.0*k_sq/num)*fhat[idx].y + dt * (1.5*rhs_f[idx].y - 0.5*rhs_f_old[idx].y) ) / (1.0 + dt/2.0*k_sq/num);

	return;
}

void timestep(const int flag, gpuinfo gpu, double **k, fielddata vel, fielddata rhs, fielddata rhs_old)
{
	int n;
	for(n=0; n<gpu.nGPUs; ++n){
		cudaSetDevice(n);

		// Set thread and block dimensions for kernal calls
		const dim3 blockSize(TX, TY, TZ);
		const dim3 gridSize(divUp(NX, TX), divUp(gpu.ny[n], TY), divUp(NZ2, TZ));

		if(flag){
			// printf("Using Euler Method\n");
			eulerKernel_mgpu<<<gridSize, blockSize>>>((double) Re,    gpu.start_y[n], k[n], vel.uh[n], rhs.uh[n]);
			eulerKernel_mgpu<<<gridSize, blockSize>>>((double) Re,    gpu.start_y[n], k[n], vel.vh[n], rhs.vh[n]);
			eulerKernel_mgpu<<<gridSize, blockSize>>>((double) Re,    gpu.start_y[n], k[n], vel.wh[n], rhs.wh[n]);
			eulerKernel_mgpu<<<gridSize, blockSize>>>((double) Re*Sc, gpu.start_y[n], k[n], vel.sh[n], rhs.sh[n]);
		}
		else {
			// printf("Using A-B Method\n");
			adamsBashforthKernel_mgpu<<<gridSize, blockSize>>>((double) Re,    gpu.start_y[n], k[n], vel.uh[n], rhs.uh[n], rhs_old.uh[n]);
			adamsBashforthKernel_mgpu<<<gridSize, blockSize>>>((double) Re,    gpu.start_y[n], k[n], vel.vh[n], rhs.vh[n], rhs_old.vh[n]);
			adamsBashforthKernel_mgpu<<<gridSize, blockSize>>>((double) Re,    gpu.start_y[n], k[n], vel.wh[n], rhs.wh[n], rhs_old.wh[n]);
			adamsBashforthKernel_mgpu<<<gridSize, blockSize>>>((double) Re*Sc, gpu.start_y[n], k[n], vel.sh[n], rhs.sh[n], rhs_old.sh[n]);
		}
	}

	return;
}

__global__
void updateKernel_mgpu(int start_y, cufftDoubleComplex *rhs_f, cufftDoubleComplex *rhs_f_old)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int j = blockIdx.y * blockDim.y + threadIdx.y;
	const int k = blockIdx.z * blockDim.z + threadIdx.z;
	if (( i >= NX) || ((j+start_y) >= NY) || (k >= NZ2)) return;
	const int idx = flatten( j, i, k, NY, NX, NZ2);

	// Update old variables to store current iteration
	rhs_f_old[idx].x = rhs_f[idx].x;
	rhs_f_old[idx].y = rhs_f[idx].y;

	// Zero out RHS arrays
	rhs_f[idx].x = 0.0;
	rhs_f[idx].y = 0.0;

	return;
}

void update(gpuinfo gpu, fielddata rhs, fielddata rhs_old)
{
	int n;
	for(n=0; n<gpu.nGPUs; ++n){
		cudaSetDevice(n);

		// Set thread and block dimensions for kernal calls
		const dim3 blockSize(TX, TY, TZ);
		const dim3 gridSize(divUp(NX, TX), divUp(gpu.ny[n], TY), divUp(NZ2, TZ));

		updateKernel_mgpu<<<gridSize, blockSize>>>(gpu.start_y[n], rhs.uh[n], rhs_old.uh[n]);
		updateKernel_mgpu<<<gridSize, blockSize>>>(gpu.start_y[n], rhs.vh[n], rhs_old.vh[n]);
		updateKernel_mgpu<<<gridSize, blockSize>>>(gpu.start_y[n], rhs.wh[n], rhs_old.wh[n]);
		updateKernel_mgpu<<<gridSize, blockSize>>>(gpu.start_y[n], rhs.sh[n], rhs_old.sh[n]);
	}

	return;
}

__global__
void scalarFilterkernel_mgpu(int start_x, cufftDoubleReal *f)
{	// Function to compute the non-linear terms on the RHS

	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int j = blockIdx.y * blockDim.y + threadIdx.y;
	const int k = blockIdx.z * blockDim.z + threadIdx.z;
	if (((i + start_x) >= NX) || (j >= NY) || (k >= NZ)) return;
	const int idx = flatten(i, j, k, NX, NY, 2*NZ2);

	// Check is value is greater than 1 or less than 0
	if (f[idx] > 1.0)
		f[idx] = 1.0;
	else if (f[idx] < 0.0)
		f[idx] = 0.0;
	else
		return;
	
	return;
}

void scalarFilter(gpuinfo gpu, cufftDoubleReal **f)
{
	// Loops through data and looks for spurious values of the scalar field
	// For a conserved, passive scalar the value should always be 0<=Z<=1

	int n;
	for(n = 0; n<gpu.nGPUs; ++n){
		cudaSetDevice(n);
		
		const dim3 blockSize(TX, TY, TZ);
		const dim3 gridSize(divUp(gpu.nx[n], TX), divUp(NY, TY), divUp(NZ, TZ));

		// Apply filter to scalar field in physical space
		scalarFilterkernel_mgpu<<<gridSize, blockSize>>>(gpu.start_x[n], f[n]);
	}

	return;  
}

void solver_ps(const int euler, fftinfo fft, gpuinfo gpu, fielddata vel, fielddata rhs, fielddata rhs_old, double **k, cufftDoubleComplex **temp_advective)
{	// Pseudo spectral Navier-Stokes solver with conserved, passive scalar field

	// Form the vorticity in Fourier space
	calcVorticity(gpu, k, vel, rhs);

	// Inverse Fourier Transform the vorticity to physical space.
	inverseTransform(fft, gpu, rhs.uh);
	inverseTransform(fft, gpu, rhs.vh);
	inverseTransform(fft, gpu, rhs.wh);

	// printf("Vorticity transformed to physical coordinates...\n");

	// Inverse transform the velocity to physical space to for advective terms
	inverseTransform(fft, gpu, vel.uh);
	inverseTransform(fft, gpu, vel.vh);
	inverseTransform(fft, gpu, vel.wh);

	// Form non-linear terms in physical space
	formCrossProduct(gpu, vel, rhs);

	// Transform omegaXu from physical space to fourier space 
	forwardTransform(fft, gpu, rhs.u);
	forwardTransform(fft, gpu, rhs.v);
	forwardTransform(fft, gpu, rhs.w);

	// Form advective terms in scalar equation
	formScalarAdvection(fft, gpu, temp_advective, k, vel, rhs);
	
	// Transform the non-linear term in rhs from physical space to Fourier space for timestepping
	forwardTransform(fft, gpu, rhs.s);

	// Transform velocity back to fourier space for timestepping
	forwardTransform(fft, gpu, vel.u);
	forwardTransform(fft, gpu, vel.v);
	forwardTransform(fft, gpu, vel.w);

	// Form right hand side of the N-S and scalar equations
	makeRHS(gpu, k, rhs);

	// Dealias the solution by truncating RHS
	deAlias(gpu, k, rhs);

	// Step the vector fields forward in time
	timestep(euler, gpu, k, vel, rhs, rhs_old);

	// Update loop variables to next timestep
	update(gpu, rhs, rhs_old);

	// // Remove spurious scalar values
	// inverseTransform(fft, gpus, vel.sh);
	// scalarFilter(gpus, vel.s);
	// forwardTransform(fft, gpus, vel.s);

	return;
}
