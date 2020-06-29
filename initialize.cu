#include <stdio.h>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <math.h>
#include <cufft.h>
#include <complex.h>
#include <cuComplex.h>

#include "dnsparams.h"
#include "cudafuncs.h"
#include "initialize.h"
#include "fftfuncs.h"
#include "solver.h"
#include "iofuncs.h"

__global__ 
void initializeTGkernel(int start_x, cufftDoubleReal *f1, cufftDoubleReal *f2, cufftDoubleReal *f3, cufftDoubleReal *f4)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int j = blockIdx.y * blockDim.y + threadIdx.y;
	const int k = blockIdx.z * blockDim.z + threadIdx.z;
	if (( (i+start_x) >= NX) || (j >= NY) || (k >= NZ)) return;
	const int idx = flatten( i, j, k, NX, NY, 2*NZ2);	// Index local to the GPU

/*	// For domain centered at (pi,pi,pi)
	double x = (i + start_x) * (double)LX / NX;
	double y = j * (double)LY / NY;
	double z = k * (double)LZ / NZ; */

	// For domain centered at (0,0,0):
	double x = -(double)LX/2 + (i + start_x)*(double)LX/NX ;
	double y = -(double)LY/2 + j*(double)LY/NY;
	double z = -(double)LZ/2 + k*(double)LZ/NZ;

	// Initialize starting array - Taylor Green Vortex
	f1[idx] = sin(x)*cos(y)*cos(z);		// u
	f2[idx] = -cos(x)*sin(y)*cos(z);	// v
	f3[idx] = 0.0;						// w
	f4[idx] = 0.5 - 0.5*tanh( H/(4.0*theta)*( 2.0*fabs(y)/H - 1.0 ));	// z

	return;
}

void initializeTaylorGreen(gpudata gpu, fielddata vel)
{
	int n;
	for (n = 0; n<gpu.nGPUs; ++n){
		cudaSetDevice(n);

		const dim3 blockSize(TX, TY, TZ);
		const dim3 gridSize(divUp(gpu.nx[n], TX), divUp(NY, TY), divUp(NZ, TZ));

		initializeTGkernel<<<gridSize, blockSize>>>(gpu.start_x[n], vel.u[n], vel.v[n], vel.w[n], vel.s[n]);
		printf("Velocity initialized on GPU #%d...\n",n);
	}

	return;
}

__global__
void hpFilterKernel_mgpu(int start_y, double *waveNum, cufftDoubleComplex *fhat){

	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int j = blockIdx.y * blockDim.y + threadIdx.y;
	const int k = blockIdx.z * blockDim.z + threadIdx.z;
	if (( i >= NX) || ((j+start_y) >= NY) || (k >= NZ2)) return;
	const int idx = flatten( j, i, k, NY, NX, NZ2);

	double k_sq = waveNum[i]*waveNum[i] + waveNum[(j+start_y)]*waveNum[(j+start_y)] + waveNum[k]*waveNum[k];

	if( k_sq <= (k_fil*k_fil) )
	{
		fhat[idx].x = 0.0;
		fhat[idx].y = 0.0;
	}

	return;
}

void hpFilter(gpudata gpu, fftdata fft, griddata grid, fielddata vel)
{	// Filter out low wavenumbers

	// Transform isotropic noise (stored in rhs_u) to Fourier Space
	forwardTransform(fft, gpu, vel.u);
	forwardTransform(fft, gpu, vel.v);
	forwardTransform(fft, gpu, vel.w);

	int n;
	for(n = 0; n<gpu.nGPUs; ++n){
		cudaSetDevice(n);

		// Set thread and block dimensions for kernal calls
		const dim3 blockSize(TX, TY, TZ);
		const dim3 gridSize(divUp(NX, TX), divUp(gpu.ny[n], TY), divUp(NZ2, TZ));

		// Call the kernel
		hpFilterKernel_mgpu<<<gridSize, blockSize>>>(gpu.start_y[n], grid.kx[n], vel.uh[n]);
		hpFilterKernel_mgpu<<<gridSize, blockSize>>>(gpu.start_y[n], grid.kx[n], vel.vh[n]);
		hpFilterKernel_mgpu<<<gridSize, blockSize>>>(gpu.start_y[n], grid.kx[n], vel.wh[n]);
	}
	
	// Transform filtered noise back to physical space
	inverseTransform(fft, gpu, vel.uh);
	inverseTransform(fft, gpu, vel.vh);
	inverseTransform(fft, gpu, vel.wh);

	return;
}

__global__ 
void initializeVelocityKernel_mgpu(int start_x, cufftDoubleReal *f1, cufftDoubleReal *f2, cufftDoubleReal *f3)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int j = blockIdx.y * blockDim.y + threadIdx.y;
	const int k = blockIdx.z * blockDim.z + threadIdx.z;
	if (( (i+start_x) >= NX) || (j >= NY) || (k >= NZ)) return;
	const int idx = flatten( i, j, k, NX, NY, 2*NZ2);	// Index local to the GPU

	// Create physical vectors in temporary memory
	// For domain centered at (0,0,0):
	// double x = -(double)LX/2 + (i + start_x)*(double)LX/NX ;
	double y = -(double)LY/2 + j*(double)LY/NY;
	// double z = -(double)LZ/2 + k*(double)LZ/NZ;

/*
	// Initialize velocity array - adding shear profile onto isotropic velocity field
	f1[idx] = 0.5 - 0.5*tanh( H/(4.0*theta)*( 2.0*fabs(y)/H - 1.0 )) + 0.02*f1[idx];
	f2[idx] = 0.02*f2[idx];
	f3[idx] = 0.02*f3[idx];
*/
	// Initialize smooth jet velocity profile
	f1[idx] = 0.5 - 0.5*tanh( H/(4.0*theta)*( 2.0*fabs(y)/H - 1.0 ));
	f2[idx] = 0.0;
	f3[idx] = 0.0;


	return;
}

void initializeVelocity(gpudata gpu, fielddata vel)
{
	int n;
	for (n = 0; n<gpu.nGPUs; ++n){
		cudaSetDevice(n);

		const dim3 blockSize(TX, TY, TZ);
		const dim3 gridSize(divUp(gpu.nx[n], TX), divUp(NY, TY), divUp(NZ, TZ));

		initializeVelocityKernel_mgpu<<<gridSize, blockSize>>>(gpu.start_x[n], vel.u[n], vel.v[n], vel.w[n]);
		printf("Velocity initialized on GPU #%d...\n",n);
	}

	return;
}

__global__ 
void initializeScalarKernel_mgpu(int start_x, cufftDoubleReal *Z)
{	// Creates initial conditions in the physical domain
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int j = blockIdx.y * blockDim.y + threadIdx.y;
	const int k = blockIdx.z * blockDim.z + threadIdx.z;
	if (((i+start_x) >= NX) || (j >= NY) || (k >= NZ)) return;
	const int idx = flatten(i, j, k, NX, NY, 2*NZ2);	// Index local to each GPU

	// For mixing layer following daSilva and Pereira, 2007 PoF:
	// Create physical vectors in temporary memory
	// double x = -(double)LX/2 + (i + start_x)*(double)LX/NX ;
	double y = -(double)LY/2 + j*(double)LY/NY;
	// double z = -(double)LZ/2 + k*(double)NZ/NZ;

	// Initialize scalar field
	Z[idx] = 0.5 - 0.5*tanh( H/(4.0*theta_s)*( 2.0*fabs(y)/H - 1.0 ));
/*
	// For mixing layer used in Blakeley et al., 2019 JoT
  // Create physical vectors in temporary memory
	double x = (i + start_x) * (double)LX / NX;

	// Initialize starting array
	if ( (i+start_x) < NX/2 ){
		Z[idx] = 0.5 * (1 + tanh( (x - PI/2) * LX) );
	}
	else {
		Z[idx] = 0.5 * (1 - tanh( (x - 3*PI/2) * LX) );
	}
*/

	return;
}

void initializeScalar(gpudata gpu, fielddata vel)
{
	int n;
	for (n = 0; n<gpu.nGPUs; ++n){
		cudaSetDevice(n);

		const dim3 blockSize(TX, TY, TZ);
		const dim3 gridSize(divUp(gpu.nx[n], TX), divUp(NY, TY), divUp(NZ, TZ));

		initializeScalarKernel_mgpu<<<gridSize, blockSize>>>(gpu.start_x[n], vel.s[n]);
		initializeScalarKernel_mgpu<<<gridSize, blockSize>>>(gpu.start_x[n], vel.c[n]);
		printf("Scalar field initialized on GPU #%d...\n",n);
	}

	return;

}

__global__
void velocitySuperpositionKernel_mgpu(int start_x, cufftDoubleReal *u, cufftDoubleReal *v, cufftDoubleReal *w, cufftDoubleReal *noise_u, cufftDoubleReal *noise_v, cufftDoubleReal *noise_w, double scale )
{ // This function is designed to add a 3D isotropic turbulent velocity background perturbation
	// onto the shear layer region of a temporal jet.  
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int j = blockIdx.y * blockDim.y + threadIdx.y;
	const int k = blockIdx.z * blockDim.z + threadIdx.z;
	if (((i+start_x) >= NX) || (j >= NY) || (k >= NZ)) return;
	const int idx = flatten(i, j, k, NX, NY, 2*NZ2);	// Index local to each GPU

	u[idx] = u[idx] + scale*noise_u[idx];
	v[idx] = v[idx] + scale*noise_v[idx];
	w[idx] = w[idx] + scale*noise_w[idx];

	return;
}

// Adding isotropic velocity field only in shear layer of temporal jet
void initializeJet_Superposition(fftdata fft, gpudata gpu, griddata grid, fielddata h_vel, fielddata vel, fielddata rhs)
{
	int n;

	// Import isotropic velocity field
	importData(gpu, h_vel, rhs);

	// High-pass filter to remove lowest wavenumbers
	hpFilter(gpu, fft, grid, rhs);

	// Initialize smooth jet velocity field (hyperbolic tangent profile from da Silva and Pereira)
	initializeVelocity(gpu, vel);

	// Superimpose isotropic noise on top of jet velocity initialization
	for (n = 0; n<gpu.nGPUs; ++n){
		cudaSetDevice(n);

		const dim3 blockSize(TX, TY, TZ);
		const dim3 gridSize(divUp(gpu.nx[n], TX), divUp(NY, TY), divUp(NZ, TZ));

		velocitySuperpositionKernel_mgpu<<<gridSize, blockSize>>>(gpu.start_x[n], vel.u[n], vel.v[n], vel.w[n], rhs.u[n], rhs.v[n], rhs.w[n], 0.02);
		printf("Superimposing Jet velocity profile with isotropic noise...\n");
	}	

	initializeScalar(gpu, vel);

	synchronizeGPUs(gpu.nGPUs);

	return;
}



__global__
void scaleDataKernel_mgpu(int start_x, cufftDoubleReal *u, cufftDoubleReal *v, cufftDoubleReal *w, double val)
{ // This function is designed to add a 3D isotropic turbulent velocity background perturbation
	// onto the shear layer region of a temporal jet.  
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int j = blockIdx.y * blockDim.y + threadIdx.y;
	const int k = blockIdx.z * blockDim.z + threadIdx.z;
	if (((i+start_x) >= NX) || (j >= NY) || (k >= NZ)) return;
	const int idx = flatten(i, j, k, NX, NY, 2*NZ2);	// Index local to each GPU

	u[idx] = val*u[idx];
	v[idx] = val*v[idx];
	w[idx] = val*w[idx];

	return;
}

void scaleData(gpudata gpu, fielddata vel, double val)
{	// Subroutine to scale the velocity field prior to convolution

	int n;

	for (n = 0; n<gpu.nGPUs; ++n){
		cudaSetDevice(n);

		const dim3 blockSize(TX, TY, TZ);
		const dim3 gridSize(divUp(gpu.nx[n], TX), divUp(NY, TY), divUp(NZ, TZ));

		scaleDataKernel_mgpu<<<gridSize, blockSize>>>(gpu.start_x[n], vel.u[n], vel.v[n], vel.w[n], val);
		printf("Scaling isotropic velocity...\n");
	}	

	return;
}

__global__
void waveNumber_kernel(double *waveNum)
{   // Creates the wavenumber vectors used in Fourier space
	const int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= NX) return;

	if (i < NX/2)
		waveNum[i] = (2*PI/LX)*(double)i;
	else
		waveNum[i] = (2*PI/LX)*( (double)i - NX );

	return;
}

void initializeWaveNumbers(gpudata gpu, griddata grid)
{    // Initialize wavenumbers in Fourier space

	int n;
	for (n = 0; n<gpu.nGPUs; ++n){
		cudaSetDevice(n);

		waveNumber_kernel<<<divUp(NX,TX), TX>>>(grid.kx[n]);
	}

	printf("Wave domain setup complete..\n");

	return;
}
