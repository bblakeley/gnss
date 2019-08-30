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

void hpFilter(gpuinfo gpu, fftinfo fft, double **k, fielddata vel)
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
		hpFilterKernel_mgpu<<<gridSize, blockSize>>>(gpu.start_y[n], k[n], vel.uh[n]);
		hpFilterKernel_mgpu<<<gridSize, blockSize>>>(gpu.start_y[n], k[n], vel.vh[n]);
		hpFilterKernel_mgpu<<<gridSize, blockSize>>>(gpu.start_y[n], k[n], vel.wh[n]);
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

/*	// For domain centered at (pi,pi,pi)
	double x = (i + start_x) * (double)LX / NX;
	double y = j * (double)LY / NY;
	double z = k * (double)LZ / NZ;*/

/*	// Initialize starting array - Taylor Green Vortex
	f1[idx] = sin(x)*cos(y)*cos(z);
	f2[idx] = -cos(x)*sin(y)*cos(z);
	f3[idx] = 0.0;*/
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


void initializeVelocity(gpuinfo gpu, fielddata vel)
{
	int n;
	for (n = 0; n<gpu.nGPUs; ++n){
		cudaSetDevice(n);

		const dim3 blockSize(TX, TY, TZ);
		const dim3 gridSize(divUp(gpu.nx[n], TX), divUp(NY, TY), divUp(NZ, TZ));

		initializeVelocityKernel_mgpu<<<gridSize, blockSize>>>(gpu.start_x[n], vel.u[n], vel.v[n], vel.w[n]);
		printf("Data initialized on GPU #%d...\n",n);
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
	Z[idx] = 0.5 - 0.5*tanh( H/(4.0*theta)*( 2.0*fabs(y)/H - 1.0 ));

/*	// For mixing layer used in Blakeley et al., 2019 JoT
  // Create physical vectors in temporary memory
	double x = (i + start_x) * (double)LX / NX;

	// Initialize starting array
	if ( (i+start_x) < NX/2 ){
		Z[idx] = 0.5 * (1 + tanh( (x - PI/2) * LX) );
	}
	else {
		Z[idx] = 0.5 * (1 - tanh( (x - 3*PI/2) * LX) );
	}*/


	return;
}

void initializeScalar(gpuinfo gpu, fielddata vel)
{
	int n;
	for (n = 0; n<gpu.nGPUs; ++n){
		cudaSetDevice(n);

		const dim3 blockSize(TX, TY, TZ);
		const dim3 gridSize(divUp(gpu.nx[n], TX), divUp(NY, TY), divUp(NZ, TZ));

		initializeScalarKernel_mgpu<<<gridSize, blockSize>>>(gpu.start_x[n], vel.s[n]);
		printf("Data initialized on GPU #%d...\n",n);
	}

	return;

}

void initializeData(gpuinfo gpu, fftinfo fft, fielddata vel)
{ // Initialize DNS data

	initializeVelocity(gpu, vel);

	initializeScalar(gpu, vel);

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
void initializeJet_Superposition(fftinfo fft, gpuinfo gpu, double **wave, fielddata h_vel, fielddata vel, fielddata rhs)
{
	int n;

	// Import isotropic velocity field
	importData(gpu, h_vel, rhs);

	// High-pass filter to remove lowest wavenumbers
	hpFilter(gpu, fft, wave, rhs);

	// Initialize smooth jet velocity field (hyperbolic tangent profile from da Silva and Pereira)
	initializeVelocity(gpu, vel);

	// Superimpose isotropic noise on top of jet velocity initialization
	for (n = 0; n<gpu.nGPUs; ++n){
		cudaSetDevice(n);

		const dim3 blockSize(TX, TY, TZ);
		const dim3 gridSize(divUp(gpu.nx[n], TX), divUp(NY, TY), divUp(NZ, TZ));

		velocitySuperpositionKernel_mgpu<<<gridSize, blockSize>>>(gpu.start_x[n], vel.u[n], vel.v[n], vel.w[n], rhs.u[n], rhs.v[n], rhs.w[n], 0.002);
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

void scaleData(gpuinfo gpu, fielddata vel, double val)
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
void velocityConvolutionKernel_mgpu(int start_y, cufftDoubleComplex *u, cufftDoubleComplex *v, cufftDoubleComplex *w, cufftDoubleComplex *noise_u, cufftDoubleComplex *noise_v, cufftDoubleComplex *noise_w )
{ // This function is designed to add a 3D isotropic turbulent velocity background perturbation
	// onto the shear layer region of a temporal jet.  
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int j = blockIdx.y * blockDim.y + threadIdx.y;
	const int k = blockIdx.z * blockDim.z + threadIdx.z;
	if (( i >= NX) || ((j+start_y) >= NY) || (k >= NZ2)) return;
	const int idx = flatten( j, i, k, NY, NX, NZ2);

	double a,b,c,d;
	// Multiplication in complex space: (a+bi)*(c+di) = ac - bd + (ad + bc)i
	
	// X-component of velocity
	a = u[idx].x;
	b = u[idx].y;
	c = noise_u[idx].x;
	d = noise_u[idx].y;

	u[idx].x = (a*c - b*d);		// Real component
	u[idx].y = (a*d + b*c);		// Imaginary component

	a=0.0; b=0.0; c=0.0; d=0.0;  // Clear a,b,c,d

	// Y-component of velocity
	a = v[idx].x;
	b = v[idx].y;
	c = noise_v[idx].x;
	d = noise_v[idx].y;	

	v[idx].x = a*c - b*d;		// Real component
	v[idx].y = a*d + b*c;		// Imaginary component

	a=0.0; b=0.0; c=0.0; d=0.0;  // Clear a,b,c,d

	// W-component of velocity
	a = w[idx].x;
	b = w[idx].y;
	c = noise_w[idx].x;
	d = noise_w[idx].y;

	w[idx].x = a*c - b*d;		// Real component
	w[idx].y = a*d + b*c;		// Imaginary component

	// // Zero out rhs
	// noise_u[idx].x = 0.0;
	// noise_u[idx].y = 0.0;
	// noise_v[idx].x = 0.0;
	// noise_v[idx].y = 0.0;
	// noise_w[idx].x = 0.0;
	// noise_w[idx].y = 0.0;

	return;
}

void velocityConvolution_mgpu(gpuinfo gpu, fielddata vel, fielddata rhs)
{
	int n;

	// Launch kernel on GPUs
	for(n=0; n<gpu.nGPUs; ++n){
		cudaSetDevice(n);

		// Set thread and block dimensions for kernal calls
		const dim3 blockSize(TX, TY, TZ);
		const dim3 gridSize(divUp(NX, TX), divUp(gpu.ny[n], TY), divUp(NZ2, TZ));

		// Multiply jet velocity and isotropic noise in Fourier space
		velocityConvolutionKernel_mgpu<<<gridSize, blockSize>>>(gpu.start_y[n], vel.uh[n], vel.vh[n], vel.wh[n], rhs.uh[n], rhs.vh[n], rhs.wh[n]);
	}

	return;
}

// Adding isotropic velocity field only in shear layer of temporal jet
void initializeJet_Convolution(fftinfo fft, gpuinfo gpu, fielddata h_vel, fielddata vel, fielddata rhs)
{
	// Import isotropic velocity field
	importData(gpu, h_vel, vel);

	// Scale initial condition to lower value for background noise
	scaleData(gpu, vel, 0.0002);

	// Initialize smooth jet velocity field (hyperbolic tangent profile)
	initializeVelocity(gpu, rhs);

	// Transform Jet profile (stored in rhs_u) to Fourier Space
	forwardTransform(fft, gpu, vel.u);
	forwardTransform(fft, gpu, vel.v);
	forwardTransform(fft, gpu, vel.w);
	forwardTransform(fft, gpu, rhs.u);
	forwardTransform(fft, gpu, rhs.v);
	forwardTransform(fft, gpu, rhs.w);

	// Convolve initial jet velocity with isotropic noise
	velocityConvolution_mgpu(gpu, vel, rhs);
	printf("Convolving Jet velocity profile with isotropic noise...\n");

	inverseTransform(fft, gpu, vel.uh);
	inverseTransform(fft, gpu, vel.vh);
	inverseTransform(fft, gpu, vel.wh);

	initializeScalar(gpu, vel);

	synchronizeGPUs(gpu.nGPUs);

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

void initializeWaveNumbers(gpuinfo gpu, double **waveNum)
{    // Initialize wavenumbers in Fourier space

	int n;
	for (n = 0; n<gpu.nGPUs; ++n){
		cudaSetDevice(n);

		waveNumber_kernel<<<divUp(NX,TX), TX>>>(waveNum[n]);
	}

	printf("Wave domain setup complete..\n");

	return;
}
