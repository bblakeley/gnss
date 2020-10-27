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
void deAliasKernel_mgpu(int start_y, double *k1, double *k2, double *k3, cufftDoubleComplex *fhat){

	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int j = blockIdx.y * blockDim.y + threadIdx.y;
	const int k = blockIdx.z * blockDim.z + threadIdx.z;
	const int jj = j + start_y;  // Absolute index for referencing wavenumbers
	if (( i >= NX) || (jj >= NY) || (k >= NZ2)) return;
	const int idx = flatten( j, i, k, NY, NX, NZ2);

	double k_sq = k1[i]*k1[i] + k2[jj]*k2[jj] + k3[k]*k3[k];
	
	double kx_max = NX*PI/LX;
	
	double k_max = alias_filter*kx_max;
	//double ky_max = alias_filter*PI*NY/LY;
	//double kz_max = alias_filter*PI*NZ/LZ;
	//double k_max = kx_max*kx_max; // + ky_max*ky_max + kz_max*kz_max;

	if( k_sq > k_max*k_max ){
		fhat[idx].x = 0.0;
		fhat[idx].y = 0.0;
	}

	return;
}

void deAlias(gpudata gpu, griddata grid, fielddata vel)
{	// Truncate data for de-aliasing

	int n;
	for(n = 0; n<gpu.nGPUs; ++n){
		cudaSetDevice(n);

		// Set thread and block dimensions for kernal calls
		const dim3 blockSize(TX, TY, TZ);
		const dim3 gridSize(divUp(NX, TX), divUp(gpu.ny[n], TY), divUp(NZ2, TZ));

		// Call the kernel

		deAliasKernel_mgpu<<<gridSize, blockSize>>>(gpu.start_y[n], grid.kx[n], grid.ky[n], grid.kz[n], vel.uh[n]);
		deAliasKernel_mgpu<<<gridSize, blockSize>>>(gpu.start_y[n], grid.kx[n], grid.ky[n], grid.kz[n], vel.vh[n]);
		deAliasKernel_mgpu<<<gridSize, blockSize>>>(gpu.start_y[n], grid.kx[n], grid.ky[n], grid.kz[n], vel.wh[n]);
		deAliasKernel_mgpu<<<gridSize, blockSize>>>(gpu.start_y[n], grid.kx[n], grid.ky[n], grid.kz[n], vel.sh[n]);
		deAliasKernel_mgpu<<<gridSize, blockSize>>>(gpu.start_y[n], grid.kx[n], grid.ky[n], grid.kz[n], vel.ch[n]);

	}
	
	return;
}

__global__
void calcOmega1Kernel_mgpu(int start_y, double *k2, double *k3, cufftDoubleComplex *u2hat, cufftDoubleComplex *u3hat, cufftDoubleComplex *omega1){

	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int j = blockIdx.y * blockDim.y + threadIdx.y;
	const int k = blockIdx.z * blockDim.z + threadIdx.z;
	const int jj = j + start_y;  // Absolute index for referencing wavenumbers
	if ((i >= NX) || (jj >= NY) || (k >= NZ2)) return;
	const int idx = flatten( j, i, k, NY, NX, NZ2);

	omega1[idx].x = -k2[jj]*u3hat[idx].y + k3[k]*u2hat[idx].y;
	omega1[idx].y = k2[jj]*u3hat[idx].x - k3[k]*u2hat[idx].x;

	return;
}

__global__
void calcOmega2Kernel_mgpu(int start_y, double *k1, double *k3, cufftDoubleComplex *u1hat, cufftDoubleComplex *u3hat, cufftDoubleComplex *omega2){

	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int j = blockIdx.y * blockDim.y + threadIdx.y;
	const int k = blockIdx.z * blockDim.z + threadIdx.z;
	const int jj = j + start_y;  // Absolute index for referencing wavenumbers
	if ((i >= NX) || (jj >= NY) || (k >= NZ2)) return;
	const int idx = flatten( j, i, k, NY, NX, NZ2);

	omega2[idx].x = k1[i]*u3hat[idx].y - k3[k]*u1hat[idx].y;
	omega2[idx].y = -k1[i]*u3hat[idx].x + k3[k]*u1hat[idx].x;

	return;
}

__global__
void calcOmega3Kernel_mgpu(int start_y, double *k1, double *k2, cufftDoubleComplex *u1hat, cufftDoubleComplex *u2hat, cufftDoubleComplex *omega3){

	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int j = blockIdx.y * blockDim.y + threadIdx.y;
	const int k = blockIdx.z * blockDim.z + threadIdx.z;
	const int jj = j + start_y;  // Absolute index for referencing wavenumbers
	if ((i >= NX) || (jj >= NY) || (k >= NZ2)) return;
	const int idx = flatten( j, i, k, NY, NX, NZ2);

	omega3[idx].x = -k1[i]*u2hat[idx].y + k2[jj]*u1hat[idx].y;
	omega3[idx].y = k1[i]*u2hat[idx].x - k2[jj]*u1hat[idx].x;

	return;
}

void vorticity(gpudata gpu, griddata grid, fielddata vel, fielddata rhs){
	// Function to calculate the vorticity in Fourier Space and transform to physical space
	
	int n;
	for(n=0; n<gpu.nGPUs; ++n){
		cudaSetDevice(n);

		// Set thread and block dimensions for kernal calls
		const dim3 blockSize(TX, TY, TZ);
		const dim3 gridSize(divUp(NX, TX), divUp(gpu.ny[n], TY), divUp(NZ2, TZ));

		// Call kernels to calculate vorticity
		calcOmega1Kernel_mgpu<<<gridSize, blockSize>>>(gpu.start_y[n],  grid.ky[n], grid.kz[n], vel.vh[n], vel.wh[n], rhs.uh[n]);
		calcOmega2Kernel_mgpu<<<gridSize, blockSize>>>(gpu.start_y[n],  grid.kx[n], grid.kz[n], vel.uh[n], vel.wh[n], rhs.vh[n]);
		calcOmega3Kernel_mgpu<<<gridSize, blockSize>>>(gpu.start_y[n],  grid.kx[n], grid.ky[n], vel.uh[n], vel.vh[n], rhs.wh[n]);
		// Kernel calls include scaling for post-FFT
	}

	// printf("Vorticity calculated in fourier space...\n");

	return;
}

__global__
void crossProductKernel_mgpu(int start_x, cufftDoubleReal *u1, cufftDoubleReal *u2, cufftDoubleReal *u3, cufftDoubleReal *omega1, cufftDoubleReal *omega2, cufftDoubleReal *omega3){

	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int j = blockIdx.y * blockDim.y + threadIdx.y;
	const int k = blockIdx.z * blockDim.z + threadIdx.z;
	const int ii = i + start_x;  // Absolute index for referencing wavenumbers
	if ((ii >= NX) || (j >= NY) || (k >= NZ)) return;
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

void crossProduct(gpudata gpu, fielddata vel, fielddata rhs){
// Function to evaluate omega x u in physical space and then transform the result to Fourier Space

	int n;
	for (n = 0; n<gpu.nGPUs; ++n){
		cudaSetDevice(n);

		const dim3 blockSize(TX, TY, TZ);
		const dim3 gridSize(divUp(gpu.nx[n], TX), divUp(NY, TY), divUp(NZ, TZ));

		// Call kernel to calculate vorticity
		crossProductKernel_mgpu<<<gridSize, blockSize>>>(gpu.start_x[n], vel.u[n], vel.v[n], vel.w[n], rhs.u[n], rhs.v[n], rhs.w[n]);

		cudaDeviceSynchronize();
	}

	// printf("Cross Product calculated!\n");

	return;
}

__global__
void multIkKernel_mgpu(const int dir, int start_y, double *k1, double *k2, double *k3, cufftDoubleComplex *f, cufftDoubleComplex *fIk)
{   // Multiples an input array by ik 
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int j = blockIdx.y * blockDim.y + threadIdx.y;
	const int k = blockIdx.z * blockDim.z + threadIdx.z;
	const int jj = j + start_y;  // Absolute index for referencing wavenumbers
	if (( i >= NX) || (jj >= NY) || (k >= NZ2)) return;
	const int idx = flatten( j, i, k, NY, NX, NZ2);

	if(dir == 1){
		fIk[idx].x = -k1[i]*f[idx].y;
		fIk[idx].y = k1[i]*f[idx].x;
	}

	if(dir == 2){
		fIk[idx].x = -k2[jj]*f[idx].y;
		fIk[idx].y = k2[jj]*f[idx].x;
	}

	if(dir == 3){
		fIk[idx].x = -k3[k]*f[idx].y;
		fIk[idx].y = k3[k]*f[idx].x;
	}

	return;
}

void takeDerivative(int dir, gpudata gpu, griddata grid, cufftDoubleComplex **f, cufftDoubleComplex **fIk)
{
	// Loop through GPUs and multiply by iK
	// Note: Data assumed to be transposed during 3D FFt process; k + NZ*i + NZ*NX*j
	int n;
	for(n = 0; n<gpu.nGPUs; ++n){
		cudaSetDevice(n);
		
		const dim3 blockSize(TX, TY, TZ);
		const dim3 gridSize(divUp(NX, TX), divUp(gpu.ny[n], TY), divUp(NZ2, TZ));

		// Take derivative (dir = 1 => x-direction, 2 => y-direction, 3 => z-direction)
		multIkKernel_mgpu<<<gridSize, blockSize>>>(dir, gpu.start_y[n], grid.kx[n], grid.ky[n], grid.kz[n], f[n], fIk[n]);
	}

	return;  
}

__global__
void gradientKernel_mgpu(int start_y, double *k1, double *k2, double *k3, cufftDoubleComplex *f, cufftDoubleComplex *f_x, cufftDoubleComplex *f_y, cufftDoubleComplex *f_z)
{   // Multiples an input array by ik 
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int j = blockIdx.y * blockDim.y + threadIdx.y;
	const int k = blockIdx.z * blockDim.z + threadIdx.z;
	const int jj = j + start_y;  // Absolute index for referencing wavenumbers
	if (( i >= NX) || (jj >= NY) || (k >= NZ2)) return;
	const int idx = flatten( j, i, k, NY, NX, NZ2);

  // x-direction
	f_x[idx].x = -k1[i]*f[idx].y;
	f_x[idx].y = k1[i]*f[idx].x;

  // y-direction
	f_y[idx].x = -k2[jj]*f[idx].y; 
	f_y[idx].y = k2[jj]*f[idx].x;

  // z-direction
	f_z[idx].x = -k3[k]*f[idx].y; 
	f_z[idx].y = k3[k]*f[idx].x;

	return;
}

void gradient(gpudata gpu, griddata grid, cufftDoubleComplex **f, fielddata grad)
{
	// Loop through GPUs and multiply by ik in each direction
	// Note: Data assumed to be transposed during 3D FFt process; k + NZ*i + NZ*NX*j
	int n;
	for(n = 0; n<gpu.nGPUs; ++n){
		cudaSetDevice(n);
		const dim3 blockSize(TX, TY, TZ);
		const dim3 gridSize(divUp(NX, TX), divUp(gpu.ny[n], TY), divUp(NZ2, TZ));

		// Take gradient
		gradientKernel_mgpu<<<gridSize, blockSize>>>(gpu.start_y[n], grid.kx[n], grid.ky[n], grid.kz[n], f[n], grad.uh[n], grad.vh[n], grad.wh[n]);
	}

	return;  
}

__global__
void divergenceKernel_mgpu(int start_y, double *k1, double *k2, double *k3, cufftDoubleComplex *f_x, cufftDoubleComplex *f_y, cufftDoubleComplex *f_z, cufftDoubleComplex *result)
{   // Multiples an input array by ik 
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int j = blockIdx.y * blockDim.y + threadIdx.y;
	const int k = blockIdx.z * blockDim.z + threadIdx.z;
	const int jj = j + start_y;  // Absolute index for referencing wavenumbers
	if (( i >= NX) || (jj >= NY) || (k >= NZ2)) return;
	const int idx = flatten( j, i, k, NY, NX, NZ2);

  // i*kx*f_x + i*ky*f_y + i*kz*f_z
	result[idx].x = -k1[i]*f_x[idx].y - k2[jj]*f_y[idx].y - k3[k]*f_z[idx].y; 
	result[idx].y = k1[i]*f_x[idx].x + k2[jj]*f_y[idx].x + k3[k]*f_z[idx].x ;

	return;
}

void divergence(gpudata gpu, griddata grid, fielddata f, cufftDoubleComplex **result)
{
	// Loop through GPUs and multiply by ik in each direction
	// Note: Data assumed to be transposed during 3D FFt process; k + NZ*i + NZ*NX*j
	int n;
	for(n = 0; n<gpu.nGPUs; ++n){
		cudaSetDevice(n);
		const dim3 blockSize(TX, TY, TZ);
		const dim3 gridSize(divUp(NX, TX), divUp(gpu.ny[n], TY), divUp(NZ2, TZ));

		// Take divergence
		divergenceKernel_mgpu<<<gridSize, blockSize>>>(gpu.start_y[n], grid.kx[n], grid.ky[n], grid.kz[n], f.uh[n], f.vh[n], f.wh[n], result[n]);
	}

	return;  
}

__global__
void multAndAddKernel_mgpu(int start_x, cufftDoubleReal *f1, cufftDoubleReal *f2, cufftDoubleReal *f3)
{	// Function to compute the non-linear terms on the RHS

	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int j = blockIdx.y * blockDim.y + threadIdx.y;
	const int k = blockIdx.z * blockDim.z + threadIdx.z;
	const int ii = i + start_x;  // Absolute index for referencing wavenumbers
	if ((ii >= NX) || (j >= NY) || (k >= NZ)) return;
	const int idx = flatten(i, j, k, NX, NY, 2*NZ2);

	f3[idx] = f3[idx] + f1[idx] * f2[idx];
		
	return;
}

void multAndAdd( gpudata gpu, cufftDoubleReal **f1, cufftDoubleReal **f2, cufftDoubleReal **f3)
{
	// Loop through GPUs and perform operation: f1*f2 + f3 = f3
	// Note: Data assumed to be transposed during 3D FFt process; k + NZ*i + NZ*NX*j
	int n;
	for(n = 0; n<gpu.nGPUs; ++n){
		cudaSetDevice(n);
		
		const dim3 blockSize(TX, TY, TZ);
		const dim3 gridSize(divUp(gpu.nx[n], TX), divUp(NY, TY), divUp(NZ, TZ));

		multAndAddKernel_mgpu<<<gridSize, blockSize>>>(gpu.start_x[n], f1[n], f2[n], f3[n]);
	}

	return;  
}

__global__
void dotKernel_mgpu(int start_x, cufftDoubleReal *a1, cufftDoubleReal *a2, cufftDoubleReal *a3, cufftDoubleReal *b1, cufftDoubleReal *b2, cufftDoubleReal *b3, cufftDoubleReal *result)
{	// Function to compute the non-linear terms on the RHS

	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int j = blockIdx.y * blockDim.y + threadIdx.y;
	const int k = blockIdx.z * blockDim.z + threadIdx.z;
	const int ii = i + start_x;  // Absolute index for referencing wavenumbers
	if ((ii >= NX) || (j >= NY) || (k >= NZ)) return;
	const int idx = flatten(i, j, k, NX, NY, 2*NZ2);

  result[idx] = a1[idx]*b1[idx] + a2[idx]*b2[idx] + a3[idx]*b3[idx];

	return;
}

void dotProduct( gpudata gpu, fielddata a, fielddata b, cufftDoubleReal **result)
{
	// Loop through GPUs and perform dot product, result = a.u*b.u + a.v*b.v + a.w*b.w
	// Note: Data assumed to be transposed during 3D FFt process; k + NZ*i + NZ*NX*j
	int n;
	for(n = 0; n<gpu.nGPUs; ++n){
		cudaSetDevice(n);
		const dim3 blockSize(TX, TY, TZ);
		const dim3 gridSize(divUp(gpu.nx[n], TX), divUp(NY, TY), divUp(NZ, TZ));
		// Take derivative (dir = 1 => x-direction, 2 => y-direction, 3 => z-direction)
		dotKernel_mgpu<<<gridSize, blockSize>>>(gpu.start_x[n], a.u[n], a.v[n], a.w[n], b.u[n], b.v[n], b.w[n], result[n]);
	}

	return;  
}

__global__
void colloidAdvectionKernel_mgpu(int start_x, cufftDoubleReal *u, cufftDoubleReal *v, cufftDoubleReal *w, cufftDoubleReal *c, cufftDoubleReal *c_x, cufftDoubleReal *c_y, cufftDoubleReal *c_z, cufftDoubleReal *udp, cufftDoubleReal *vdp, cufftDoubleReal *wdp, cufftDoubleReal *divVdp, double a)
{	// Function to compute the non-linear terms on the RHS

	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int j = blockIdx.y * blockDim.y + threadIdx.y;
	const int k = blockIdx.z * blockDim.z + threadIdx.z;
	if (((i + start_x) >= NX) || (j >= NY) || (k >= NZ)) return;
	const int idx = flatten(i, j, k, NX, NY, 2*NZ2);

	divVdp[idx] = a*( divVdp[idx]*c[idx] + c_x[idx]*udp[idx] + c_y[idx]*vdp[idx] + c_z[idx]*wdp[idx] ) 
	            + u[idx]*c_x[idx] + v[idx]*c_y[idx] + w[idx]*c_z[idx];
	            
	return;
}

void colloidAdvection( gpudata gpu, fielddata vel, fielddata gradC, fielddata Vdp, cufftDoubleReal **divVdp)
{
	// Loop through GPUs and calculate advection terms in colloid transport equation
	// Note: Data assumed to be transposed during 3D FFt process; k + NZ*i + NZ*NX*j
	int n;
	for(n = 0; n<gpu.nGPUs; ++n){
		cudaSetDevice(n);
		
		const dim3 blockSize(TX, TY, TZ);
		const dim3 gridSize(divUp(gpu.nx[n], TX), divUp(NY, TY), divUp(NZ, TZ));

		colloidAdvectionKernel_mgpu<<<gridSize, blockSize>>>(gpu.start_x[n], vel.u[n], vel.v[n], vel.w[n], vel.c[n], gradC.u[n], gradC.v[n], gradC.w[n], Vdp.u[n], Vdp.v[n], Vdp.w[n], divVdp[n], alpha);
	}

	return;  
}

void scalarAdvection(fftdata fft, gpudata gpu, griddata grid, fielddata vel, fielddata rhs, fielddata temp)
{	// Compute the advection term in the scalar equation

	// Zero out right hand side term before beginning calculation
	int n;
	for(n = 0; n<gpu.nGPUs; ++n){
		cudaSetDevice(n);
		checkCudaErrors( cudaMemset(rhs.s[n], 0.0, sizeof(cufftDoubleComplex)*NZ2*NX*gpu.ny[n]) );
		checkCudaErrors( cudaMemset(rhs.c[n], 0.0, sizeof(cufftDoubleComplex)*NZ2*NX*gpu.ny[n]) );
	}

  // Form advection term for passive scalar field
	//===============================================================
	// ( u \dot grad ) z = u * dZ/dx + v * dZ/dy + w * dZ/dz
	//===============================================================

  gradient(gpu, grid, vel.sh, rhs);  // Calculate gradient of passive scalar, store result in rhs.u,v,w
  // Calculate divergence of gradient of passive scalar (used in colloid eqn)
  divergence(gpu, grid, rhs, rhs.ch);    // Calculate divergence of rhs, stores result in rhs.ch
  
  // Transform derivatives from Fourier space to physical space
	inverseTransform(fft, gpu, rhs.uh);
	inverseTransform(fft, gpu, rhs.vh);
	inverseTransform(fft, gpu, rhs.wh);

	dotProduct(gpu, vel, rhs, rhs.s); // Calculates dot product between u,v,w components of vel, rhs; places result in rhs.s

	// rhs.s now holds the advective term of the scalar equation in physical domain. 
	
	//====== Colloid Transport Equation ======
	// 
	//===============================================================
	// div( (u + u_dp)*c ) = u.*grad(c) + u_dp.*grad(c) + div(u_dp)*c + ( div(u)*c == 0 ) 
	// Drift velocity defined as: u_dp = alpha*grad(Z)
	// --> div( (u + u_dp)*c ) = u.*grad(c) + alpha*grad(Z).*grad(c) + alpha*div(grad(Z))*c
	//===============================================================

  gradient(gpu, grid, vel.ch, temp);  // Calculate gradient of colloid
  
  // Transform derivatives to physical domain to form non-linear terms
  inverseTransform(fft, gpu, temp.uh);
	inverseTransform(fft, gpu, temp.vh);
	inverseTransform(fft, gpu, temp.wh);
  inverseTransform(fft, gpu, rhs.ch);
  inverseTransform(fft, gpu, vel.ch);
  
  // grad(Z) aka Vdp stored in rhs.u,v,w; grad(C) stored in temp; div(u_dp) stored in rhs.c ; velocity, scalars stored in vel
  colloidAdvection(gpu, vel, temp, rhs, rhs.c);
  
  //rhs.c now stores advection terms for colloid equation in physical domain
  
	return;
}

__global__
void computeRHSKernel_mgpu(int start_y, double *k1, double *k2, double *k3, cufftDoubleComplex *rhs_u1, cufftDoubleComplex *rhs_u2, cufftDoubleComplex *rhs_u3, cufftDoubleComplex *rhs_Z, cufftDoubleComplex *rhs_C)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int j = blockIdx.y * blockDim.y + threadIdx.y;
	const int k = blockIdx.z * blockDim.z + threadIdx.z;
	const int jj = j + start_y;  // Absolute index for referencing wavenumbers
	if (( i >= NX) || (jj >= NY) || (k >= NZ2)) return;
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
	double k_sq = k1[i]*k1[i] + k2[jj]*k2[jj] + k3[k]*k3[k];

	// Form RHS
	if( i == 0 && jj == 0 && k == 0){
		rhs_u1[idx].x = 0.0;
		rhs_u1[idx].y = 0.0;

		rhs_u2[idx].x = 0.0;
		rhs_u2[idx].y = 0.0;

		rhs_u3[idx].x = 0.0;
		rhs_u3[idx].y = 0.0;

		rhs_Z[idx].x = 0.0;
		rhs_Z[idx].y = 0.0;
		
		rhs_C[idx].x = 0.0;
		rhs_C[idx].y = 0.0;
	}
	else {
		rhs_u1[idx].x = (k1[i]*k1[i] / k_sq - 1.0)*temp1_r + (k1[i]*k2[jj] / k_sq)*temp2_r + (k1[i]*k3[k] / k_sq)*temp3_r;
		rhs_u1[idx].y = (k1[i]*k1[i] / k_sq - 1.0)*temp1_c + (k1[i]*k2[jj] / k_sq)*temp2_c + (k1[i]*k3[k] / k_sq)*temp3_c;

		rhs_u2[idx].x = (k2[jj]*k1[i] / k_sq)*temp1_r + (k2[jj]*k2[jj] / k_sq - 1.0)*temp2_r + (k2[jj]*k3[k] / k_sq)*temp3_r;
		rhs_u2[idx].y = (k2[jj]*k1[i] / k_sq)*temp1_c + (k2[jj]*k2[jj] / k_sq - 1.0)*temp2_c + (k2[jj]*k3[k] / k_sq)*temp3_c;

		rhs_u3[idx].x = (k3[k]*k1[i] / k_sq)*temp1_r + (k3[k]*k2[jj] / k_sq)*temp2_r + (k3[k]*k3[k] / k_sq - 1.0)*temp3_r;
		rhs_u3[idx].y = (k3[k]*k1[i] / k_sq)*temp1_c + (k3[k]*k2[jj] / k_sq)*temp2_c + (k3[k]*k3[k] / k_sq - 1.0)*temp3_c;

		rhs_Z[idx].x = -rhs_Z[idx].x;
		rhs_Z[idx].y = -rhs_Z[idx].y;
		
		rhs_C[idx].x = -rhs_C[idx].x;
		rhs_C[idx].y = -rhs_C[idx].y;
	}

	return;
}

void makeRHS(gpudata gpu, griddata grid, fielddata rhs)
{	// Function to create the rhs of the N-S equations in Fourier Space

	int n;
	for(n=0; n<gpu.nGPUs; ++n){
		cudaSetDevice(n);

		// Set thread and block dimensions for kernal calls
		const dim3 blockSize(TX, TY, TZ);
		const dim3 gridSize(divUp(NX, TX), divUp(gpu.ny[n], TY), divUp(NZ2, TZ));

		// Call the kernel
		computeRHSKernel_mgpu<<<gridSize, blockSize>>>(gpu.start_y[n], grid.kx[n], grid.ky[n], grid.kz[n], rhs.uh[n], rhs.vh[n], rhs.wh[n], rhs.sh[n], rhs.ch[n]);
		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess) 
	    printf("Error: %s\n", cudaGetErrorString(err));	
	}

	// printf("Right hand side of equations formed!\n");

	return;
}


__global__
void eulerKernel_mgpu(double num, int start_y, double *k1, double *k2, double *k3, cufftDoubleComplex *fhat,  cufftDoubleComplex *rhs_f)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int j = blockIdx.y * blockDim.y + threadIdx.y;
	const int k = blockIdx.z * blockDim.z + threadIdx.z;
	const int jj = j + start_y;  // Absolute index for referencing wavenumbers
	if (( i >= NX) || (jj >= NY) || (k >= NZ2)) return;
	const int idx = flatten( j, i, k, NY, NX, NZ2);

	// Calculate k^2 for each index
	double k_sq = k1[i]*k1[i] + k2[jj]*k2[jj] + k3[k]*k3[k];

	// Timestep in X-direction
	fhat[idx].x = ( (1.0 - dt/2.0*k_sq/num)*fhat[idx].x + dt * rhs_f[idx].x ) / (1.0 + dt/2.0*k_sq/num);
	fhat[idx].y = ( (1.0 - dt/2.0*k_sq/num)*fhat[idx].y + dt * rhs_f[idx].y ) / (1.0 + dt/2.0*k_sq/num);

	return;
}

__global__
void adamsBashforthKernel_mgpu(double num, int start_y, double *k1, double *k2, double *k3, cufftDoubleComplex *fhat, cufftDoubleComplex *rhs_f, cufftDoubleComplex *rhs_f_old)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int j = blockIdx.y * blockDim.y + threadIdx.y;
	const int k = blockIdx.z * blockDim.z + threadIdx.z;
	const int jj = j + start_y;  // Absolute index for referencing wavenumbers
	if (( i >= NX) || (jj >= NY) || (k >= NZ2)) return;
	const int idx = flatten( j, i, k, NY, NX, NZ2);

	// Calculate k^2 for each index
	double k_sq = k1[i]*k1[i] + k2[jj]*k2[jj] + k3[k]*k3[k];

	// Timestep in X-direction
	fhat[idx].x = ( (1.0 - dt/2.0*k_sq/num)*fhat[idx].x + dt * (1.5*rhs_f[idx].x - 0.5*rhs_f_old[idx].x) ) / (1.0 + dt/2.0*k_sq/num);
	fhat[idx].y = ( (1.0 - dt/2.0*k_sq/num)*fhat[idx].y + dt * (1.5*rhs_f[idx].y - 0.5*rhs_f_old[idx].y) ) / (1.0 + dt/2.0*k_sq/num);

	return;
}

void timestep(const int flag, gpudata gpu, griddata grid, fielddata vel, fielddata rhs, fielddata rhs_old)
{
	int n;
	for(n=0; n<gpu.nGPUs; ++n){
		cudaSetDevice(n);

		// Set thread and block dimensions for kernal calls
		const dim3 blockSize(TX, TY, TZ);
		const dim3 gridSize(divUp(NX, TX), divUp(gpu.ny[n], TY), divUp(NZ2, TZ));

		if(flag){
			// printf("Using Euler Method\n");
			eulerKernel_mgpu<<<gridSize, blockSize>>>((double) Re,    gpu.start_y[n], grid.kx[n], grid.ky[n], grid.kz[n], vel.uh[n], rhs.uh[n]);
			eulerKernel_mgpu<<<gridSize, blockSize>>>((double) Re,    gpu.start_y[n], grid.kx[n], grid.ky[n], grid.kz[n], vel.vh[n], rhs.vh[n]);
			eulerKernel_mgpu<<<gridSize, blockSize>>>((double) Re,    gpu.start_y[n], grid.kx[n], grid.ky[n], grid.kz[n], vel.wh[n], rhs.wh[n]);
			eulerKernel_mgpu<<<gridSize, blockSize>>>((double) Re*Sc, gpu.start_y[n], grid.kx[n], grid.ky[n], grid.kz[n], vel.sh[n], rhs.sh[n]);
			eulerKernel_mgpu<<<gridSize, blockSize>>>((double) Re*Sc_c, gpu.start_y[n], grid.kx[n], grid.ky[n], grid.kz[n], vel.ch[n], rhs.ch[n]);
		}
		else {
			// printf("Using A-B Method\n");
			adamsBashforthKernel_mgpu<<<gridSize, blockSize>>>((double)Re, gpu.start_y[n], grid.kx[n], grid.ky[n], grid.kz[n], vel.uh[n], rhs.uh[n], rhs_old.uh[n]);
			adamsBashforthKernel_mgpu<<<gridSize, blockSize>>>((double)Re, gpu.start_y[n], grid.kx[n], grid.ky[n], grid.kz[n], vel.vh[n], rhs.vh[n], rhs_old.vh[n]);
			adamsBashforthKernel_mgpu<<<gridSize, blockSize>>>((double)Re, gpu.start_y[n], grid.kx[n], grid.ky[n], grid.kz[n], vel.wh[n], rhs.wh[n], rhs_old.wh[n]);
			adamsBashforthKernel_mgpu<<<gridSize, blockSize>>>((double)Re*Sc, gpu.start_y[n], grid.kx[n], grid.ky[n], grid.kz[n], vel.sh[n], rhs.sh[n], rhs_old.sh[n]);
			adamsBashforthKernel_mgpu<<<gridSize, blockSize>>>((double)Re*Sc_c, gpu.start_y[n], grid.kx[n], grid.ky[n], grid.kz[n], vel.ch[n], rhs.ch[n], rhs_old.ch[n]);
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
	const int jj = j + start_y;  // Absolute index for referencing wavenumbers
	if (( i >= NX) || (jj >= NY) || (k >= NZ2)) return;
	const int idx = flatten( j, i, k, NY, NX, NZ2);

	// Update old variables to store current iteration
	rhs_f_old[idx].x = rhs_f[idx].x;
	rhs_f_old[idx].y = rhs_f[idx].y;

	// Zero out RHS arrays
	rhs_f[idx].x = 0.0;
	rhs_f[idx].y = 0.0;

	return;
}

void update(gpudata gpu, fielddata rhs, fielddata rhs_old)
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
		updateKernel_mgpu<<<gridSize, blockSize>>>(gpu.start_y[n], rhs.ch[n], rhs_old.ch[n]);
	}

	return;
}

__global__
void scalarFilterkernel_mgpu(int start_x, cufftDoubleReal *f)
{	// Function to compute the non-linear terms on the RHS

	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int j = blockIdx.y * blockDim.y + threadIdx.y;
	const int k = blockIdx.z * blockDim.z + threadIdx.z;
	const int ii = i + start_x;  // Absolute index for referencing wavenumbers
	if ((ii >= NX) || (j >= NY) || (k >= NZ)) return;
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

void scalarFilter(gpudata gpu, cufftDoubleReal **f)
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

void solver_ps(const int euler, fftdata fft, gpudata gpu, griddata grid, fielddata vel, fielddata rhs, fielddata rhs_old, fielddata temp)
{	// Pseudo spectral Navier-Stokes solver with conserved, passive scalar field

	// Dealias primitive variables
	deAlias(gpu, grid, vel);

	// Inverse transform the velocity to physical space for scalar advective terms
	inverseTransform(fft, gpu, vel.uh);
	inverseTransform(fft, gpu, vel.vh);
	inverseTransform(fft, gpu, vel.wh);

	// Form advective term in scalar equations
	scalarAdvection(fft, gpu, grid, vel, rhs, temp);
	
	// Transform primitive variables back to fourier space
	forwardTransform(fft, gpu, vel.u);
	forwardTransform(fft, gpu, vel.v);
	forwardTransform(fft, gpu, vel.w);
	forwardTransform(fft, gpu, vel.c);
	
	// Transform the non-linear term in rhs from physical space to Fourier space for timestepping
	forwardTransform(fft, gpu, rhs.s);
	forwardTransform(fft, gpu, rhs.c);
		
	// Form the vorticity in Fourier space
	vorticity(gpu, grid, vel, rhs);

	// Inverse Fourier Transform the vorticity to physical space.
	inverseTransform(fft, gpu, rhs.uh);
	inverseTransform(fft, gpu, rhs.vh);
	inverseTransform(fft, gpu, rhs.wh);

	// printf("Vorticity transformed to physical coordinates...\n");

	// Inverse transform the velocity to physical space for advective terms
	inverseTransform(fft, gpu, vel.uh);
	inverseTransform(fft, gpu, vel.vh);
	inverseTransform(fft, gpu, vel.wh);

	// Form non-linear terms in physical space
	crossProduct(gpu, vel, rhs);

	// Transform omegaXu from physical space to fourier space 
	forwardTransform(fft, gpu, rhs.u);
	forwardTransform(fft, gpu, rhs.v);
	forwardTransform(fft, gpu, rhs.w);
	
	// Transform velocity back to fourier space for timestepping
	forwardTransform(fft, gpu, vel.u);
	forwardTransform(fft, gpu, vel.v);
	forwardTransform(fft, gpu, vel.w);

	// Form right hand side of the N-S and scalar equations
	makeRHS(gpu, grid, rhs);

	// Dealias the solution by truncating RHS
	deAlias(gpu, grid, rhs);

	// Step the vector fields forward in time
	timestep(euler, gpu, grid, vel, rhs, rhs_old);

	// Update loop variables to next timestep
	update(gpu, rhs, rhs_old);

	return;
}
