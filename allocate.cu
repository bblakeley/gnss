#include <stdlib.h>
#include <complex.h>
#include <cufft.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include "dnsparams.h"
#include "struct_def.h"

void allocate_memory(){
	// Allocate memory for statistics structs on both device and host
	int n, nGPUs;

	// // Declare extern variables (to pull def's from declare.h)
	extern gpudata gpu;	
	extern fftdata fft;
	extern statistics *stats;	
	extern profile Yprof;

  extern griddata grid;

  extern fielddata h_vel;
  extern fielddata vel;
  extern fielddata rhs;
  extern fielddata rhs_old;
	extern fielddata temp;

	// Make local copy of number of GPUs (for readability)
	nGPUs = gpu.nGPUs;
	printf("Allocating data on %d GPUs!\n",nGPUs);
	
	// Allocate pinned memory on the host side that stores array of pointers for FFT operations
	cudaHostAlloc((void**)&fft,         		 sizeof(fftdata),   					       cudaHostAllocMapped);		
	cudaHostAlloc((void**)&fft.p1d,    			 nGPUs*sizeof(cufftHandle *), 			 cudaHostAllocMapped);		// Allocate memory for array of cufftHandles to store nGPUs worth 1d plans
	cudaHostAlloc((void**)&fft.p2d,    			 nGPUs*sizeof(cufftHandle *), 			 cudaHostAllocMapped);		// Allocate memory array of 2dplans
	cudaHostAlloc((void**)&fft.invp2d, 			 nGPUs*sizeof(cufftHandle *), 			 cudaHostAllocMapped);		// Array of inverse 2d plans
	cudaHostAlloc((void**)&fft.p3d,          1*sizeof(cufftHandle *),                cudaHostAllocMapped); // Only used when nGPUs=1
	cudaHostAlloc((void**)&fft.invp3d,       1*sizeof(cufftHandle *),                cudaHostAllocMapped); // Only used when nGPUs=1
	cudaHostAlloc((void**)&fft.wsize_f, 		 nGPUs*sizeof(size_t *), 						 cudaHostAllocMapped);		// Size of workspace required for forward transform
	cudaHostAlloc((void**)&fft.wsize_i, 		 nGPUs*sizeof(size_t *), 						 cudaHostAllocMapped);		// Size of workspace required for inverse transform
	cudaHostAlloc((void**)&fft.wspace, 			 nGPUs*sizeof(cufftDoubleComplex *), cudaHostAllocMapped);		// Array of pointers to FFT workspace on each device
	cudaHostAlloc((void**)&fft.temp, 				 nGPUs*sizeof(cufftDoubleComplex *), cudaHostAllocMapped);		// Array of pointers to scratch (temporary) memory on each device
	cudaHostAlloc((void**)&fft.temp_reorder, nGPUs*sizeof(cufftDoubleComplex *), cudaHostAllocMapped);		// Same as above, different temp variable
	
	// Allocate memory on host to store averaged profile data
	cudaHostAlloc((void**)&Yprof,         sizeof(profile),   					       cudaHostAllocMapped);
	cudaHostAlloc((void**)&Yprof.u,       nGPUs*sizeof(double *),    	       cudaHostAllocMapped);
	cudaHostAlloc((void**)&Yprof.v,       nGPUs*sizeof(double *),    	       cudaHostAllocMapped);
	cudaHostAlloc((void**)&Yprof.w,       nGPUs*sizeof(double *),    	       cudaHostAllocMapped);
	cudaHostAlloc((void**)&Yprof.s,       nGPUs*sizeof(double *),    	       cudaHostAllocMapped);

	// Allocate pinned memory on the host side that stores array of pointers
	cudaHostAlloc((void**)&grid, sizeof(fielddata), cudaHostAllocMapped);
	cudaHostAlloc((void**)&grid.kx, nGPUs*sizeof(double *), cudaHostAllocMapped);

	// Allocate memory on host
	cudaHostAlloc((void**)&h_vel, sizeof(fielddata), cudaHostAllocMapped);
	h_vel.u = (double **)malloc(sizeof(double *)*nGPUs);
	h_vel.v = (double **)malloc(sizeof(double *)*nGPUs);
	h_vel.w = (double **)malloc(sizeof(double *)*nGPUs);
	h_vel.s = (double **)malloc(sizeof(double *)*nGPUs);

  cudaHostAlloc((void**)&vel, sizeof(fielddata), cudaHostAllocMapped);
	cudaHostAlloc((void**)&vel.uh, 		 nGPUs*sizeof(cufftDoubleComplex *), cudaHostAllocMapped);
	cudaHostAlloc((void**)&vel.vh, 		 nGPUs*sizeof(cufftDoubleComplex *), cudaHostAllocMapped);
	cudaHostAlloc((void**)&vel.wh, 		 nGPUs*sizeof(cufftDoubleComplex *), cudaHostAllocMapped);
	cudaHostAlloc((void**)&vel.sh, 		 nGPUs*sizeof(cufftDoubleComplex *), cudaHostAllocMapped);
	cudaHostAlloc((void**)&vel.left, 	 nGPUs*sizeof(cufftDoubleComplex *), cudaHostAllocMapped);
	cudaHostAlloc((void**)&vel.right,  nGPUs*sizeof(cufftDoubleComplex *), cudaHostAllocMapped);

  cudaHostAlloc((void**)&rhs, sizeof(fielddata), cudaHostAllocMapped);
	cudaHostAlloc((void**)&rhs.uh, 		 nGPUs*sizeof(cufftDoubleComplex *), cudaHostAllocMapped);
	cudaHostAlloc((void**)&rhs.vh, 		 nGPUs*sizeof(cufftDoubleComplex *), cudaHostAllocMapped);
	cudaHostAlloc((void**)&rhs.wh, 		 nGPUs*sizeof(cufftDoubleComplex *), cudaHostAllocMapped);
	cudaHostAlloc((void**)&rhs.sh, 		 nGPUs*sizeof(cufftDoubleComplex *), cudaHostAllocMapped);
	cudaHostAlloc((void**)&rhs.left, 		 nGPUs*sizeof(cufftDoubleComplex *), cudaHostAllocMapped);
	cudaHostAlloc((void**)&rhs.right, 		 nGPUs*sizeof(cufftDoubleComplex *), cudaHostAllocMapped);

  cudaHostAlloc((void**)&rhs_old, sizeof(fielddata), cudaHostAllocMapped);
	cudaHostAlloc((void**)&rhs_old.uh, nGPUs*sizeof(cufftDoubleComplex *), cudaHostAllocMapped);
	cudaHostAlloc((void**)&rhs_old.vh, nGPUs*sizeof(cufftDoubleComplex *), cudaHostAllocMapped);
	cudaHostAlloc((void**)&rhs_old.wh, nGPUs*sizeof(cufftDoubleComplex *), cudaHostAllocMapped);
	cudaHostAlloc((void**)&rhs_old.sh, nGPUs*sizeof(cufftDoubleComplex *), cudaHostAllocMapped);

  cudaHostAlloc((void**)&temp, sizeof(fielddata), cudaHostAllocMapped);
	cudaHostAlloc((void**)&temp.uh, nGPUs*sizeof(cufftDoubleComplex *), cudaHostAllocMapped);
	
	// For statistics
	cudaHostAlloc(&stats, nGPUs*sizeof(statistics *), cudaHostAllocMapped);

	// Allocate memory for arrays on each GPU
	for (n = 0; n<nGPUs; ++n){
		cudaSetDevice(n);

		h_vel.u[n] = (double *)malloc(sizeof(complex double)*gpu.nx[n]*NY*NZ2);
		h_vel.v[n] = (double *)malloc(sizeof(complex double)*gpu.nx[n]*NY*NZ2);
		h_vel.w[n] = (double *)malloc(sizeof(complex double)*gpu.nx[n]*NY*NZ2);
		h_vel.s[n] = (double *)malloc(sizeof(complex double)*gpu.nx[n]*NY*NZ2);

		checkCudaErrors( cudaMalloc((void **)&grid.kx[n], sizeof(double)*NX ) );

		// Allocate memory for velocity fields
		checkCudaErrors( cudaMalloc((void **)&vel.uh[n],    sizeof(cufftDoubleComplex)*gpu.nx[n]*NY*NZ2) ); 
		checkCudaErrors( cudaMalloc((void **)&vel.vh[n],    sizeof(cufftDoubleComplex)*gpu.nx[n]*NY*NZ2) );
		checkCudaErrors( cudaMalloc((void **)&vel.wh[n],    sizeof(cufftDoubleComplex)*gpu.nx[n]*NY*NZ2) );
		checkCudaErrors( cudaMalloc((void **)&vel.sh[n],    sizeof(cufftDoubleComplex)*gpu.nx[n]*NY*NZ2) );
		checkCudaErrors( cudaMalloc((void **)&vel.left[n],  sizeof(cufftDoubleComplex)*RAD*NY*NZ2) );
		checkCudaErrors( cudaMalloc((void **)&vel.right[n], sizeof(cufftDoubleComplex)*RAD*NY*NZ2) );

		checkCudaErrors( cudaMalloc((void **)&rhs.uh[n], sizeof(cufftDoubleComplex)*gpu.nx[n]*NY*NZ2) ); 
		checkCudaErrors( cudaMalloc((void **)&rhs.vh[n], sizeof(cufftDoubleComplex)*gpu.nx[n]*NY*NZ2) );
		checkCudaErrors( cudaMalloc((void **)&rhs.wh[n], sizeof(cufftDoubleComplex)*gpu.nx[n]*NY*NZ2) );
		checkCudaErrors( cudaMalloc((void **)&rhs.sh[n], sizeof(cufftDoubleComplex)*gpu.nx[n]*NY*NZ2) );
		checkCudaErrors( cudaMalloc((void **)&rhs.left[n],  sizeof(cufftDoubleComplex)*RAD*NY*NZ2) );
		checkCudaErrors( cudaMalloc((void **)&rhs.right[n],  sizeof(cufftDoubleComplex)*RAD*NY*NZ2) );

		checkCudaErrors( cudaMallocManaged((void **)&rhs_old.uh[n], sizeof(cufftDoubleComplex)*gpu.nx[n]*NY*NZ2) ); 
		checkCudaErrors( cudaMallocManaged((void **)&rhs_old.vh[n], sizeof(cufftDoubleComplex)*gpu.nx[n]*NY*NZ2) );
		checkCudaErrors( cudaMallocManaged((void **)&rhs_old.wh[n], sizeof(cufftDoubleComplex)*gpu.nx[n]*NY*NZ2) );
		checkCudaErrors( cudaMallocManaged((void **)&rhs_old.sh[n], sizeof(cufftDoubleComplex)*gpu.nx[n]*NY*NZ2) );

		checkCudaErrors( cudaMalloc((void **)&temp.uh[n],   sizeof(cufftDoubleComplex)*gpu.nx[n]*NY*NZ2) );
		checkCudaErrors( cudaMalloc((void **)&fft.temp[n], 			 	 sizeof(cufftDoubleComplex)*gpu.nx[n]*NY*NZ2) );
		checkCudaErrors( cudaMalloc((void **)&fft.temp_reorder[n], sizeof(cufftDoubleComplex)*gpu.nx[n]*NZ2) );
		
		// Statistics
		checkCudaErrors( cudaMallocManaged( (void **)&stats[n], sizeof(statistics) ));
		
		// Averaged Profiles
		checkCudaErrors( cudaMallocManaged( (void **)&Yprof.u[n], sizeof(double)*NY) );
		checkCudaErrors( cudaMallocManaged( (void **)&Yprof.v[n], sizeof(double)*NY) );
		checkCudaErrors( cudaMallocManaged( (void **)&Yprof.w[n], sizeof(double)*NY) );
		checkCudaErrors( cudaMallocManaged( (void **)&Yprof.s[n], sizeof(double)*NY) );

		printf("Data allocated on Device %d\n", n);
	}

		// Cast pointers to complex arrays to real array and store in the proper struct field
		vel.u = (cufftDoubleReal **)vel.uh;
		vel.v = (cufftDoubleReal **)vel.vh;
		vel.w = (cufftDoubleReal **)vel.wh;
		vel.s = (cufftDoubleReal **)vel.sh;
	
		rhs.u = (cufftDoubleReal **)rhs.uh;
		rhs.v = (cufftDoubleReal **)rhs.vh;
		rhs.w = (cufftDoubleReal **)rhs.wh;
		rhs.s = (cufftDoubleReal **)rhs.sh;

		rhs_old.u = (cufftDoubleReal **)rhs_old.uh;
		rhs_old.v = (cufftDoubleReal **)rhs_old.vh;
		rhs_old.w = (cufftDoubleReal **)rhs_old.wh;
		rhs_old.s = (cufftDoubleReal **)rhs_old.sh;		
		
		temp.u = (cufftDoubleReal **)temp.uh;

	// Initialize everything to 0 before entering the rest of the routine
	for (n = 0; n<nGPUs; ++n){
		cudaSetDevice(n);

		checkCudaErrors( cudaMemset(grid.kx[n], 0.0, sizeof(double)*NX) );

		checkCudaErrors( cudaMemset(vel.u[n], 0.0, sizeof(cufftDoubleComplex)*gpu.nx[n]*NY*NZ2) );
		checkCudaErrors( cudaMemset(vel.v[n], 0.0, sizeof(cufftDoubleComplex)*gpu.nx[n]*NY*NZ2) );
		checkCudaErrors( cudaMemset(vel.w[n], 0.0, sizeof(cufftDoubleComplex)*gpu.nx[n]*NY*NZ2) );
		checkCudaErrors( cudaMemset(vel.s[n], 0.0, sizeof(cufftDoubleComplex)*gpu.nx[n]*NY*NZ2) );

		checkCudaErrors( cudaMemset(rhs.u[n], 0.0, sizeof(cufftDoubleComplex)*gpu.nx[n]*NY*NZ2) );
		checkCudaErrors( cudaMemset(rhs.v[n], 0.0, sizeof(cufftDoubleComplex)*gpu.nx[n]*NY*NZ2) );
		checkCudaErrors( cudaMemset(rhs.w[n], 0.0, sizeof(cufftDoubleComplex)*gpu.nx[n]*NY*NZ2) );
		checkCudaErrors( cudaMemset(rhs.s[n], 0.0, sizeof(cufftDoubleComplex)*gpu.nx[n]*NY*NZ2) );

		checkCudaErrors( cudaMemset(rhs_old.u[n], 0.0, sizeof(cufftDoubleComplex)*gpu.nx[n]*NY*NZ2) );
		checkCudaErrors( cudaMemset(rhs_old.v[n], 0.0, sizeof(cufftDoubleComplex)*gpu.nx[n]*NY*NZ2) );
		checkCudaErrors( cudaMemset(rhs_old.w[n], 0.0, sizeof(cufftDoubleComplex)*gpu.nx[n]*NY*NZ2) );
		checkCudaErrors( cudaMemset(rhs_old.s[n], 0.0, sizeof(cufftDoubleComplex)*gpu.nx[n]*NY*NZ2) );

		checkCudaErrors( cudaMemset(temp.uh[n], 0.0, sizeof(cufftDoubleComplex)*gpu.nx[n]*NY*NZ2) );
		
		checkCudaErrors( cudaMemset(Yprof.u[n], 0.0, sizeof(double)*NY) );
		checkCudaErrors( cudaMemset(Yprof.v[n], 0.0, sizeof(double)*NY) );
		checkCudaErrors( cudaMemset(Yprof.w[n], 0.0, sizeof(double)*NY) );
		checkCudaErrors( cudaMemset(Yprof.s[n], 0.0, sizeof(double)*NY) );
	}

	return;
}


void deallocate_memory(){
	int n, nGPUs;
	// // Declare extern variables (to pull def's from declare.h)
	extern gpudata gpu;	
	extern fftdata fft;
	extern statistics h_stats;
	extern statistics *stats;	
	extern profile Yprof;

  extern griddata grid;

  extern fielddata h_vel;
  extern fielddata vel;
  extern fielddata rhs;
  extern fielddata rhs_old;

	extern fielddata temp;

	// Make local copy of number of GPUs (for readability)
	nGPUs = gpu.nGPUs;

	// Deallocate GPU memory
	for(n = 0; n<nGPUs; ++n){
		cudaSetDevice(n);

		cudaFree(fft.temp[n]);
		cudaFree(fft.temp_reorder[n]);
   	cudaFree(fft.wspace[n]);

		cudaFree(grid.kx[n]);

		free(h_vel.u[n]);
		free(h_vel.v[n]);
		free(h_vel.w[n]);
		free(h_vel.s[n]);

		cudaFree(vel.u[n]);
		cudaFree(vel.v[n]);
		cudaFree(vel.w[n]);
		cudaFree(vel.s[n]);

		cudaFree(rhs.u[n]);
		cudaFree(rhs.v[n]);
		cudaFree(rhs.w[n]);
		cudaFree(rhs.s[n]);

		cudaFree(rhs_old.u[n]);
		cudaFree(rhs_old.v[n]);
		cudaFree(rhs_old.w[n]);
		cudaFree(rhs_old.s[n]);

		cudaFree(temp.u[n]);

		cudaFree(&stats[n]);
		// Averaged Profiles
		cudaFree(Yprof.u[n]);
		cudaFree(Yprof.v[n]);
		cudaFree(Yprof.w[n]);
		cudaFree(Yprof.s[n]);

		// Destroy cufft plans
		cufftDestroy(fft.p1d[n]);
		cufftDestroy(fft.p2d[n]);
		cufftDestroy(fft.invp2d[n]);
		cufftDestroy(fft.p3d[n]);
		cufftDestroy(fft.invp3d[n]);
	}
	
	// Deallocate pointer arrays on host memory
	cudaFreeHost(gpu.gpunum);
	cudaFreeHost(gpu.ny);
	cudaFreeHost(gpu.nx);
	cudaFreeHost(gpu.start_x);
	cudaFreeHost(gpu.start_y);

	cudaFreeHost(grid.kx);
	cudaFreeHost(&grid);

	cudaFreeHost(temp.uh);
	cudaFreeHost(&temp);

	cudaFreeHost(fft.wsize_f);
	cudaFreeHost(fft.wsize_i);
	cudaFreeHost(fft.wspace);
	cudaFreeHost(fft.temp);
	cudaFreeHost(fft.temp_reorder);
	cudaFreeHost(&fft);

	cudaFreeHost(vel.uh);
	cudaFreeHost(vel.vh);
	cudaFreeHost(vel.wh);
	cudaFreeHost(vel.sh);
	cudaFreeHost(&vel);

	cudaFreeHost(rhs.uh);
	cudaFreeHost(rhs.vh);
	cudaFreeHost(rhs.wh);
	cudaFreeHost(rhs.sh);
	cudaFreeHost(&rhs);

	cudaFreeHost(rhs_old.uh);
	cudaFreeHost(rhs_old.vh);
	cudaFreeHost(rhs_old.wh);
	cudaFreeHost(rhs_old.sh);
	cudaFreeHost(&rhs_old);

	cudaFreeHost(stats);
	
	// Averaged Profiles
	cudaFreeHost(Yprof.u);
	cudaFreeHost(Yprof.v);
	cudaFreeHost(Yprof.w);
	cudaFreeHost(Yprof.s);
	cudaFreeHost(&Yprof);

	// Deallocate memory on CPU
	free(h_vel.u);
	free(h_vel.v);
	free(h_vel.w);
	free(h_vel.s);
	cudaFreeHost(&h_vel);

	return;
}
