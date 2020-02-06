#include <stdlib.h>
#include <complex.h>
#include <cufft.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include "dnsparams.h"
#include "struct_def.h"

void allocate_memory(){
	// // Allocate memory for statistics structs on both device and host
	int n, nGPUs;

	// // Declare extern variables (to pull def's from declare.h)
	extern gpuinfo gpu;	
	extern fftinfo fft;
	extern statistics *stats;	
	extern profile Yprofile;

  extern double **k;

  extern fielddata h_vel;
  extern fielddata vel;
  extern fielddata rhs;
  extern fielddata rhs_old;

	extern cufftDoubleComplex **temp_advective;

	// Make local copy of number of GPUs (for readability)
	nGPUs = gpu.nGPUs;
	printf("Allocating data on %d GPUs!\n",nGPUs);
	
	// Allocate pinned memory on the host side that stores array of pointers for FFT operations
	cudaHostAlloc((void**)&fft,         		 sizeof(fftinfo),   					       cudaHostAllocMapped);		
	cudaHostAlloc((void**)&fft.p1d,    			 nGPUs*sizeof(cufftHandle *), 			 cudaHostAllocMapped);		// Allocate memory for array of cufftHandles to store nGPUs worth 1d plans
	cudaHostAlloc((void**)&fft.p2d,    			 nGPUs*sizeof(cufftHandle *), 			 cudaHostAllocMapped);		// Allocate memory array of 2dplans
	cudaHostAlloc((void**)&fft.invp2d, 			 nGPUs*sizeof(cufftHandle *), 			 cudaHostAllocMapped);		// Array of inverse 2d plans
	cudaHostAlloc((void**)&fft.wsize_f, 		 nGPUs*sizeof(size_t *), 						 cudaHostAllocMapped);		// Size of workspace required for forward transform
	cudaHostAlloc((void**)&fft.wsize_i, 		 nGPUs*sizeof(size_t *), 						 cudaHostAllocMapped);		// Size of workspace required for inverse transform
	cudaHostAlloc((void**)&fft.wspace, 			 nGPUs*sizeof(cufftDoubleComplex *), cudaHostAllocMapped);		// Array of pointers to FFT workspace on each device
	cudaHostAlloc((void**)&fft.temp, 				 nGPUs*sizeof(cufftDoubleComplex *), cudaHostAllocMapped);		// Array of pointers to scratch (temporary) memory on each device
	cudaHostAlloc((void**)&fft.temp_reorder, nGPUs*sizeof(cufftDoubleComplex *), cudaHostAllocMapped);		// Same as above, different temp variable
	
	// Allocate memory on host to store averaged profile data
	cudaHostAlloc((void**)&Yprofile,         sizeof(profile),   					       cudaHostAllocMapped);
	cudaHostAlloc((void**)&Yprofile.u,       nGPUs*sizeof(double *),    	       cudaHostAllocMapped);
	cudaHostAlloc((void**)&Yprofile.v,       nGPUs*sizeof(double *),    	       cudaHostAllocMapped);
	cudaHostAlloc((void**)&Yprofile.w,       nGPUs*sizeof(double *),    	       cudaHostAllocMapped);
	cudaHostAlloc((void**)&Yprofile.s,       nGPUs*sizeof(double *),    	       cudaHostAllocMapped);
	cudaHostAlloc((void**)&Yprofile.uu,       nGPUs*sizeof(double *),    	       cudaHostAllocMapped);
	cudaHostAlloc((void**)&Yprofile.vv,       nGPUs*sizeof(double *),    	       cudaHostAllocMapped);
	cudaHostAlloc((void**)&Yprofile.ww,       nGPUs*sizeof(double *),    	       cudaHostAllocMapped);
	cudaHostAlloc((void**)&Yprofile.ss,       nGPUs*sizeof(double *),    	       cudaHostAllocMapped);

	// Allocate memory on host
	h_vel.u = (double **)malloc(sizeof(double *)*nGPUs);
	h_vel.v = (double **)malloc(sizeof(double *)*nGPUs);
	h_vel.w = (double **)malloc(sizeof(double *)*nGPUs);
	h_vel.s = (double **)malloc(sizeof(double *)*nGPUs);

	// Allocate pinned memory on the host side that stores array of pointers
	cudaHostAlloc((void**)&k, nGPUs*sizeof(double *), cudaHostAllocMapped);

	cudaHostAlloc((void**)&vel.uh, 		 nGPUs*sizeof(cufftDoubleComplex *), cudaHostAllocMapped);
	cudaHostAlloc((void**)&vel.vh, 		 nGPUs*sizeof(cufftDoubleComplex *), cudaHostAllocMapped);
	cudaHostAlloc((void**)&vel.wh, 		 nGPUs*sizeof(cufftDoubleComplex *), cudaHostAllocMapped);
	cudaHostAlloc((void**)&vel.sh, 		 nGPUs*sizeof(cufftDoubleComplex *), cudaHostAllocMapped);
	cudaHostAlloc((void**)&vel.left, 	 nGPUs*sizeof(cufftDoubleComplex *), cudaHostAllocMapped);
	cudaHostAlloc((void**)&vel.right,  nGPUs*sizeof(cufftDoubleComplex *), cudaHostAllocMapped);

	cudaHostAlloc((void**)&rhs.uh, 		 nGPUs*sizeof(cufftDoubleComplex *), cudaHostAllocMapped);
	cudaHostAlloc((void**)&rhs.vh, 		 nGPUs*sizeof(cufftDoubleComplex *), cudaHostAllocMapped);
	cudaHostAlloc((void**)&rhs.wh, 		 nGPUs*sizeof(cufftDoubleComplex *), cudaHostAllocMapped);
	cudaHostAlloc((void**)&rhs.sh, 		 nGPUs*sizeof(cufftDoubleComplex *), cudaHostAllocMapped);
	cudaHostAlloc((void**)&rhs.left, 		 nGPUs*sizeof(cufftDoubleComplex *), cudaHostAllocMapped);
	cudaHostAlloc((void**)&rhs.right, 		 nGPUs*sizeof(cufftDoubleComplex *), cudaHostAllocMapped);

	cudaHostAlloc((void**)&rhs_old.uh, nGPUs*sizeof(cufftDoubleComplex *), cudaHostAllocMapped);
	cudaHostAlloc((void**)&rhs_old.vh, nGPUs*sizeof(cufftDoubleComplex *), cudaHostAllocMapped);
	cudaHostAlloc((void**)&rhs_old.wh, nGPUs*sizeof(cufftDoubleComplex *), cudaHostAllocMapped);
	cudaHostAlloc((void**)&rhs_old.sh, nGPUs*sizeof(cufftDoubleComplex *), cudaHostAllocMapped);

	cudaHostAlloc((void**)&temp_advective, nGPUs*sizeof(cufftDoubleComplex *), cudaHostAllocMapped);
	
	// For statistics
	cudaHostAlloc(&stats, nGPUs*sizeof(statistics *), cudaHostAllocMapped);

	// Allocate memory for arrays
	for (n = 0; n<nGPUs; ++n){
		cudaSetDevice(n);

		h_vel.u[n] = (double *)malloc(sizeof(complex double)*gpu.nx[n]*NY*NZ2);
		h_vel.v[n] = (double *)malloc(sizeof(complex double)*gpu.nx[n]*NY*NZ2);
		h_vel.w[n] = (double *)malloc(sizeof(complex double)*gpu.nx[n]*NY*NZ2);
		h_vel.s[n] = (double *)malloc(sizeof(complex double)*gpu.nx[n]*NY*NZ2);

		checkCudaErrors( cudaMalloc((void **)&k[n], sizeof(double)*NX ) );

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

		checkCudaErrors( cudaMalloc((void **)&temp_advective[n],   sizeof(cufftDoubleComplex)*gpu.nx[n]*NY*NZ2) );
		checkCudaErrors( cudaMalloc((void **)&fft.temp[n], 			 	 sizeof(cufftDoubleComplex)*gpu.nx[n]*NY*NZ2) );
		checkCudaErrors( cudaMalloc((void **)&fft.temp_reorder[n], sizeof(cufftDoubleComplex)*gpu.nx[n]*NZ2) );
		
		// Statistics
		checkCudaErrors( cudaMallocManaged( (void **)&stats[n], sizeof(statistics) ));
		
		// Averaged Profiles
		checkCudaErrors( cudaMallocManaged( (void **)&Yprofile.u[n], sizeof(double)*NY) );
		checkCudaErrors( cudaMallocManaged( (void **)&Yprofile.v[n], sizeof(double)*NY) );
		checkCudaErrors( cudaMallocManaged( (void **)&Yprofile.w[n], sizeof(double)*NY) );
		checkCudaErrors( cudaMallocManaged( (void **)&Yprofile.s[n], sizeof(double)*NY) );
		checkCudaErrors( cudaMallocManaged( (void **)&Yprofile.uu[n], sizeof(double)*NY) );
		checkCudaErrors( cudaMallocManaged( (void **)&Yprofile.vv[n], sizeof(double)*NY) );
		checkCudaErrors( cudaMallocManaged( (void **)&Yprofile.ww[n], sizeof(double)*NY) );
		checkCudaErrors( cudaMallocManaged( (void **)&Yprofile.ss[n], sizeof(double)*NY) );
		
		// Area statistics
		checkCudaErrors( cudaMallocManaged( (void **)&stats[n].area_scalar, sizeof(double)*64) );
		checkCudaErrors( cudaMallocManaged( (void **)&stats[n].area_omega, sizeof(double)*64) );

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

	// Initialize everything to 0 before entering the rest of the routine
	for (n = 0; n<nGPUs; ++n){
		cudaSetDevice(n);

		checkCudaErrors( cudaMemset(k[n], 0, sizeof(double)*NX) );

		checkCudaErrors( cudaMemset(vel.u[n], 0, sizeof(cufftDoubleComplex)*gpu.nx[n]*NY*NZ2) );
		checkCudaErrors( cudaMemset(vel.v[n], 0, sizeof(cufftDoubleComplex)*gpu.nx[n]*NY*NZ2) );
		checkCudaErrors( cudaMemset(vel.w[n], 0, sizeof(cufftDoubleComplex)*gpu.nx[n]*NY*NZ2) );
		checkCudaErrors( cudaMemset(vel.s[n], 0, sizeof(cufftDoubleComplex)*gpu.nx[n]*NY*NZ2) );

		checkCudaErrors( cudaMemset(rhs.u[n], 0, sizeof(cufftDoubleComplex)*gpu.nx[n]*NY*NZ2) );
		checkCudaErrors( cudaMemset(rhs.v[n], 0, sizeof(cufftDoubleComplex)*gpu.nx[n]*NY*NZ2) );
		checkCudaErrors( cudaMemset(rhs.w[n], 0, sizeof(cufftDoubleComplex)*gpu.nx[n]*NY*NZ2) );
		checkCudaErrors( cudaMemset(rhs.s[n], 0, sizeof(cufftDoubleComplex)*gpu.nx[n]*NY*NZ2) );

		checkCudaErrors( cudaMemset(rhs_old.u[n], 0, sizeof(cufftDoubleComplex)*gpu.nx[n]*NY*NZ2) );
		checkCudaErrors( cudaMemset(rhs_old.v[n], 0, sizeof(cufftDoubleComplex)*gpu.nx[n]*NY*NZ2) );
		checkCudaErrors( cudaMemset(rhs_old.w[n], 0, sizeof(cufftDoubleComplex)*gpu.nx[n]*NY*NZ2) );
		checkCudaErrors( cudaMemset(rhs_old.s[n], 0, sizeof(cufftDoubleComplex)*gpu.nx[n]*NY*NZ2) );

		checkCudaErrors( cudaMemset(temp_advective[n], 0, sizeof(cufftDoubleComplex)*gpu.nx[n]*NY*NZ2) );
		
		checkCudaErrors( cudaMemset(Yprofile.u[n], 0, sizeof(double)*NY) );
		checkCudaErrors( cudaMemset(Yprofile.v[n], 0, sizeof(double)*NY) );
		checkCudaErrors( cudaMemset(Yprofile.w[n], 0, sizeof(double)*NY) );
		checkCudaErrors( cudaMemset(Yprofile.s[n], 0, sizeof(double)*NY) );
	}

	return;
}


void deallocate_memory(){
	int n, nGPUs;
	// // Declare extern variables (to pull def's from declare.h)
	extern gpuinfo gpu;	
	extern fftinfo fft;
	extern statistics h_stats;
	extern statistics *stats;	
	extern profile Yprofile;

  extern double **k;

  extern fielddata h_vel;
  extern fielddata vel;
  extern fielddata rhs;
  extern fielddata rhs_old;

	extern cufftDoubleComplex **temp_advective;

	// Make local copy of number of GPUs (for readability)
	nGPUs = gpu.nGPUs;

	// Deallocate GPU memory
	for(n = 0; n<nGPUs; ++n){
		cudaSetDevice(n);

		cudaFree(fft.temp[n]);
		cudaFree(fft.temp_reorder[n]);
   	cudaFree(fft.wspace[n]);

		cudaFree(k[n]);

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

		cudaFree(temp_advective[n]);

		cudaFree(&stats[n]);
		// Averaged Profiles
		cudaFree(Yprofile.u[n]);
		cudaFree(Yprofile.v[n]);
		cudaFree(Yprofile.w[n]);
		cudaFree(Yprofile.s[n]);

		// Destroy cufft plans
		cufftDestroy(fft.p1d[n]);
		cufftDestroy(fft.p2d[n]);
		cufftDestroy(fft.invp2d[n]);
	}
	
	// Deallocate pointer arrays on host memory
	cudaFreeHost(gpu.gpunum);
	cudaFreeHost(gpu.ny);
	cudaFreeHost(gpu.nx);
	cudaFreeHost(gpu.start_x);
	cudaFreeHost(gpu.start_y);

	cudaFreeHost(k);

	cudaFreeHost(temp_advective);

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

	cudaFreeHost(rhs.uh);
	cudaFreeHost(rhs.vh);
	cudaFreeHost(rhs.wh);
	cudaFreeHost(rhs.sh);

	cudaFreeHost(rhs_old.uh);
	cudaFreeHost(rhs_old.vh);
	cudaFreeHost(rhs_old.wh);
	cudaFreeHost(rhs_old.sh);

	cudaFreeHost(stats);
	
	// Averaged Profiles
	cudaFreeHost(Yprofile.u);
	cudaFreeHost(Yprofile.v);
	cudaFreeHost(Yprofile.w);
	cudaFreeHost(Yprofile.s);
	cudaFreeHost(&Yprofile);

	// Deallocate memory on CPU
	free(h_vel.u);
	free(h_vel.v);
	free(h_vel.w);
	free(h_vel.s);

	return;
}
