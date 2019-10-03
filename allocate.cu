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
	extern statistics h_stats;
	extern statistics stats;	

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

	// Allocate memory on host
	h_vel.u = (double **)malloc(sizeof(double *)*nGPUs);
	h_vel.v = (double **)malloc(sizeof(double *)*nGPUs);
	h_vel.w = (double **)malloc(sizeof(double *)*nGPUs);
	h_vel.s = (double **)malloc(sizeof(double *)*nGPUs);

	// Declare struct for holding statistics on the host
	// &h_stats = malloc(sizeof(statistics));
	// Allocate memory on host for statistics
	h_stats.Vrms    			= (double **)malloc(sizeof(double *));
	h_stats.KE      			= (double **)malloc(sizeof(double *));
	h_stats.epsilon 			= (double **)malloc(sizeof(double *));
	h_stats.eta     			= (double **)malloc(sizeof(double *));
	h_stats.l      		 		= (double **)malloc(sizeof(double *));
	h_stats.lambda 				= (double **)malloc(sizeof(double *));
	h_stats.chi   			  = (double **)malloc(sizeof(double *));
	h_stats.area_scalar  	= (double **)malloc(sizeof(double *));
	h_stats.area_tnti	  	= (double **)malloc(sizeof(double *));
	h_stats.energy_spect	= (double **)malloc(sizeof(double *));

	h_stats.Vrms[0]    			= (double *)malloc(sizeof(double)*(nt/n_stats+1));
	h_stats.KE[0]      			= (double *)malloc(sizeof(double)*(nt/n_stats+1));
	h_stats.epsilon[0] 			= (double *)malloc(sizeof(double)*(nt/n_stats+1));
	h_stats.eta[0]     			= (double *)malloc(sizeof(double)*(nt/n_stats+1));
	h_stats.l[0]      		 		= (double *)malloc(sizeof(double)*(nt/n_stats+1));
	h_stats.lambda[0] 				= (double *)malloc(sizeof(double)*(nt/n_stats+1));
	h_stats.chi[0]   			  = (double *)malloc(sizeof(double)*(nt/n_stats+1));
	h_stats.area_scalar[0]  	= (double *)malloc(sizeof(double)*(nt/n_stats+1));
	h_stats.area_tnti[0]	  	= (double *)malloc(sizeof(double)*(nt/n_stats+1));
	h_stats.energy_spect[0]		= (double *)malloc(sizeof(double)*(nt/n_stats+1)*NX/2);

	// Allocate pinned memory on the host side that stores array of pointers
	cudaHostAlloc((void**)&k, nGPUs*sizeof(double *), cudaHostAllocMapped);

	cudaHostAlloc((void**)&vel.uh, 		 nGPUs*sizeof(cufftDoubleComplex *), cudaHostAllocMapped);
	cudaHostAlloc((void**)&vel.vh, 		 nGPUs*sizeof(cufftDoubleComplex *), cudaHostAllocMapped);
	cudaHostAlloc((void**)&vel.wh, 		 nGPUs*sizeof(cufftDoubleComplex *), cudaHostAllocMapped);
	cudaHostAlloc((void**)&vel.sh, 		 nGPUs*sizeof(cufftDoubleComplex *), cudaHostAllocMapped);
	cudaHostAlloc((void**)&vel.left, 	 nGPUs*sizeof(cufftDoubleComplex *), cudaHostAllocMapped);
	cudaHostAlloc((void**)&vel.right,  nGPUs*sizeof(cufftDoubleComplex *), cudaHostAllocMapped);

	// cudaHostAlloc((void**)&vel.u, 		 nGPUs*sizeof(cufftDoubleComplex *), cudaHostAllocMapped);
	// cudaHostAlloc((void**)&vel.v, 		 nGPUs*sizeof(cufftDoubleComplex *), cudaHostAllocMapped);
	// cudaHostAlloc((void**)&vel.w, 		 nGPUs*sizeof(cufftDoubleComplex *), cudaHostAllocMapped);
	// cudaHostAlloc((void**)&vel.s, 		 nGPUs*sizeof(cufftDoubleComplex *), cudaHostAllocMapped);

	cudaHostAlloc((void**)&rhs.uh, 		 nGPUs*sizeof(cufftDoubleComplex *), cudaHostAllocMapped);
	cudaHostAlloc((void**)&rhs.vh, 		 nGPUs*sizeof(cufftDoubleComplex *), cudaHostAllocMapped);
	cudaHostAlloc((void**)&rhs.wh, 		 nGPUs*sizeof(cufftDoubleComplex *), cudaHostAllocMapped);
	cudaHostAlloc((void**)&rhs.sh, 		 nGPUs*sizeof(cufftDoubleComplex *), cudaHostAllocMapped);

	// cudaHostAlloc((void**)&rhs.u, 		 nGPUs*sizeof(cufftDoubleComplex *), cudaHostAllocMapped);
	// cudaHostAlloc((void**)&rhs.v, 		 nGPUs*sizeof(cufftDoubleComplex *), cudaHostAllocMapped);
	// cudaHostAlloc((void**)&rhs.w, 		 nGPUs*sizeof(cufftDoubleComplex *), cudaHostAllocMapped);
	// cudaHostAlloc((void**)&rhs.s, 		 nGPUs*sizeof(cufftDoubleComplex *), cudaHostAllocMapped);

	cudaHostAlloc((void**)&rhs_old.uh, nGPUs*sizeof(cufftDoubleComplex *), cudaHostAllocMapped);
	cudaHostAlloc((void**)&rhs_old.vh, nGPUs*sizeof(cufftDoubleComplex *), cudaHostAllocMapped);
	cudaHostAlloc((void**)&rhs_old.wh, nGPUs*sizeof(cufftDoubleComplex *), cudaHostAllocMapped);
	cudaHostAlloc((void**)&rhs_old.sh, nGPUs*sizeof(cufftDoubleComplex *), cudaHostAllocMapped);

	// cudaHostAlloc((void**)&rhs_old.u, nGPUs*sizeof(cufftDoubleComplex *), cudaHostAllocMapped);
	// cudaHostAlloc((void**)&rhs_old.v, nGPUs*sizeof(cufftDoubleComplex *), cudaHostAllocMapped);
	// cudaHostAlloc((void**)&rhs_old.w, nGPUs*sizeof(cufftDoubleComplex *), cudaHostAllocMapped);
	// cudaHostAlloc((void**)&rhs_old.s, nGPUs*sizeof(cufftDoubleComplex *), cudaHostAllocMapped);

	cudaHostAlloc((void**)&temp_advective, nGPUs*sizeof(cufftDoubleComplex *), cudaHostAllocMapped);
	
	// For statistics
	// cudaHostAlloc((void**)&stats, sizeof(stats), cudaHostAllocMapped);
	cudaHostAlloc((void**)&stats.Vrms,    nGPUs*sizeof(cufftDoubleComplex *), cudaHostAllocMapped);
	cudaHostAlloc((void**)&stats.KE,      nGPUs*sizeof(cufftDoubleComplex *), cudaHostAllocMapped);
	cudaHostAlloc((void**)&stats.epsilon, nGPUs*sizeof(cufftDoubleComplex *), cudaHostAllocMapped);
	cudaHostAlloc((void**)&stats.eta,     nGPUs*sizeof(cufftDoubleComplex *), cudaHostAllocMapped);
	cudaHostAlloc((void**)&stats.l,       nGPUs*sizeof(cufftDoubleComplex *), cudaHostAllocMapped);
	cudaHostAlloc((void**)&stats.lambda,  nGPUs*sizeof(cufftDoubleComplex *), cudaHostAllocMapped);
	cudaHostAlloc((void**)&stats.chi,     nGPUs*sizeof(cufftDoubleComplex *), cudaHostAllocMapped);
	cudaHostAlloc((void**)&stats.area_scalar,     nGPUs*sizeof(cufftDoubleComplex *), cudaHostAllocMapped);
	cudaHostAlloc((void**)&stats.area_tnti,     nGPUs*sizeof(cufftDoubleComplex *), cudaHostAllocMapped);
	cudaHostAlloc((void**)&stats.energy_spect,     nGPUs*sizeof(cufftDoubleComplex *), cudaHostAllocMapped);

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

		checkCudaErrors( cudaMallocManaged((void **)&rhs_old.uh[n], sizeof(cufftDoubleComplex)*gpu.nx[n]*NY*NZ2) ); 
		checkCudaErrors( cudaMallocManaged((void **)&rhs_old.vh[n], sizeof(cufftDoubleComplex)*gpu.nx[n]*NY*NZ2) );
		checkCudaErrors( cudaMallocManaged((void **)&rhs_old.wh[n], sizeof(cufftDoubleComplex)*gpu.nx[n]*NY*NZ2) );
		checkCudaErrors( cudaMallocManaged((void **)&rhs_old.sh[n], sizeof(cufftDoubleComplex)*gpu.nx[n]*NY*NZ2) );

		checkCudaErrors( cudaMalloc((void **)&temp_advective[n],   sizeof(cufftDoubleComplex)*gpu.nx[n]*NY*NZ2) );
		checkCudaErrors( cudaMalloc((void **)&fft.temp[n], 			 	 sizeof(cufftDoubleComplex)*gpu.nx[n]*NY*NZ2) );
		checkCudaErrors( cudaMalloc((void **)&fft.temp_reorder[n], sizeof(cufftDoubleComplex)*gpu.nx[n]*NZ2) );
		
		// Statistics
		checkCudaErrors( cudaMallocManaged((void **)&stats.Vrms[n],    			sizeof(double)*(nt/n_stats+1)) );
		checkCudaErrors( cudaMallocManaged((void **)&stats.KE[n],      			sizeof(double)*(nt/n_stats+1)) );
		checkCudaErrors( cudaMallocManaged((void **)&stats.epsilon[n], 			sizeof(double)*(nt/n_stats+1)) );
		checkCudaErrors( cudaMallocManaged((void **)&stats.eta[n],     			sizeof(double)*(nt/n_stats+1)) );
		checkCudaErrors( cudaMallocManaged((void **)&stats.l[n],       			sizeof(double)*(nt/n_stats+1)) );
		checkCudaErrors( cudaMallocManaged((void **)&stats.lambda[n], 		  sizeof(double)*(nt/n_stats+1)) );
		checkCudaErrors( cudaMallocManaged((void **)&stats.chi[n],     			sizeof(double)*(nt/n_stats+1)) );
		checkCudaErrors( cudaMallocManaged((void **)&stats.area_scalar[n],  sizeof(double)*(nt/n_stats+1)) );
		checkCudaErrors( cudaMallocManaged((void **)&stats.area_tnti[n],  	sizeof(double)*(nt/n_stats+1)) );
		checkCudaErrors( cudaMallocManaged((void **)&stats.energy_spect[n], sizeof(double)*(nt/n_stats+1)*NX/2) );

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

		checkCudaErrors( cudaMemset(stats.Vrms[n], 				0, sizeof(double)*(nt/n_stats+1)) );
		checkCudaErrors( cudaMemset(stats.KE[n], 					0, sizeof(double)*(nt/n_stats+1)) );
		checkCudaErrors( cudaMemset(stats.epsilon[n], 		0, sizeof(double)*(nt/n_stats+1)) );
		checkCudaErrors( cudaMemset(stats.eta[n], 				0, sizeof(double)*(nt/n_stats+1)) );
		checkCudaErrors( cudaMemset(stats.l[n], 					0, sizeof(double)*(nt/n_stats+1)) );
		checkCudaErrors( cudaMemset(stats.lambda[n], 			0, sizeof(double)*(nt/n_stats+1)) );
		checkCudaErrors( cudaMemset(stats.chi[n], 				0, sizeof(double)*(nt/n_stats+1)) );
		checkCudaErrors( cudaMemset(stats.area_scalar[n], 0, sizeof(double)*(nt/n_stats+1)) );
		checkCudaErrors( cudaMemset(stats.area_tnti[n],	  0, sizeof(double)*(nt/n_stats+1)) );
	}

	return;
}
