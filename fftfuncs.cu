#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cufft.h>
#include <cuComplex.h>
#include <helper_cuda.h>
#include "dnsparams.h"
#include "cudafuncs.h"
#include "fftfuncs.h"

//==============================================================================
// Transpose algorithm
//==============================================================================
__global__ 
void organizeData(cufftDoubleComplex *in, cufftDoubleComplex *out, int N, int j)
{// Function to grab non-contiguous chunks of data and make them contiguous

	const int k = blockIdx.x * blockDim.x + threadIdx.x;
	if(k >= NZ2) return;

	for(int i=0; i<N; ++i){

		// printf("For thread %d, indexing begins at local index of %d, which maps to temp at location %d\n", k, (k+ NZ*j), k);
		out[k + i*NZ2] = in[k + NZ2*j + i*NY*NZ2];

	}

	return;
}

__global__ 
void organizeData_coalesced(cufftDoubleComplex *in, cufftDoubleComplex *out, int N, int j)
{// Function to grab non-contiguous chunks of data and make them contiguous

	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= N) return;

	for(int k=0; k<NZ2; ++k){

		// printf("For thread %d, indexing begins at local index of %d, which maps to temp at location %d\n", k, (k+ NZ*j), k);
		out[k + i*NZ2] = in[k + NZ2*j + i*NY*NZ2];

	}

	return;
}

void transpose_xy_mgpu(gpuinfo gpu, cufftDoubleComplex **src, cufftDoubleComplex **dst, cufftDoubleComplex **temp)
{   // Transpose x and y directions (for a z-contiguous 1d array distributed across multiple GPUs)
	// This function loops through GPUs to do the transpose. Requires extra conversion to calculate the local index at the source location.
	// printf("Taking Transpose...\n");

	int n, j, local_idx_dst, dstNum;

	// for (n=0; n<gpu.nGPUs; ++n){
	// 	printf("gpu.ny(%d) = %d \n", n, gpu.ny[n]);
	// 	printf("gpu.start_y(%d) = %d \n", n, gpu.start_y[n]);
	// }

   
	for(j=0; j<NY; ++j){
		for(n=0; n<gpu.nGPUs; ++n){
			cudaSetDevice(n); 

			// Determine which GPU to send data to based on y-index, j
			dstNum = (j*gpu.nGPUs)/NY;
			// printf("dstNum = %d\n",dstNum);

			// Open kernel that grabs all data 
			// organizeData<<<divUp(NZ2,TX), TX>>>(src[n], temp[n], NX/gpu.nGPUs, j);
			// organizeData_coalesced<<<divUp(NX/gpu.nGPUs,TX), TX>>>(src[n], temp[n], gpu.nx[n], j);
			organizeData_coalesced<<<divUp(gpu.nx[n],TX), TX>>>(src[n], temp[n], gpu.nx[n], j);

			// local_idx_dst = n*NX/gpu.nGPUs*NZ2 + (j - dstNum*NY/gpu.nGPUs)*NZ2*NX;
			local_idx_dst = gpu.start_x[n]*NZ2 + (j - gpu.start_y[dstNum])*NZ2*NX;
			// printf("For j = %d, GPU = %d, the local idx at destination = %d \n",j,gpu.gpunum[n], local_idx_dst);

			checkCudaErrors( cudaMemcpyAsync(&dst[dstNum][local_idx_dst], temp[n], sizeof(cufftDoubleComplex)*NZ2*gpu.nx[n], cudaMemcpyDefault) );
			// printf("Offending values: dstNum = %d, local index = %d \n",dstNum, local_idx_dst);
		}
	}

	return;
}

//==============================================================================
// FFT functions
//==============================================================================

void plan2dFFT(gpuinfo gpu, fftinfo fft){
// This function plans a 2-dimensional FFT to operate on the X and Y directions (assumes X-direction is contiguous in memory)
	int result;

	int n;
	for(n = 0; n<gpu.nGPUs; ++n){
	  cudaSetDevice(n);

	  //Create plan for 2-D cuFFT, set cuFFT parameters
	  int rank = 2;
	  int size[] = {NY,NZ};           
	  int inembed[] = {NY,2*NZ2};         // inembed measures distance between dimensions of data
	  int onembed[] = {NY,NZ2};     // Uses half the domain for a R2C transform
	  int istride = 1;                        // istride is distance between consecutive elements
	  int ostride = 1;
	  int idist = NY*2*NZ2;                      // idist is the total length of one signal
	  int odist = NY*NZ2;
	  // int batch = NX_per_GPU[n];                        // # of 2D FFTs to perform
	  int batch = gpu.nx[n];                        // # of 2D FFTs to perform

	  // Create empty plan handles
	  cufftCreate(&fft.p2d[n]);
	  cufftCreate(&fft.invp2d[n]);

	  // Disable auto allocation of workspace memory for cuFFT plans
	  result = cufftSetAutoAllocation(fft.p2d[n], 0);
	  if ( result != CUFFT_SUCCESS){
      printf("CUFFT error: cufftSetAutoAllocation failed on line %d, Error code %d\n", __LINE__, result);
	  return; }
	  result = cufftSetAutoAllocation(fft.invp2d[n], 0);
	  if ( result != CUFFT_SUCCESS){
      printf("CUFFT error: cufftSetAutoAllocation failed on line %d, Error code %d\n", __LINE__, result);
	  return; }

	  // Plan Forward 2DFFT
	  result = cufftMakePlanMany(fft.p2d[n], rank, size, inembed, istride, idist, onembed, ostride, odist, CUFFT_D2Z, batch, &fft.wsize_f[n]);
	  if ( result != CUFFT_SUCCESS){
      fprintf(stderr, "CUFFT error: cufftPlanforward 2D failed");
      printf(", Error code %d\n", result);
	  return; 
	  }

	  // Plan inverse 2DFFT
	  result = cufftMakePlanMany(fft.invp2d[n], rank, size, onembed, ostride, odist, inembed, istride, idist, CUFFT_Z2D, batch, &fft.wsize_i[n]);
	  if ( result != CUFFT_SUCCESS){
      fprintf(stderr, "CUFFT error: cufftPlanforward 2D failed");
      printf(", Error code %d\n", result);
	  return; 
	  }

	  printf("The workspace size required for the forward transform is %lu.\n", fft.wsize_f[n]);
	  // printf("The workspace size required for the inverse transform is %lu.\n", fft.wsize_i[n]);

	  // Assuming that both workspaces are the same size (seems to be generally true), then the two workspaces can share an allocation - need to use maximum value here
	  // Allocate workspace memory
	  checkCudaErrors( cudaMalloc(&fft.wspace[n], fft.wsize_f[n]) );

	  // Set cuFFT to use allocated workspace memory
	  result = cufftSetWorkArea(fft.p2d[n], fft.wspace[n]);
	  if ( result != CUFFT_SUCCESS){
    	printf("CUFFT error: ExecD2Z failed on line %d, Error code %d\n", __LINE__, result);
	  return; }
	  result = cufftSetWorkArea(fft.invp2d[n], fft.wspace[n]);
	  if ( result != CUFFT_SUCCESS){
      printf("CUFFT error: ExecD2Z failed on line %d, Error code %d\n", __LINE__, result);
	  return; }    

	}

	return;
}

void plan1dFFT(int nGPUs, fftinfo fft){
// This function plans a 1-dimensional FFT to operate on the Z direction (assuming Z-direction is contiguous in memory)
    int result;

    int n;
    for(n = 0; n<nGPUs; ++n){
        cudaSetDevice(n);
        //Create plan for cuFFT, set cuFFT parameters
        int rank = 1;               // Dimensionality of the FFT - constant at rank 1
        int size[] = {NX};          // size of each rank
        int inembed[] = {0};            // inembed measures distance between dimensions of data
        int onembed[] = {0};       // For complex to complex transform, input and output data have same dimensions
        int istride = NZ2;                        // istride is distance between consecutive elements
        int ostride = NZ2;
        int idist = 1;                     // idist is the total length of one signal
        int odist = 1;
        int batch = NZ2;                      // # of 1D FFTs to perform (assuming data has been transformed previously in the Z-Y directions)

        // Plan Forward 1DFFT
        result = cufftPlanMany(&fft.p1d[n], rank, size, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2Z, batch);
        if ( result != CUFFT_SUCCESS){
            fprintf(stderr, "CUFFT error: cufftPlanforward failed");
        return; 
        }
    }
    
    return;
}

void Execute1DFFT_Forward(cufftHandle plan, int NY_per_GPU, cufftDoubleComplex *f, cufftDoubleComplex *fhat)
{

	cufftResult result;
	// Loop through each slab in the Y-direction
	// Perform forward FFT
	for(int i=0; i<NY_per_GPU; ++i){
		result = cufftExecZ2Z(plan, &f[i*NZ2*NX], &fhat[i*NZ2*NX], CUFFT_FORWARD);
		if (  result != CUFFT_SUCCESS){
			fprintf(stderr, "CUFFT error: ExecZ2Z failed, error code %d\n",(int)result);
		return; 
		}       
	}

	return;
}

void Execute1DFFT_Inverse(cufftHandle plan, int NY_per_GPU, cufftDoubleComplex *fhat, cufftDoubleComplex *f)
{
	cufftResult result;

	// Loop through each slab in the Y-direction
	// Perform forward FFT
	for(int i=0; i<NY_per_GPU; ++i){
		result = cufftExecZ2Z(plan, &fhat[i*NZ2*NX], &f[i*NZ2*NX], CUFFT_INVERSE);
		if (  result != CUFFT_SUCCESS){
			fprintf(stderr, "CUFFT error: ExecZ2Z failed, error code %d\n",(int)result);
		return; 
		}       
	}

	return;
}


void forwardTransform(fftinfo fft, gpuinfo gpu, cufftDoubleReal **f )
{ // Transform from physical to wave domain

	int RESULT, n;

	// Take FFT in Z and Y directions
	for(n = 0; n<gpu.nGPUs; ++n){
		cudaSetDevice(n);

		RESULT = cufftExecD2Z(fft.p2d[n], f[n], (cufftDoubleComplex *)f[n]);
		if ( RESULT != CUFFT_SUCCESS){
			printf("CUFFT error: ExecD2Z failed on line %d, Error code %d\n", __LINE__, RESULT);
		return; }
		// printf("Taking 2D forward FFT on GPU #%2d\n",n);
	}

	// Transpose X and Y dimensions
	transpose_xy_mgpu(gpu, (cufftDoubleComplex **)f, fft.temp, fft.temp_reorder);

	// Take FFT in X direction (which has been transposed to what used to be the Y dimension)
	for(n = 0; n<gpu.nGPUs; ++n){
		cudaSetDevice(n);

		Execute1DFFT_Forward(fft.p1d[n], gpu.ny[n], fft.temp[n], (cufftDoubleComplex *)f[n]);
		// printf("Taking 1D forward FFT on GPU #%2d\n",n);
	}

	// Results remain in transposed coordinates

	// printf("Forward Transform Completed...\n");

	return;
}

void inverseTransform(fftinfo fft, gpuinfo gpu, cufftDoubleComplex **f)
{ // Transform variables from wavespace to the physical domain 
	int RESULT, n;

	// Data starts in transposed coordinates, x,y flipped

	// Take FFT in X direction (which has been transposed to what used to be the Y dimension)
	for(n = 0; n<gpu.nGPUs; ++n){
		cudaSetDevice(n);
		Execute1DFFT_Inverse(fft.p1d[n], gpu.ny[n], f[n], fft.temp[n]);
		// printf("Taking 1D inverse FFT on GPU #%2d\n",n);
	}

	// Transpose X and Y directions
	transpose_xy_mgpu(gpu, fft.temp, f, fft.temp_reorder);

	for(n = 0; n<gpu.nGPUs; ++n){
		cudaSetDevice(n);
		// Take inverse FFT in Z and Y direction
		RESULT = cufftExecZ2D(fft.invp2d[n], f[n], (cufftDoubleReal *)f[n]);
		if ( RESULT != CUFFT_SUCCESS){
			printf("CUFFT error: ExecD2Z failed on line %d, Error code %d\n", __LINE__, RESULT);
		return; }
		// printf("Taking 2D inverse FFT on GPU #%2d\n",n);
	}

	for(n = 0; n<gpu.nGPUs; ++n){
		cudaSetDevice(n);
		const dim3 blockSize(TX, TY, TZ);
		const dim3 gridSize(divUp(gpu.nx[n], TX), divUp(NY, TY), divUp(NZ, TZ));

		scaleKernel_mgpu<<<gridSize, blockSize>>>(gpu.start_x[n], (cufftDoubleReal *)f[n]);
	}

	// printf("Scaled Inverse Transform Completed...\n");

	return;
}
