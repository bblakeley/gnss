#include <stdio.h>
#include <stdlib.h>
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
	double x = -(double)LX/2 + (i + start_x)*(double)LX/NX;
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
void hpFilterKernel_mgpu(int start_y, double *k1, double *k2, double *k3, cufftDoubleComplex *fhat, double k_fil){

	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int j = blockIdx.y * blockDim.y + threadIdx.y;
	const int k = blockIdx.z * blockDim.z + threadIdx.z;
	if (( i >= NX) || ((j+start_y) >= NY) || (k >= NZ2)) return;
	const int idx = flatten( j, i, k, NY, NX, NZ2);

	double k_sq = k1[i]*k1[i] + k2[(j+start_y)]*k2[(j+start_y)] + k3[k]*k3[k];

	if( k_sq <= (k_fil*k_fil) )
	{
		fhat[idx].x = 0.0;
		fhat[idx].y = 0.0;
	}

	return;
}

void hpFilter(gpudata gpu, fftdata fft, griddata grid, fielddata vel)
{	// Filter out low wavenumbers

  double k_fil = 0.0;       // default
  
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
		hpFilterKernel_mgpu<<<gridSize, blockSize>>>(gpu.start_y[n], grid.kx[n], grid.ky[n], grid.kz[n], vel.uh[n],k_fil);
		hpFilterKernel_mgpu<<<gridSize, blockSize>>>(gpu.start_y[n], grid.kx[n], grid.ky[n], grid.kz[n], vel.vh[n],k_fil);
		hpFilterKernel_mgpu<<<gridSize, blockSize>>>(gpu.start_y[n], grid.kx[n], grid.ky[n], grid.kz[n], vel.wh[n],k_fil);
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

	// Create physical vectors in temporary memory
	double y = -(double)LY/2 + j*(double)LY/NY;

	// Initialize scalar field
	Z[idx] = 0.5 - 0.5*tanh( H/(4.0*theta)*( 2.0*fabs(y)/H - 1.0 ));

	return;
}

__global__ 
void initializeColloidKernel_mgpu(int start_x, cufftDoubleReal *C)
{	// Creates initial conditions in the physical domain
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int j = blockIdx.y * blockDim.y + threadIdx.y;
	const int k = blockIdx.z * blockDim.z + threadIdx.z;
	if (((i+start_x) >= NX) || (j >= NY) || (k >= NZ)) return;
	const int idx = flatten(i, j, k, NX, NY, 2*NZ2);	// Index local to each GPU

	// Create physical vectors in temporary memory
	double y = -(double)LY/2 + j*(double)LY/NY;

	// Initialize scalar field
	C[idx] = 0.5 - 0.5*tanh( H/(4.0*theta_c)*( 2.0*fabs(y)/H - 1.0 ));

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
		initializeColloidKernel_mgpu<<<gridSize, blockSize>>>(gpu.start_x[n], vel.c[n]);
		printf("Scalar field initialized on GPU #%d...\n",n);
	}

	return;

}

__global__ 
void unit_test_kernel_mgpu(int start_x, cufftDoubleReal *Z)
{	// Creates initial conditions in the physical domain
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int j = blockIdx.y * blockDim.y + threadIdx.y;
	const int k = blockIdx.z * blockDim.z + threadIdx.z;
	if (((i+start_x) >= NX) || (j >= NY) || (k >= NZ)) return;
	const int idx = flatten(i, j, k, NX, NY, 2*NZ2);	// Index local to each GPU

  double x = -(double)LX/2 + (i + start_x)*(double)LX/NX ;
	double y = -(double)LY/2 + j*(double)LY/NY;

	// Initialize scalar field
	Z[idx] = sin(x)*cos(y);

	return;
}

void init_unit_test(gpudata gpu, fftdata fft, fielddata vel)
{ // Initialize DNS data

	int n;
	for (n = 0; n<gpu.nGPUs; ++n){
		cudaSetDevice(n);

		const dim3 blockSize(TX, TY, TZ);
		const dim3 gridSize(divUp(gpu.nx[n], TX), divUp(NY, TY), divUp(NZ, TZ));

		unit_test_kernel_mgpu<<<gridSize, blockSize>>>(gpu.start_x[n], vel.s[n]);
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

double max_value(double **array, gpudata gpu)
{
    int i,j,k,idx,n;
    double max_val = -1.0;
    
    for (n=0; n<gpu.nGPUs; ++n){
     for (i=0; i<gpu.nx[n]; ++i){
	    for (j=0; j<NY; ++j){
	     for (k=0; k<NZ; ++k){
	      idx = k + j*2*NZ2 + i*2*NZ2*NY;
         if (fabs(array[n][idx]) > max_val)
          max_val = fabs(array[n][idx]);
       }
      }
     }
    }
    
    return max_val;
}

void random_normal(double *val, double mu, double sigma){
// function to generate a pair of random variables with a normal distribution of mean mu and std dev. sigma
  double A,B;
  
  val[0] = (double)rand()/RAND_MAX;   // Generate uniform random variable between 0 and 1
  val[1] = (double)rand()/RAND_MAX;
  
  A = sigma*sqrt(-2.0*log(val[0] ));
  B = 2.0*PI*val[1];
  
  val[0] = A*cos(B) + mu;
  val[1] = A*sin(B) + mu;

  return;
}

// Generate random, solenoidal velocity fields
void generateNoise(fftdata fft, gpudata gpu, griddata grid, fielddata h_vel, fielddata vel, fielddata rhs)
{
	int n,i,j,jj,k,idx,idx_g,idxp,nbins,bin;
	double kxmax,kymax,kzmax,kmax,dkx,dky,dkz,dk,ksq,sigma,pertrms,ke,rms;//,pertband,pertpeak
	double val[2];
	double *energy,*kvec,*kx,*ky,*kz;
	int *S;
	cufftDoubleComplex *A1,*A2,*A3;
	
	// Set up wavespace
	dkx = 2.0*PI/LX;
	dky = 2.0*PI/LY;
	dkz = 2.0*PI/LZ;
	kxmax = (NX/2-1)*dkx;
	kymax = (NY/2-1)*dky;
	kzmax = (NZ/2-1)*dkz;
	kmax = kxmax;
	if(kymax < kmax) kmax = kymax;
	if(kzmax < kmax) kmax = kzmax;
	dk = dkx;
	if(dky > dk) dk=dky;
	if(dkz > dk) dk=dkz;
	nbins = (int)(kmax/dk + 1);
	printf("kmax = %2.4f, dk = %2.4f, nbins=%d\n",kmax,dk,nbins);
	
	// Allocate arrays
	kx = (double *)malloc(sizeof(double)*NX);
	ky = (double *)malloc(sizeof(double)*NY);
	kz = (double *)malloc(sizeof(double)*NZ);
	kvec = (double *)malloc(sizeof(double)*nbins);
	energy = (double *)malloc(sizeof(double)*nbins);
	S = (int *)malloc(sizeof(int)*nbins);
	for (i=0; i<nbins; ++i) S[i] = 0;
	
	// Grab copies of wavenumbers
	checkCudaErrors( cudaMemcpyAsync(kx, grid.kx[0], sizeof(double)*NX, cudaMemcpyDefault) );
	checkCudaErrors( cudaMemcpyAsync(ky, grid.ky[0], sizeof(double)*NY, cudaMemcpyDefault) );
	checkCudaErrors( cudaMemcpyAsync(kz, grid.kz[0], sizeof(double)*NZ, cudaMemcpyDefault) );
	
	// Set energy spectrum	
	//pertpeak = 6.0;
	//pertband = 20.0;
	pertrms = 1.0;
	// Assign energy to wavenumber bins
	for(i = 0; i<nbins; ++i){
	  kvec[i] = i*dk;
	  double num = kvec[i]-pertpeak*dk;
	  double den = 2.0*pertband*dk;
	  energy[i] = exp(-num*num/(den*den));
	  //printf("kvec = %2.4f, energy = %2.4f\n",kvec[i],energy[i]);
	}
	
	// Scale kinetic energy and check result
	ke = 0.0;
	for(i = 0; i<nbins; ++i){
	  ke = ke + energy[i];
	}
	rms = sqrt(2.0*ke/3.0);
	
	ke = 0.0;
	for(i = 0; i<nbins; ++i){
	  energy[i] = energy[i]*(pertrms*pertrms)/(rms*rms);
	  ke = ke + energy[i];
	  //printf("Energy in bin #%d = %2.4f\n",i,energy[i]);
	}
	rms = sqrt(2.0*ke/3.0);
	//printf("Kinetic Energy = %2.4f, rms = %2.4f\n",ke,rms);
		
	// Count number of modes in each bin
	for (i=0; i<NX; ++i){
	  for (j=0; j<NY; ++j){
	    for (k=0; k<NZ2; ++k){
	      ksq = sqrt( kx[i]*kx[i] + ky[j]*ky[j] + kz[k]*kz[k] );
	      if(ksq < kmax){
	        bin = (int)(ksq/dk+0.5);
	        S[bin] = S[bin] + 1;
	        //printf("For ksq = %2.4f, bin = %d\n",ksq,bin);
	      }
	    }
	  }
	}
/*	
	for(i = 0; i<nbins; ++i){
	  printf("Number of modes in bin #%d = %d\n",i,S[i]);
	}
*/	
	// Allocate temporary arrays to hold random values
	A1 = (cufftDoubleComplex *)malloc(sizeof(cufftDoubleComplex)*NX*NY*NZ2);
	A2 = (cufftDoubleComplex *)malloc(sizeof(cufftDoubleComplex)*NX*NY*NZ2);
	A3 = (cufftDoubleComplex *)malloc(sizeof(cufftDoubleComplex)*NX*NY*NZ2);

	// Initialize random seed
	srand(666*37 + 37*13);
	
	// Assign complex amplitudes and phases based on energy spectrum
  for(i=0; i<NX; i++){
    for(j=0; j<NY; j++){
      for(k=0; k<NZ2; k++){
        idx = k + i*NZ2 + j*NX*NZ2;     // Transposed coordinates because... FFTs
        ksq = sqrt( kx[i]*kx[i] + ky[j]*ky[j] + kz[k]*kz[k] );
        if(ksq < kmax){
          bin = (int)(ksq/dk+0.5);
          sigma = sqrt( energy[bin]/(2.0*S[bin]*ksq*ksq) );
	          
          random_normal(val,0.0,sigma);
          A1[idx].x = -val[1];
          A1[idx].y = val[0];
	          
          random_normal(val,0.0,sigma);
          A2[idx].x = -val[1];
          A2[idx].y = val[0];
	          
          random_normal(val,0.0,sigma);
          A3[idx].x = -val[1];
          A3[idx].y = val[0];
        }
        if(i==0 && j==0 && k==0){
          A1[idx].x = 0.0;
          A1[idx].y = 0.0;
          A2[idx].x = 0.0;
          A2[idx].y = 0.0;
          A3[idx].x = 0.0;
          A3[idx].y = 0.0;
        }
      }
    }
  }
/*
  // Check energy spectrum	
  for(i = 0; i<nbins; ++i){
	  energy[i] = 0.0;
	}
	
  for(i=0; i<NX; i++){
    for(j=0; j<NY; j++){
      for(k=0; k<NZ2; k++){
        idx = k + i*NZ2 + j*NX*NZ2;
        ksq = sqrt( kx[i]*kx[i] + ky[j]*ky[j] + kz[k]*kz[k] );
        if(ksq < kmax){
          bin = (int)(ksq/dk+0.5);
          energy[bin] = energy[bin] + abs(A1[idx].x*A1[idx].y) + abs(A2[idx].x*A2[idx].y) + abs(A3[idx].x*A3[idx].y);
        }
      }
    }
  }
	    
	ke = 0.0;
	for(i = 0; i<nbins; ++i){
	  ke = ke + energy[i];
	  printf("Pre-cross product, Energy in bin #%d = %2.4f\n",i,energy[i]);
	}
	printf("Kinetic Energy = %2.4f, rms = %2.4f\n",ke,sqrt(2.0*ke/3.0));
*/	  
	// Take cross product to get incompressible velocity field
  for(i=0; i<NX; i++){
    for(j=0; j<NY; j++){
      for(k=0; k<NZ2; k++){
        idx = k + i*NZ2 + j*NX*NZ2;
        double a1x = A1[idx].x;
        double a1y = A1[idx].y;
        double a2x = A2[idx].x;
        double a2y = A2[idx].y;
        double a3x = A3[idx].x;
        double a3y = A3[idx].y;
        A1[idx].x = -( ky[j]*a3y - kz[k]*a2y );
        A1[idx].y =    ky[j]*a3x - kz[k]*a2x;
	        
        A2[idx].x = -( kz[k]*a1y - kx[i]*a3y );
        A2[idx].y =    kz[k]*a1x - kx[i]*a3x;
	        
        A3[idx].x = -( kx[i]*a2y - ky[j]*a1y );
        A3[idx].y =    kx[i]*a2x - ky[j]*a1x;
      }
    }
  }

	//Enforce conjugate symmetry
	for(i=NX/2+1; i<NX; i++){
	  for(j=NY/2+1; j<NY; j++){
	    k = 0; // Plane of symmetry
	    idx = k + i*NZ2 + j*NX*NZ2;
	    idxp = k + (NX-i)*NZ2 + (NY-j)*NX*NZ2;
	    //printf("i=%d, j=%d, kx = %2.2f, ky = %2.2f\n ip = %d, jp = %d, kx = %2.2f, ky=%2.2f\n",i,j,kx[i],ky[j],(NX-i),(NY-j),kx[(NX-i)],ky[(NY-j)]);
	    A1[idx].x =  A1[idxp].x;
	    A1[idx].y = -A1[idxp].y;
	    A2[idx].x =  A2[idxp].x;
	    A2[idx].y = -A2[idxp].y;
	    A3[idx].x =  A3[idxp].x;
	    A3[idx].y = -A3[idxp].y;
	  }
  }
  
	for(i=NX/2+1; i<NX; i++){
	  for(j=1; j<NY/2; j++){
	    k = 0; // Plane of symmetry
	    idx = k + i*NZ2 + j*NX*NZ2;
	    idxp = k + (NX-i)*NZ2 + (NY-j)*NX*NZ2;
	    //printf("i=%d, j=%d, kx = %2.2f, ky = %2.2f\n ip = %d, jp = %d, kx = %2.2f, ky=%2.2f\n",i,j,kx[i],ky[j],(NX-i),(NY-j),kx[(NX-i)],ky[(NY-j)]);
	    A1[idx].x =  A1[idxp].x;
	    A1[idx].y = -A1[idxp].y;
	    A2[idx].x =  A2[idxp].x;
	    A2[idx].y = -A2[idxp].y;
	    A3[idx].x =  A3[idxp].x;
	    A3[idx].y = -A3[idxp].y;
	  }
  }
  
	for(i=NX/2+1; i<NX; i++){
	  j = 0;
    k = 0; // Plane of symmetry
    idx = k + i*NZ2 + j*NX*NZ2;
    idxp = k + (NX-i)*NZ2 + j*NX*NZ2;
    //printf("i=%d, j=%d, kx = %2.2f, ky = %2.2f\n ip = %d, jp = %d, kx = %2.2f, ky=%2.2f\n",i,j,kx[i],ky[j],(NX-i),j,kx[(NX-i)],ky[j]);
    A1[idx].x =  A1[idxp].x;
    A1[idx].y = -A1[idxp].y;
    A2[idx].x =  A2[idxp].x;
    A2[idx].y = -A2[idxp].y;
    A3[idx].x =  A3[idxp].x;
    A3[idx].y = -A3[idxp].y;
  }	  
	for(j=NY/2+1; j<NY; j++){
	  i = 0;
    k = 0; // Plane of symmetry
    idx = k + i*NZ2 + j*NX*NZ2;
	  idxp = k + i*NZ2 + (NY-j)*NX*NZ2;
    //printf("i=%d, j=%d, kx = %2.2f, ky = %2.2f\n ip = %d, jp = %d, kx = %2.2f, ky=%2.2f\n",i,j,kx[i],ky[j],i,(NY-j),kx[i],ky[(NY-j)]);
    A1[idx].x =  A1[idxp].x;
    A1[idx].y = -A1[idxp].y;
    A2[idx].x =  A2[idxp].x;
    A2[idx].y = -A2[idxp].y;
    A3[idx].x =  A3[idxp].x;
    A3[idx].y = -A3[idxp].y;
  }	  
  
  // Check energy spectrum	
  for(i = 0; i<nbins; ++i){
	  energy[i] = 0.0;
	}
	
  for(i=0; i<NX; i++){
    for(j=0; j<NY; j++){
      for(k=0; k<NZ2; k++){
        idx = k + i*NZ2 + j*NX*NZ2;
        ksq = sqrt( kx[i]*kx[i] + ky[j]*ky[j] + kz[k]*kz[k] );
        if(ksq < kmax){
          bin = (int)(ksq+0.5)/dk;
          energy[bin] = energy[bin] + abs(A1[idx].x*A1[idx].y) + abs(A2[idx].x*A2[idx].y) + abs(A3[idx].x*A3[idx].y);
        }
      }
    }
  }

	ke = 0.0;
	for(i = 0; i<nbins; ++i){
	  ke = ke + energy[i];
	  //printf("Post cross-product, Energy in bin #%d = %2.4f\n",i,energy[i]);
	}
	printf("Kinetic Energy = %2.4f, rms = %2.4f\n",ke,sqrt(2.0*ke/3.0));


		    
	// Translate temporary global storage to GPU storage
	for(n=0; n<gpu.nGPUs; ++n){
    for(i=0; i<NX; i++){
      for(j=0; j<gpu.ny[n]; j++){
        for(k=0; k<NZ2; k++){
        jj = j + gpu.start_y[n];
        idx = k + i*NZ2 + j*NX*NZ2;
        idx_g = k + i*NZ2 + jj*NX*NZ2;
        h_vel.uh[n][idx].x = A1[idx_g].x*NN; // Scale by NN to counteract scaling in FFT
        h_vel.uh[n][idx].y = A1[idx_g].y*NN;
        
        h_vel.vh[n][idx].x = A2[idx_g].x*NN;
        h_vel.vh[n][idx].y = A2[idx_g].y*NN;
        
        h_vel.wh[n][idx].x = A3[idx_g].x*NN;
        h_vel.wh[n][idx].y = A3[idx_g].y*NN;
        }
      }
    }
  }
          	
	// Cleanup
	free(kvec);
	free(energy);
	free(S);
	free(kx);
	free(ky);
	free(kz);
	
	free(A1);
	free(A2);
	free(A3);

	// Copy random field to GPU
  for(n=0; n<gpu.nGPUs; ++n){
		cudaSetDevice(n);
		cudaDeviceSynchronize();
		checkCudaErrors( cudaMemcpyAsync(rhs.uh[n], h_vel.uh[n], sizeof(cufftDoubleComplex)*gpu.nx[n]*NY*NZ2, cudaMemcpyDefault) );
		checkCudaErrors( cudaMemcpyAsync(rhs.vh[n], h_vel.vh[n], sizeof(cufftDoubleComplex)*gpu.nx[n]*NY*NZ2, cudaMemcpyDefault) );
		checkCudaErrors( cudaMemcpyAsync(rhs.wh[n], h_vel.wh[n], sizeof(cufftDoubleComplex)*gpu.nx[n]*NY*NZ2, cudaMemcpyDefault) );
	}
	
	// Transform from Fourier Space to physical space for normalization
	inverseTransform(fft, gpu, rhs.uh);
	inverseTransform(fft, gpu, rhs.vh);
	inverseTransform(fft, gpu, rhs.wh);
	
	return;
}

// Initializing temporal jet and adding random noise to velocity field
void initializeJet(fftdata fft, gpudata gpu, griddata grid, fielddata h_vel, fielddata vel, fielddata rhs)
{
	int n;
	
	// Generate psuedo-random velocity field
	generateNoise(fft, gpu, grid, h_vel, vel, rhs);

	// Initialize smooth jet velocity field (hyperbolic tangent profile from da Silva and Pereira)
	initializeVelocity(gpu, vel);

	// Superimpose isotropic noise on top of jet velocity initialization
	for (n = 0; n<gpu.nGPUs; ++n){
		cudaSetDevice(n);

		const dim3 blockSize(TX, TY, TZ);
		const dim3 gridSize(divUp(gpu.nx[n], TX), divUp(NY, TY), divUp(NZ, TZ));

		velocitySuperpositionKernel_mgpu<<<gridSize, blockSize>>>(gpu.start_x[n], vel.u[n], vel.v[n], vel.w[n], rhs.u[n], rhs.v[n], rhs.w[n], pert_amp);
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
void waveNumber_kernel(int n, double l, double *waveNum)
{   // Creates the wavenumber vectors used in Fourier space
	const int i = blockIdx.x*blockDim.x + threadIdx.x;

	if (i >= n) return;

	if (i < n/2)
		waveNum[i] = (2*PI/l)*(double)i;
	else
		waveNum[i] = (2*PI/l)*( (double)i - n );

	return;
}

void initializeWaveNumbers(gpudata gpu, griddata grid)
{    // Initialize wavenumbers in Fourier space

	int n;
	for (n = 0; n<gpu.nGPUs; ++n){
		cudaSetDevice(n);
		waveNumber_kernel<<<divUp(NX,TX), TX>>>(NX,LX,grid.kx[n]);
		waveNumber_kernel<<<divUp(NY,TX), TX>>>(NY,LY,grid.ky[n]);
		waveNumber_kernel<<<divUp(NZ,TX), TX>>>(NZ,LZ,grid.kz[n]);
	}

	printf("Wave domain setup complete..\n");

	return;
}
