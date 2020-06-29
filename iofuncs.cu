#include <stdlib.h>
#include <stdio.h>
#include <complex.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <cufft.h>
#include <cuComplex.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>


#include "dnsparams.h"
#include "iofuncs.h"
#include "fftfuncs.h"

//============================================================================================
// Print to screen
//============================================================================================
void displayDeviceProps(int numGPUs){
	int i, driverVersion = 0, runtimeVersion = 0;

	for( i = 0; i<numGPUs; ++i)
	{
		cudaSetDevice(i);

		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, i);
		printf("  Device name: %s\n", deviceProp.name);

		cudaDriverGetVersion(&driverVersion);
		cudaRuntimeGetVersion(&runtimeVersion);
		printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n", driverVersion/1000, (driverVersion%100)/10, runtimeVersion/1000, (runtimeVersion%100)/10);
		printf("  CUDA Capability Major/Minor version number:    %d.%d\n", deviceProp.major, deviceProp.minor);
	
		char msg[256];
		SPRINTF(msg, "  Total amount of global memory:                 %.0f MBytes \n",
				(float)deviceProp.totalGlobalMem/1048576.0f);
		printf("%s", msg);

		printf("  (%2d) Multiprocessors, (%3d) CUDA Cores/MP:     %d CUDA Cores\n",
			   deviceProp.multiProcessorCount,
			   _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
			   _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount);

		printf("\n");
	}

	return;
}


void printTurbStats(int c, double steptime, statistics stats)
{

	if(c==0)
		printf("\n Entering time-stepping loop...\n");
	// if(c%20==0)			// Print new header every few timesteps
		printf(" iter |   u'  |   k   |  eps  |   l   |  eta  | lambda | chi  | |omega| | time \n"
			"-----------------------------------------------------------\n");
	// Print statistics to screen
	printf(" %d  | %2.3f | %2.3f | %2.3f | %2.3f | %2.3f | %2.3f | %2.3f | % 2.3f | %2.3f  \n",
			c*n_stats, stats.Vrms, stats.KE, stats.epsilon, stats.l, stats.eta, stats.lambda, stats.chi, stats.omega_z, steptime/1000);

	return;
}

void printIterTime(int c, double steptime)
{
	// Print iteration time to screen
	printf(" %d  |       |       |       |       |       |       |       |       |% 2.3f \n",
			c,steptime/1000);

	return;
}

//============================================================================================
// Write to file
//============================================================================================

//void make_name(

void writeYprofs(const int c, const char* name, double *data)
{
  char title [256];
	FILE *out;
	char folder [256];

	snprintf(folder, sizeof(folder), sim_name);
	snprintf(title, sizeof(title), "%s%sYprofs/%s.%i", rootdir, folder, name, c);
	printf("Writing data to %s \n", title);
	out = fopen(title, "wb");
	
	fwrite(data, sizeof(double), NY, out);
	
	fclose(out);

  return;
}

void saveYprofs(const int c, profile data)
{ // Save mean profs to file
  struct stat st = {0};
  char title[256];
  char folder [256];
	
	snprintf(folder, sizeof(folder), sim_name);
	if(c==0){  // Create directory for statistics if one doesn't already exist
	  snprintf(title, sizeof(title), "%s%s%s", rootdir, folder, "Yprofs/");
    if (stat(title, &st) == -1) {  
      mkdir(title, 0700);
    }
  }
  writeYprofs(c, "u_mean", data.u[0]);
  writeYprofs(c, "v_mean", data.v[0]);
  writeYprofs(c, "w_mean", data.w[0]);
  writeYprofs(c, "s_mean", data.s[0]);
  writeYprofs(c, "c_mean", data.c[0]);
  
  return;
}

void writeDouble(double v, FILE *f)  {
	fwrite((void*)(&v), sizeof(v), 1, f);

	return;
}

void writeStats(const int c, const char* name, double in) {
	char title[256];
	FILE *out;
	char folder [256];
	
	snprintf(folder, sizeof(folder), sim_name);	
	snprintf(title, sizeof(title), "%s%sstats/%s", rootdir, folder, name);
	//printf("Writing data to %s \n", title);
	if(c==0){ // First timestep, create new file
	  out = fopen(title, "wb");
	}
	else{ // append current timestep data to statistics file
	  out = fopen(title, "ab");
	}
	
	writeDouble(in, out);
		
	fclose(out);
}

void saveStatsData(const int c, statistics stats)
{
  struct stat st = {0};
  char title[256];
	char folder [256];
	
	snprintf(folder, sizeof(folder), sim_name);  
	if(c==0){  // Create directory for statistics if one doesn't already exist
	  snprintf(title, sizeof(title), "%s%s%s", rootdir, folder, "stats/");
    if (stat(title, &st) == -1) {  
      mkdir(title, 0700);
    }
  }

	// Save statistics data
	writeStats(c, "Vrms",    stats.Vrms);
	writeStats(c, "epsilon", stats.epsilon);
	writeStats(c, "eta",     stats.eta);
	writeStats(c, "KE",      stats.KE);
	writeStats(c, "lambda",  stats.lambda);
	writeStats(c, "l",       stats.l);
	writeStats(c, "chi",     stats.chi);
	writeStats(c, "omega"  , stats.omega);
	writeStats(c, "omega_x", stats.omega_x);
	writeStats(c, "omega_y", stats.omega_y);
	writeStats(c, "omega_z", stats.omega_z);
	
	// Loop required to write statistics that depend on a second variable
	//for(i=0;i<64;++i){
	//	writeStats(1, "area_z", stats.area_scalar[i]);
	//  writeStats(1, "area_omega" , stats.area_omega[i]);
	//}
	
  snprintf(title, sizeof(title), "%s%s%s", rootdir, folder, "stats/");
	printf("Statistics data written to %s \n", title);

	return;
}

void writexyfields( gpudata gpu, const int iter, const char var, double **in, const int zplane ) 
{
	int i, j, n, idx;
	char title[256];
	char folder [256];
	
	snprintf(folder, sizeof(folder), sim_name);
	snprintf(title, sizeof(title), "%s%svis/%c%s.%i", rootdir, folder, var, "_xy", iter);
	printf("Saving data to %s \n", title);
	FILE *out = fopen(title, "wb");
	writeDouble(sizeof(double) * NX*NY, out);
	for (n = 0; n < gpu.nGPUs; ++n){
		for (i = 0; i < gpu.nx[n]; ++i){
			for (j = 0; j < NY; ++j){
				idx = zplane + 2*NZ2*j + 2*NZ2*NY*i;		// Using padded index for in-place FFT
				writeDouble(in[n][idx], out);
			}
		}			
	}

	fclose(out);

	return;
}

void writexzfields( gpudata gpu, const int iter, const char var, double **in, const int yplane ) 
{
	int i, k, n, idx;
	char title[256];
	char folder [256];
	
	snprintf(folder, sizeof(folder), sim_name);	
	snprintf(title, sizeof(title), "%s%svis/%c%s.%i", rootdir, folder, var, "_xz", iter);
	printf("Saving data to %s \n", title);
	FILE *out = fopen(title, "wb");
	writeDouble(sizeof(double) * NX*NZ, out);
	// writelu(sizeof(double) * NX*NY*NZ, out);
	k=0;
	for (n = 0; n < gpu.nGPUs; ++n){
		for (i = 0; i < gpu.nx[n]; ++i){
			idx = k + 2*NZ2*yplane + 2*NZ2*NY*i;		// Using padded index for in-place FFT
			fwrite((void *)&in[n][idx], sizeof(double), NZ, out);  // Write each k vector at once
		}			
	}

	fclose(out);

	return;
}

void writeyzfields( gpudata gpu, const int iter, const char var, double **in, const int xplane ) 
{
	int j, k, n, idx;
	char title[256];
	char folder [256];
	
	snprintf(folder, sizeof(folder), sim_name);
	snprintf(title, sizeof(title), "%s%svis/%c%s.%i", rootdir, folder, var, "_yz", iter);
	printf("Saving data to %s \n", title);
	FILE *out = fopen(title, "wb");
	writeDouble(sizeof(double) * NY*NZ, out);
	// writelu(sizeof(double) * NX*NY*NZ, out);
	k=0;
	for (n = 0; n < gpu.nGPUs; ++n){
		for (j = 0; j < NY; ++j){
			idx = k + 2*NZ2*j + 2*NZ2*NY*xplane;		// Using padded index for in-place FFT
			fwrite((void *)&in[n][idx], sizeof(double), NZ, out);  // Write each k vector at once
		}
	}

	fclose(out);

	return;
}

void save2Dfields(int c, fftdata fft, gpudata gpu, fielddata h_vel, fielddata vel)
{
	int n;
	char title[256];
	struct stat st = {0};
	char folder [256];
	
	snprintf(folder, sizeof(folder), sim_name);
	if(c==0){ // Create new directory for visualization data if one doesn't already exist
	  snprintf(title, sizeof(title), "%s%s%s", rootdir, folder, "vis/");
    if (stat(title, &st) == -1) {
      mkdir(title, 0700);
    }
    
    // Copy data to host   
	  for(n=0; n<gpu.nGPUs; ++n){
		  cudaSetDevice(n);
		  checkCudaErrors( cudaMemcpyAsync(h_vel.u[n], vel.u[n], sizeof(complex double)*gpu.nx[n]*NY*NZ2, cudaMemcpyDefault) );
		  checkCudaErrors( cudaMemcpyAsync(h_vel.v[n], vel.v[n], sizeof(complex double)*gpu.nx[n]*NY*NZ2, cudaMemcpyDefault) );
		  checkCudaErrors( cudaMemcpyAsync(h_vel.w[n], vel.w[n], sizeof(complex double)*gpu.nx[n]*NY*NZ2, cudaMemcpyDefault) );
		  checkCudaErrors( cudaMemcpyAsync(h_vel.s[n], vel.s[n], sizeof(complex double)*gpu.nx[n]*NY*NZ2, cudaMemcpyDefault) );
		  checkCudaErrors( cudaMemcpyAsync(h_vel.c[n], vel.c[n], sizeof(complex double)*gpu.nx[n]*NY*NZ2, cudaMemcpyDefault) );
	  }
    
    writexyfields( gpu, c, 'u', h_vel.u, NZ/2);
    writexyfields( gpu, c, 'v', h_vel.v, NZ/2);
    writexyfields( gpu, c, 'w', h_vel.w, NZ/2);
    writexyfields( gpu, c, 'z', h_vel.s, NZ/2);
    writexyfields( gpu, c, 'c', h_vel.c, NZ/2);
    
  }
  else{
		// Inverse Fourier Transform the velocity back to physical space for saving to file.
		inverseTransform(fft, gpu, vel.uh);
		inverseTransform(fft, gpu, vel.vh);
		inverseTransform(fft, gpu, vel.wh);
		inverseTransform(fft, gpu, vel.sh);
		inverseTransform(fft, gpu, vel.ch);

		// Copy data to host   
		for(n=0; n<gpu.nGPUs; ++n){
			cudaSetDevice(n);
			checkCudaErrors( cudaMemcpyAsync(h_vel.u[n], vel.u[n], sizeof(complex double)*gpu.nx[n]*NY*NZ2, cudaMemcpyDefault) );
			checkCudaErrors( cudaMemcpyAsync(h_vel.v[n], vel.v[n], sizeof(complex double)*gpu.nx[n]*NY*NZ2, cudaMemcpyDefault) );
			checkCudaErrors( cudaMemcpyAsync(h_vel.w[n], vel.w[n], sizeof(complex double)*gpu.nx[n]*NY*NZ2, cudaMemcpyDefault) );
			checkCudaErrors( cudaMemcpyAsync(h_vel.s[n], vel.s[n], sizeof(complex double)*gpu.nx[n]*NY*NZ2, cudaMemcpyDefault) );
			checkCudaErrors( cudaMemcpyAsync(h_vel.c[n], vel.c[n], sizeof(complex double)*gpu.nx[n]*NY*NZ2, cudaMemcpyDefault) );
		}

		// Write data to file
	  writexyfields(gpu, c, 'u', h_vel.u, NZ/2);
		writexyfields(gpu, c, 'v', h_vel.v, NZ/2);
		writexyfields(gpu, c, 'w', h_vel.w, NZ/2);
		writexyfields(gpu, c, 'z', h_vel.s, NZ/2);
		writexyfields(gpu, c, 'c', h_vel.c, NZ/2);

		// Transform fields back to fourier space for timestepping
		forwardTransform(fft, gpu, vel.u);
		forwardTransform(fft, gpu, vel.v);
		forwardTransform(fft, gpu, vel.w);
		forwardTransform(fft, gpu, vel.s);
		forwardTransform(fft, gpu, vel.c);
	}

	return;
}

void write3Dfields_mgpu(gpudata gpu, const int iter, const char var, double **in ) 
{
	int i, j, k, n, idx;
	char title[256];
	char folder [256];
	
	snprintf(folder, sizeof(folder), sim_name);
	snprintf(title, sizeof(title), "%s%s%c.%i", rootdir, folder, var, iter);
	printf("Saving data to %s \n", title);
	FILE *out = fopen(title, "wb");
	writeDouble(sizeof(double) * NX*NY*NZ, out);
	// writelu(sizeof(double) * NX*NY*NZ, out);
	k=0;
	for (n = 0; n < gpu.nGPUs; ++n){
		for (i = 0; i < gpu.nx[n]; ++i){
			for (j = 0; j < NY; ++j){
				// for (k = 0; k < NZ; ++k){
					idx = k + 2*NZ2*j + 2*NZ2*NY*i;		// Using padded index for in-place FFT
					fwrite((void *)&in[n][idx], sizeof(double), NZ, out);  // Write each k vector at once
					//writeDouble(in[n][idx], out);
				//}
			}
		}			
	}


	fclose(out);

	return;
}

void save3Dfields(int c, fftdata fft, gpudata gpu, fielddata h_vel, fielddata vel){
	int n;
	struct stat st = {0};
	char title [256];
	char folder [256];
	
	snprintf(folder, sizeof(folder), sim_name);
	snprintf(title, sizeof(title), "%s%s", rootdir, folder);
	if(c==0){
	
    if (stat(title, &st) == -1) {  // Create root directory for DNS data if one doesn't already exist
      mkdir(title, 0700);
    }
	
		printf("Saving initial data...\n");
		for(n=0; n<gpu.nGPUs; ++n){
			cudaSetDevice(n);
			checkCudaErrors( cudaMemcpyAsync(h_vel.u[n], vel.u[n], sizeof(complex double)*gpu.nx[n]*NY*NZ2, cudaMemcpyDefault) );
			checkCudaErrors( cudaMemcpyAsync(h_vel.v[n], vel.v[n], sizeof(complex double)*gpu.nx[n]*NY*NZ2, cudaMemcpyDefault) );
			checkCudaErrors( cudaMemcpyAsync(h_vel.w[n], vel.w[n], sizeof(complex double)*gpu.nx[n]*NY*NZ2, cudaMemcpyDefault) );
			checkCudaErrors( cudaMemcpyAsync(h_vel.s[n], vel.s[n], sizeof(complex double)*gpu.nx[n]*NY*NZ2, cudaMemcpyDefault) );
			checkCudaErrors( cudaMemcpyAsync(h_vel.c[n], vel.c[n], sizeof(complex double)*gpu.nx[n]*NY*NZ2, cudaMemcpyDefault) );
		}

		// Write data to file
	  write3Dfields_mgpu(gpu, 0, 'u', h_vel.u);
		write3Dfields_mgpu(gpu, 0, 'v', h_vel.v);
		write3Dfields_mgpu(gpu, 0, 'w', h_vel.w);
		write3Dfields_mgpu(gpu, 0, 'z', h_vel.s);	
		write3Dfields_mgpu(gpu, 0, 'c', h_vel.c);		

		return;
	}

	else{
		// Inverse Fourier Transform the velocity back to physical space for saving to file.
		inverseTransform(fft, gpu, vel.uh);
		inverseTransform(fft, gpu, vel.vh);
		inverseTransform(fft, gpu, vel.wh);
		inverseTransform(fft, gpu, vel.sh);
		inverseTransform(fft, gpu, vel.ch);

		// Copy data to host   
		printf( "Timestep %i Complete. . .\n", c );
		for(n=0; n<gpu.nGPUs; ++n){
			cudaSetDevice(n);
			cudaDeviceSynchronize();
			checkCudaErrors( cudaMemcpyAsync(h_vel.u[n], vel.u[n], sizeof(complex double)*gpu.nx[n]*NY*NZ2, cudaMemcpyDefault) );
			checkCudaErrors( cudaMemcpyAsync(h_vel.v[n], vel.v[n], sizeof(complex double)*gpu.nx[n]*NY*NZ2, cudaMemcpyDefault) );
			checkCudaErrors( cudaMemcpyAsync(h_vel.w[n], vel.w[n], sizeof(complex double)*gpu.nx[n]*NY*NZ2, cudaMemcpyDefault) );
			checkCudaErrors( cudaMemcpyAsync(h_vel.s[n], vel.s[n], sizeof(complex double)*gpu.nx[n]*NY*NZ2, cudaMemcpyDefault) );
			checkCudaErrors( cudaMemcpyAsync(h_vel.c[n], vel.c[n], sizeof(complex double)*gpu.nx[n]*NY*NZ2, cudaMemcpyDefault) );
		}

		// Write data to file
	  write3Dfields_mgpu(gpu, c, 'u', h_vel.u);
		write3Dfields_mgpu(gpu, c, 'v', h_vel.v);
		write3Dfields_mgpu(gpu, c, 'w', h_vel.w);
		write3Dfields_mgpu(gpu, c, 'z', h_vel.s);
		write3Dfields_mgpu(gpu, c, 'c', h_vel.c);

		// Transform fields back to fourier space for timestepping
		forwardTransform(fft, gpu, vel.u);
		forwardTransform(fft, gpu, vel.v);
		forwardTransform(fft, gpu, vel.w);
		forwardTransform(fft, gpu, vel.s);
		forwardTransform(fft, gpu, vel.c);

	return;
	}

}

//============================================================================================
// Import from file
//============================================================================================

int readDataSize(FILE *f){
	int bin;

	int flag = fread((void*)(&bin), sizeof(float), 1, f);

	if(flag == 1)
		return bin;
	else{
		return 0;
	}
}

double readDouble(FILE *f){
	double v;

	int flag = fread((void*)(&v), sizeof(double), 1, f);

	if(flag == 1)
		return v;
	else{
		return 0;
	}
}

void loadData(gpudata gpu, const char *name, double **var)
{ // Function to read in velocity data into multiple GPUs

	int i, j, k, n, idx, N;
	char title[256];
	
	snprintf(title, sizeof(title), DataLocation, name);
	printf("Importing data from %s \n", title);
	FILE *file = fopen(title, "rb");
	N = readDouble(file)/sizeof(double);
	if(N!=NX*NY*NZ) {
		printf("Error! N!=NX*NY*NZ");
		return;
	}
	printf("Reading data from ");
  for (n = 0; n < gpu.nGPUs; ++n){
  	printf("GPU %d",n);
    for (i = 0; i < gpu.nx[n]; ++i){
      for (j = 0; j < NY; ++j){
        for (k = 0; k < NZ; ++k){
          idx = k + 2*NZ2*j + 2*NZ2*NY*i;	
          var[n][idx] = readDouble(file);
        }
			}
		}
    printf(" ... Done!\n");
  }

	fclose(file);

	return;
}

void importVelocity(gpudata gpu, fielddata h_vel, fielddata vel)
{	// Import data from file
	int n;

	loadData(gpu, "u", h_vel.u);
	loadData(gpu, "v", h_vel.v);
	loadData(gpu, "w", h_vel.w);

	// Copy data from host to device
	// printf("Copy results to GPU memory...\n");
	for(n=0; n<gpu.nGPUs; ++n){
		cudaSetDevice(n);
		cudaDeviceSynchronize();
		checkCudaErrors( cudaMemcpyAsync(vel.u[n], h_vel.u[n], sizeof(cufftDoubleComplex)*gpu.nx[n]*NY*NZ2, cudaMemcpyDefault) );
		checkCudaErrors( cudaMemcpyAsync(vel.v[n], h_vel.v[n], sizeof(cufftDoubleComplex)*gpu.nx[n]*NY*NZ2, cudaMemcpyDefault) );
		checkCudaErrors( cudaMemcpyAsync(vel.w[n], h_vel.w[n], sizeof(cufftDoubleComplex)*gpu.nx[n]*NY*NZ2, cudaMemcpyDefault) );
	}

	return;
}

void importScalar(gpudata gpu, fielddata h_vel, fielddata vel)
{	// Import data from file
	int n;

	loadData(gpu, "z", h_vel.s);

	// Copy data from host to device
	// printf("Copy results to GPU memory...\n");
	for(n=0; n<gpu.nGPUs; ++n){
		cudaSetDevice(n);
		cudaDeviceSynchronize();
		checkCudaErrors( cudaMemcpyAsync(vel.s[n], h_vel.s[n], sizeof(cufftDoubleComplex)*gpu.nx[n]*NY*NZ2, cudaMemcpyDefault) );
	}

	return;
}

void importData(gpudata gpu, fielddata h_vel, fielddata vel) // Deprecated
{	// Import data

	importVelocity(gpu, h_vel, vel);

	importScalar(gpu, h_vel, vel);

	return;
}
