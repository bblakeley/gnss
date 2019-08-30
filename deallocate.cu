#include <stdlib.h>
#include <complex.h>
#include <cufft.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include "dnsparams.h"
#include "struct_def.h"
#include "deallocate.h"

void deallocate_memory(){
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

		cudaFree(stats.Vrms[n]);
		cudaFree(stats.KE[n]);
		cudaFree(stats.epsilon[n]);
		cudaFree(stats.eta[n]);
		cudaFree(stats.l[n]);
		cudaFree(stats.lambda[n]);
		cudaFree(stats.chi[n]);
		cudaFree(stats.area_scalar[n]);
		cudaFree(stats.area_tnti[n]);
		cudaFree(stats.energy_spect[n]);

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

	// cudaFreeHost(vel.u);
	// cudaFreeHost(vel.v);
	// cudaFreeHost(vel.w);
	// cudaFreeHost(vel.s);

	cudaFreeHost(vel.uh);
	cudaFreeHost(vel.vh);
	cudaFreeHost(vel.wh);
	cudaFreeHost(vel.sh);

	// cudaFreeHost(rhs.u);
	// cudaFreeHost(rhs.v);
	// cudaFreeHost(rhs.w);
	// cudaFreeHost(rhs.s);

	cudaFreeHost(rhs.uh);
	cudaFreeHost(rhs.vh);
	cudaFreeHost(rhs.wh);
	cudaFreeHost(rhs.sh);

	// cudaFreeHost(rhs_old.u);
	// cudaFreeHost(rhs_old.v);
	// cudaFreeHost(rhs_old.w);
	// cudaFreeHost(rhs_old.s);

	cudaFreeHost(rhs_old.uh);
	cudaFreeHost(rhs_old.vh);
	cudaFreeHost(rhs_old.wh);
	cudaFreeHost(rhs_old.sh);

	cudaFreeHost(stats.Vrms);
	cudaFreeHost(stats.KE);
	cudaFreeHost(stats.epsilon);
	cudaFreeHost(stats.eta);
	cudaFreeHost(stats.l);
	cudaFreeHost(stats.lambda);
	cudaFreeHost(stats.chi);
	cudaFreeHost(stats.area_scalar);
	cudaFreeHost(stats.area_tnti);
	cudaFreeHost(stats.energy_spect);

	// Deallocate memory on CPU
	free(h_vel.u);
	free(h_vel.v);
	free(h_vel.w);
	free(h_vel.s);

	free(h_stats.Vrms[0]   );
	free(h_stats.KE[0]     );
	free(h_stats.epsilon[0]);
	free(h_stats.eta[0]    );
	free(h_stats.l[0]      );
	free(h_stats.lambda[0] );
	free(h_stats.chi[0]    );
	free(h_stats.area_scalar[0]);
	free(h_stats.area_tnti[0]  );
	free(h_stats.energy_spect[0]  );

	free(h_stats.Vrms   );
	free(h_stats.KE     );
	free(h_stats.epsilon);
	free(h_stats.eta    );
	free(h_stats.l      );
	free(h_stats.lambda );
	free(h_stats.chi    );
	free(h_stats.area_scalar);
	free(h_stats.area_tnti  );
	free(h_stats.energy_spect  );

	return;
}