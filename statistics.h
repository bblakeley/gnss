#ifndef STATS_H
#define STATS_H
#include "struct_def.h"

void calcTurbStats_mgpu(const int c, gpuinfo gpu, fftinfo fft, double **wave, fielddata vel, fielddata rhs, statistics *stats, profile Yprofile);

#endif	
