#ifndef STATS_H
#define STATS_H
#include "struct_def.h"

void calcTurbStats_mgpu(const int c, gpudata gpu, fftdata fft, griddata grid, fielddata vel, fielddata rhs, statistics *stats, profile Yprof);

#endif	
