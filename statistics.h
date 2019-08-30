#ifndef STATS_H
#define STATS_H
#include "struct_def.h"

void calcTurbStats_mgpu(const int iter, gpuinfo gpu, fftinfo fft, double **wave, fielddata vel, statistics stats);

#endif	