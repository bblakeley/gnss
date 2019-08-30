#ifndef IOFUNCS_H
#define IOFUNCS_H
#include "struct_def.h"

void save3Dfields(int c, fftinfo fft, gpuinfo gpu, fielddata h_vel, fielddata vel);

void save2Dfield(int c, fftinfo fft, gpuinfo gpu, cufftDoubleComplex **fhat, double **h_f);

void writeStats( const char* name, double *in);

void importData(gpuinfo gpu, fielddata h_vel, fielddata vel);

void displayDeviceProps(int numGPUs);

void printTurbStats(int iter, double time, statistics stats);

void printIterTime(int iter, double time);

void saveStatsData(statistics stats, statistics h_stats);

#endif