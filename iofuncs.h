#ifndef IOFUNCS_H
#define IOFUNCS_H
#include "struct_def.h"

void save3Dfields(int c, fftdata fft, gpudata gpu, fielddata h_vel, fielddata vel);

void save2Dfields(int c, fftdata fft, gpudata gpu, fielddata h_vel, fielddata vel);

void writeStats( const char* name, double *in);

void importData(gpudata gpu, fielddata h_vel, fielddata vel);

void importScalar(gpudata gpu, fielddata h_vel, fielddata vel);

void importVelocity(gpudata gpu, fielddata h_vel, fielddata vel);

void displayDeviceProps(int numGPUs);

void printTurbStats(int iter, double time, statistics stats);

void printIterTime(int iter, double time);

void saveStatsData(const int c, statistics stats);

void saveYprofs(const int c, profile data);

#endif
