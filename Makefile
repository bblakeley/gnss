NVCC = /usr/local/cuda/bin/nvcc
NVCC_FLAGS = -g -G -Xcompiler -Wall -std c++03 -arch=sm_60 -O2	## Requires -arch=sm_60 flag to perform Atomics on double precision arrays
LIBRARIES = -L/usr/local/cuda/lib -lcufft 
INCLUDES = -I/usr/local/cuda/samples/common/inc -I/usr/local/cuda/inc

gnss.exe: main.o solver.o statistics.o cudafuncs.o iofuncs.o initialize.o fftfuncs.o allocate.o
	$(NVCC) $(NVCC_FLAGS) $^ -o $@ $(INCLUDES) $(LIBRARIES)

solver.o: solver.cu dnsparams.h cudafuncs.h fftfuncs.h struct_def.h
	$(NVCC) $(NVCC_FLAGS) -dc -o $@ $< $(INCLUDES) $(LIBRARIES)	

fftfuncs.o: fftfuncs.cu dnsparams.h cudafuncs.h
	$(NVCC) $(NVCC_FLAGS) -dc -o $@ $< $(INCLUDES) $(LIBRARIES)

initialize.o: initialize.cu dnsparams.h cudafuncs.h initialize.h
	$(NVCC) $(NVCC_FLAGS) -dc -o $@ $< $(INCLUDES) $(LIBRARIES)

cudafuncs.o: cudafuncs.cu dnsparams.h
	$(NVCC) $(NVCC_FLAGS) -dc -o $@ $< $(INCLUDES) $(LIBRARIES)

statistics.o: statistics.cu dnsparams.h statistics.h cudafuncs.h fftfuncs.h
	$(NVCC) $(NVCC_FLAGS) -dc -o $@ $< $(INCLUDES) $(LIBRARIES)

iofuncs.o: iofuncs.cu iofuncs.h dnsparams.h fftfuncs.h
	$(NVCC) $(NVCC_FLAGS) -dc -o $@ $< $(INCLUDES) $(LIBRARIES)

allocate.o: allocate.cu dnsparams.h allocate.h struct_def.h
	$(NVCC) $(NVCC_FLAGS) -dc -o $@ $< $(INCLUDES) $(LIBRARIES)

main.o: main.cu dnsparams.h statistics.h cudafuncs.h struct_def.h declare.h allocate.h
	$(NVCC) $(NVCC_FLAGS) -dc -o $@ $< $(INCLUDES) $(LIBRARIES)

clean:
	rm -f *.o *.exe
