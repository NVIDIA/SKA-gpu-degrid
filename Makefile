PRECISION ?= double
ifeq ($(MANAGED),1)
	USERFLAGS += -D__MANAGED -x cu
endif

all:  degrid

degrid: degrid.cpp cucommon.cuh degrid_gpu.cuh degrid_gpu.o
	nvcc -arch=sm_35 -std=c++11 -DPRECISION=${PRECISION} $(USERFLAGS) -o degrid degrid.cpp degrid_gpu.o

degrid_gpu.o: degrid_gpu.cu degrid_gpu.cuh cucommon.cuh
	nvcc -c -arch=sm_35 -std=c++11 $(USERFLAGS) -o degrid_gpu.o degrid_gpu.cu

degrid-debug: degrid.cpp degrid_gpu-debug.o cucommon.cuh
	nvcc -arch=sm_35 -std=c++11 -DPRECISION=${PRECISION} -g -G -lineinfo $(USERFLAGS) -o degrid-debug degrid_gpu-debug.o degrid.cpp

degrid_gpu-debug.o: degrid_gpu.cu degrid_gpu.cuh cucommon.cuh
	nvcc -c -arch=sm_35 -std=c++11 -g -G -lineinfo $(USERFLAGS) -o degrid_gpu-debug.o degrid_gpu.cu

