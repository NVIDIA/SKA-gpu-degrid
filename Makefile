PRECISION ?= double
ifeq ($(MANAGED),1)
	USERFLAGS += -D__MANAGED
endif
ifeq ($(DEBUG),1)
	USERFLAGS += -g -G -lineinfo -D__CPU_CHECK
endif
ifeq ($(SCATTER),1)
	USERFLAGS += -D__SCATTER
endif
ifeq ($(CPU_CHECK),1)
	USERFLAGS += -D__CPU_CHECK
endif
ifeq ($(COMPUTE_GCF),1)
	USERFLAGS += -D__COMPUTE_GCF
endif
USERFLAGS += -Xcompiler -fopenmp

all:  degrid

degrid: degrid.cu cucommon.cuh degrid_gpu.cuh degrid_gpu.o Defines.h
	nvcc -arch=sm_35 -std=c++11 -DPRECISION=${PRECISION} $(USERFLAGS) -o degrid degrid.cu degrid_gpu.o

degrid_gpu.o: degrid_gpu.cu degrid_gpu.cuh cucommon.cuh Defines.h
	nvcc -c -arch=sm_35 -std=c++11 $(USERFLAGS) -o degrid_gpu.o degrid_gpu.cu

degrid-debug: degrid.cu degrid_gpu-debug.o cucommon.cuh Defines.h
	nvcc -arch=sm_35 -std=c++11 -DPRECISION=${PRECISION} -g -G -lineinfo $(USERFLAGS) -o degrid-debug degrid_gpu-debug.o degrid.cu

degrid_gpu-debug.o: degrid_gpu.cu degrid_gpu.cuh cucommon.cuh Defines.h
	nvcc -c -arch=sm_35 -std=c++11 -g -G -lineinfo $(USERFLAGS) -o degrid_gpu-debug.o degrid_gpu.cu

