PRECISION ?= double
ifeq ($(MANAGED),1)
	USERFLAGS += -D__MANAGED 
endif
ifeq ($(DEBUG),1)
	USERFLAGS += -g -G -lineinfo -D__CPU_CHECK
endif
#USERFLAGS += -Xcompiler -fPIC

all:  degrid GPUDegrid.so

degrid: degrid.cu cucommon.cuh degrid_gpu.cuh degrid_gpu.o Defines.h
	nvcc -arch=sm_35 -std=c++11 -DPRECISION=${PRECISION} $(USERFLAGS) -o degrid degrid.cu degrid_gpu.o

degrid_gpu.o: degrid_gpu.cu degrid_gpu.cuh cucommon.cuh Defines.h
	nvcc -c -arch=sm_35 -std=c++11 $(USERFLAGS) -o degrid_gpu.o degrid_gpu.cu

degrid_gpu_pic.o: degrid_gpu.cu degrid_gpu.cuh cucommon.cuh Defines.h
	nvcc -Xcompiler -fPIC -c -arch=sm_35 -std=c++11 $(USERFLAGS) -o degrid_gpu_pic.o degrid_gpu.cu

degrid-debug: degrid.cu degrid_gpu-debug.o cucommon.cuh Defines.h
	nvcc -arch=sm_35 -std=c++11 -DPRECISION=${PRECISION} -g -G -lineinfo $(USERFLAGS) -o degrid-debug degrid_gpu-debug.o degrid.cu

degrid_gpu-debug.o: degrid_gpu.cu degrid_gpu.cuh cucommon.cuh Defines.h
	nvcc -c -arch=sm_35 -std=c++11 -g -G -lineinfo $(USERFLAGS) -o degrid_gpu-debug.o degrid_gpu.cu

GPUDegrid.so: GPUDegrid.cpp degrid_gpu_pic.o
	nvcc -std=c++11 -shared -Xcompiler -fPIC -I/usr/include/python2.7/ -lpython2.7 -o GPUDegrid.so GPUDegrid.cpp  degrid_gpu_pic.o
