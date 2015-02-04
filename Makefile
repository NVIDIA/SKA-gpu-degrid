PRECISION ?= double

all:  degrid

degrid: degrid.cu
	nvcc -arch=sm_35 -DPRECISION=${PRECISION} -o degrid degrid.cu

degrid-debug: degrid.cu
	nvcc -arch=sm_35 -DPRECISION=${PRECISION} -g -G -lineinfo -o degrid-debug degrid.cu

