all:  degrid

degrid: degrid.cu
	nvcc -arch=sm_35 -o degrid degrid.cu

degrid-debug: degrid.cu
	nvcc -arch=sm_35 -g -G -lineinfo -o degrid-debug degrid.cu

