# GPUDegrid

Radio astronomy degridding on the GPU. Not optimized. 

Invoke "make" to make and "degrid" to run.

Specify PRECISION on the make line to switch between
double and single precision, i.e. 
%> make PRECISION=float

To use unified virtual memory, add MANAGED=1 to the build line

%> make PRECISION=float MANAGED=1
