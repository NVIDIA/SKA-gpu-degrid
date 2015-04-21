# GPUDegrid

Radio astronomy degridding on the GPU. Not optimized. 

Invoke "make" to make and "degrid" to run.

===Make options in brief===

- PRECISION (float/double) specifies the precision. Default is double
- MANAGED (1 to enable) uses managed memory. Default is off
- CPU_CHECK (1 to enable) computes on both CPU and GPU and checks results
- SCATTER (1 to enable) computes with 1 image point per thread. Default
                        is 1 visibility per 32 threads. Details below 
- DEBUG (1 to enable) compiles with debug flags for GPU and CPU. Also
                      enables CPU_CHECK
- COMPUTE_GCF (1 to enable) compute the GCF in the kernel rather than
                            reading from memory

===Make option details===
Specify PRECISION on the make line to switch between
double and single precision, i.e. 
%> make PRECISION=float

To use unified virtual memory, add MANAGED=1 to the build line

%> make PRECISION=float MANAGED=1

To check results against a CPU computation, add CPU_CHECK=1 to
your make line

%> make PRECISION=float MANAGED=1 CPU_CHECK=1

You can use either of 2 methods to compute. By default, each visibility
is computed by a set of 32 consecutive threads. This method currently 
gives the best performance. A second method assigns each thread to a 
single point on the image and pushes data to each visibility. There is 
cooperation between threads in a threadblock. Multiple threadblocks 
contribute to the same visibilty via an atomic addition. To enable
this alternate method, call make with SCATTER=1. For example,

%> make PRECISION=float MANAGED=1 CPU_CHECK=1 SCATTER=1

If you specify COMPUTE_GCF=1, the GCF will not be loaded from memory.
Instead, the first value of gcf[] will be read as (T,w) with T and w
used to compute gcf(x,y) via

gcf(x,y) = exp(2*pi*j*w*(1-T*sqrt(x**2+y**2)))

where x,y is the distance from the image point to the visibility.
This is for exploration purposes only. The values do not match the
results in crocodile or the randomly generated gcf in degrid.cu.

