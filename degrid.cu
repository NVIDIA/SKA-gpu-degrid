#include <iostream>
#include "math.h"
#include "stdlib.h"

#define NPOINTS 1000000
#define GCF_DIM 128
#define IMG_SIZE 8192

#define single 77
#if PRECISION==single
#define PRECISION float
#endif

#ifndef PRECISION
#define PRECISION double
#endif
#define PASTER(x) x ## 2
#define EVALUATOR(x) PASTER(x)
#define PRECISION2 EVALUATOR(PRECISION)

void CUDA_CHECK_ERR(unsigned lineNumber, const char* fileName) {

   cudaError_t err = cudaGetLastError();
   if (err) std::cout << "Error " << err << " on line " << lineNumber << " of " << fileName << ": " << cudaGetErrorString(err) << std::endl;
}

float getElapsed(cudaEvent_t start, cudaEvent_t stop) {
   float elapsed;
   cudaEventRecord(stop);
   cudaEventSynchronize(stop);
   cudaEventElapsedTime(&elapsed, start, stop);
   return elapsed;
}

//typedef struct {float x,y;} float2;
//typedef struct {double x,y;} double2;
typedef struct {PRECISION x,y; PRECISION r,i;} ipt;

void init_gcf(PRECISION2 *gcf, size_t size) {

  for (size_t sub_x=0; sub_x<8; sub_x++ )
   for (size_t sub_y=0; sub_y<8; sub_y++ )
    for(size_t x=0; x<size; x++)
     for(size_t y=0; y<size; y++) {
       //Some nonsense GCF
       PRECISION tmp = sin(6.28*x/size/8)*exp(-(1.0*x*x+1.0*y*y*sub_y)/size/size/2);
       gcf[size*size*(sub_x+sub_y*8)+x+y*size].x = tmp*sin(1.0*x*sub_x/(y+1));
       gcf[size*size*(sub_x+sub_y*8)+x+y*size].y = tmp*cos(1.0*x*sub_x/(y+1));
       //std::cout << tmp << gcf[x+y*size].x << gcf[x+y*size].y << std::endl;
     }

}

__global__ void doNothing(ipt* out, PRECISION2* img, PRECISION2* gcf, size_t npts){ }

__global__ void degrid_kernel(PRECISION2* out, PRECISION2* in, PRECISION2* img, PRECISION2* gcf, size_t npts) {
   
   __shared__ PRECISION2 shm[1024/GCF_DIM][GCF_DIM+1];
   for (int n = blockIdx.x; n<npts; n+= gridDim.x) {
      int sub_x = floorf(8*(in[n].x-floorf(in[n].x)));
      int sub_y = floorf(8*(in[n].y-floorf(in[n].y)));
      int main_x = floorf(in[n].x); 
      int main_y = floorf(in[n].y); 
      PRECISION sum_r = 0.0;
      PRECISION sum_i = 0.0;
      int a = threadIdx.x-GCF_DIM/2;
      for(int b = threadIdx.y-GCF_DIM/2;b<GCF_DIM/2;b+=blockDim.y)
      {
         PRECISION r1 = img[main_x+a+IMG_SIZE*(main_y+b)].x; 
         PRECISION i1 = img[main_x+a+IMG_SIZE*(main_y+b)].y; 
         PRECISION r2 = gcf[GCF_DIM*GCF_DIM*(8*sub_y+sub_x) + 
                        GCF_DIM*b+a].x;
         PRECISION i2 = gcf[GCF_DIM*GCF_DIM*(8*sub_y+sub_x) + 
                        GCF_DIM*b+a].y;
         sum_r += r1*r2 - i1*i2; 
         sum_i += r1*i2 + r2*i1;
      }

      //reduce in two directions
      //WARNING: Adjustments must be made if blockDim.y and blockDim.x are no
      //         powers of 2 
      shm[threadIdx.y][threadIdx.x].x = sum_r;
      shm[threadIdx.y][threadIdx.x].y = sum_i;
      __syncthreads();
      //Reduce in y
      for(int s = blockDim.y/2;s>0;s/=2) {
         if (threadIdx.y < s) {
           shm[threadIdx.y][threadIdx.x].x += shm[threadIdx.y+s][threadIdx.x].x;
           shm[threadIdx.y][threadIdx.x].y += shm[threadIdx.y+s][threadIdx.x].y;
         }
         __syncthreads();
         if (s==1) break;
      }

      //Reduce the top row
      if (threadIdx.y > 0) continue;
      for(int s = blockDim.x/2;s>16;s/=2) {
         if (threadIdx.x < s) shm[0][threadIdx.x].x += shm[0][threadIdx.x+s].x;
         if (threadIdx.x < s) shm[0][threadIdx.x].y += shm[0][threadIdx.x+s].y;
         __syncthreads();
      }
      //Reduce the final warp using shuffle
      PRECISION2 tmp = shm[0][threadIdx.x];
      for(int s = blockDim.x < 16 ? blockDim.x : 16; s>0;s/=2) {
         tmp.x += __shfl_down(tmp.x,s);
         tmp.y += __shfl_down(tmp.y,s);
      }
         
      if (threadIdx.x == 0) {
         out[n] = tmp;
      }
   }
}

void degridGPU(PRECISION2* out, PRECISION2* in, PRECISION2 *img, PRECISION2 *gcf) {
//degrid on the CPU
//  out (inout) - the locations to be interpolated
//  img (in) - the image
//  gcf (in) - the gridding convolution function
   PRECISION2 *d_out, *d_in, *d_img, *d_gcf;

   cudaEvent_t start, stop;
   cudaEventCreate(&start); cudaEventCreate(&stop);

   CUDA_CHECK_ERR(__LINE__,__FILE__);
   //img is padded to avoid overruns. Subtract to find the real head
   img -= IMG_SIZE*GCF_DIM+GCF_DIM;

   //Allocate GPU memory
   std::cout << "img size = " << (IMG_SIZE*IMG_SIZE+2*IMG_SIZE*GCF_DIM+2*GCF_DIM)*sizeof(PRECISION2) << std::endl;
   cudaMalloc(&d_img, sizeof(PRECISION2)*(IMG_SIZE*IMG_SIZE+2*IMG_SIZE*GCF_DIM+2*GCF_DIM));
   cudaMalloc(&d_gcf, sizeof(PRECISION2)*64*GCF_DIM*GCF_DIM);
   cudaMalloc(&d_out, sizeof(PRECISION2)*NPOINTS);
   cudaMalloc(&d_in, sizeof(PRECISION2)*NPOINTS);
   std::cout << "out size = " << sizeof(ipt)*NPOINTS << std::endl;
   CUDA_CHECK_ERR(__LINE__,__FILE__);

   //Copy in img, gcf and out
   cudaEventRecord(start);
   cudaMemcpy(d_img, img, 
              sizeof(PRECISION2)*(IMG_SIZE*IMG_SIZE+2*IMG_SIZE*GCF_DIM+2*GCF_DIM), 
              cudaMemcpyHostToDevice);
   CUDA_CHECK_ERR(__LINE__,__FILE__);
   cudaMemcpy(d_gcf, gcf, sizeof(PRECISION2)*64*GCF_DIM*GCF_DIM, 
              cudaMemcpyHostToDevice);
   CUDA_CHECK_ERR(__LINE__,__FILE__);
   cudaMemcpy(d_in, in, sizeof(PRECISION2)*NPOINTS,
              cudaMemcpyHostToDevice);
   CUDA_CHECK_ERR(__LINE__,__FILE__);
   std::cout << "memcpy time: " << getElapsed(start, stop) << std::endl;

   //move d_img and d_gcf to remove padding
   d_img += IMG_SIZE*GCF_DIM+GCF_DIM;
   //offset gcf to point to the middle of the first GCF for cleaner code later
   d_gcf += GCF_DIM*(GCF_DIM+1)/2;

   cudaEventRecord(start);
   degrid_kernel<<<NPOINTS,dim3(GCF_DIM,1024/GCF_DIM)>>>(d_out,d_in,d_img,d_gcf,NPOINTS); 
   float kernel_time = getElapsed(start,stop);
   std::cout << "kernel time: " << kernel_time << std::endl;
   std::cout << NPOINTS / 1000000.0 / kernel_time * GCF_DIM * GCF_DIM * 8 << "Gflops" << std::endl;
   CUDA_CHECK_ERR(__LINE__,__FILE__);

   cudaMemcpy(out, d_out, sizeof(PRECISION2)*NPOINTS, cudaMemcpyDeviceToHost);
   CUDA_CHECK_ERR(__LINE__,__FILE__);

   //Restore d_img and d_gcf for deallocation
   d_img -= IMG_SIZE*GCF_DIM+GCF_DIM;
   d_gcf -= GCF_DIM*(GCF_DIM+1)/2;
   cudaFree(d_out);
   cudaFree(d_img);
   cudaEventDestroy(start); cudaEventDestroy(stop);
   CUDA_CHECK_ERR(__LINE__,__FILE__);
}
void degridCPU(PRECISION2* out, PRECISION2 *in, PRECISION2 *img, PRECISION2 *gcf) {
//degrid on the CPU
//  out (out) - the output values for each location
//  in  (in)  - the locations to be interpolated 
//  img (in) - the image
//  gcf (in) - the gridding convolution function
   //offset gcf to point to the middle for cleaner code later
   gcf += GCF_DIM*(GCF_DIM+1)/2;
#pragma acc parallel loop copyout(out[0:NPOINTS]) copyin(in[0:NPOINTS],gcf[0:64*GCF_DIM*GCF_DIM],img[IMG_SIZE*IMG_SIZE]) gang
   for(size_t n=0; n<NPOINTS; n++) {
      //std::cout << "in = " << in[n].x << ", " << in[n].y << std::endl;
      int sub_x = floorf(8*(in[n].x-floorf(in[n].x)));
      int sub_y = floorf(8*(in[n].y-floorf(in[n].y)));
      //std::cout << "sub = "  << sub_x << ", " << sub_y << std::endl;
      int main_x = floor(in[n].x); 
      int main_y = floor(in[n].y); 
      //std::cout << "main = " << main_x << ", " << main_y << std::endl;
      PRECISION sum_r = 0.0;
      PRECISION sum_i = 0.0;
      #pragma acc parallel loop collapse(2) reduction(+:sum_r,sum_i) vector
      for (int a=-GCF_DIM/2; a<GCF_DIM/2 ;a++)
      for (int b=-GCF_DIM/2; b<GCF_DIM/2 ;b++) {
         PRECISION r1 = img[main_x+a+IMG_SIZE*(main_y+b)].x; 
         PRECISION i1 = img[main_x+a+IMG_SIZE*(main_y+b)].y; 
         PRECISION r2 = gcf[GCF_DIM*GCF_DIM*(8*sub_y+sub_x) + 
                        GCF_DIM*b+a].x;
         PRECISION i2 = gcf[GCF_DIM*GCF_DIM*(8*sub_y+sub_x) + 
                        GCF_DIM*b+a].y;
         //std::cout << r1 << std::endl;
         //std::cout << i1 << std::endl;
         //std::cout << r2 << std::endl;
         //std::cout << i2 << std::endl;
         sum_r += r1*r2 - i1*i2; 
         sum_i += r1*i2 + r2*i1;
      }
      out[n].x = sum_r;
      out[n].y = sum_i;
      //std::cout << "val = " << out[n].r << "+ i" << out[n].i << std::endl;
   } 
   gcf -= GCF_DIM*(GCF_DIM+1)/2;
}
int main(void) {

   PRECISION2* out = (PRECISION2*) malloc(sizeof(PRECISION2)*NPOINTS);
   PRECISION2* in = (PRECISION2*) malloc(sizeof(PRECISION2)*NPOINTS);
   PRECISION2 *img = (PRECISION2*) malloc((IMG_SIZE*IMG_SIZE+2*IMG_SIZE*GCF_DIM+2*GCF_DIM)*sizeof(PRECISION2));

   PRECISION2 *gcf = (PRECISION2*) malloc(64*GCF_DIM*GCF_DIM*sizeof(PRECISION2));

   //img is padded (above and below) to avoid overruns
   img += IMG_SIZE*GCF_DIM+GCF_DIM;
    
   init_gcf(gcf, GCF_DIM);
   srand(2541617);
   for(size_t n=0; n<NPOINTS; n++) {
      in[n].x = ((float)rand())/RAND_MAX*1000;
      in[n].y = ((float)rand())/RAND_MAX*1000;
   }
   for(size_t x=0; x<IMG_SIZE;x++)
   for(size_t y=0; y<IMG_SIZE;y++) {
      img[x+IMG_SIZE*y].x = exp(-((x-1400.0)*(x-1400.0)+(y-3800.0)*(y-3800.0))/8000000.0)+1.0;
      img[x+IMG_SIZE*y].y = 0.4;
   }
   //Zero the data in the offset areas
   for (int x=-IMG_SIZE*GCF_DIM-GCF_DIM;x<0;x++) {
      img[x].x = 0.0; img[x].y = 0.0;
   }
   for (int x=0;x<IMG_SIZE*GCF_DIM+GCF_DIM;x++) {
      img[x+IMG_SIZE*IMG_SIZE].x = 0.0; img[x+IMG_SIZE*IMG_SIZE].y = 0.0;
   }

   degridGPU(out,in,img,gcf);
#ifdef __CPU_CHECK
   PRECISION2 out_cpu=(PRECISION2*)malloc(sizeof(PRECISION2)*NPOINTS);
   degridCPU(out_cpu,in,img,gcf);
#endif


#if 0
   for (size_t n = 0; n < NPOINTS; n++) {
     std::cout << "F(" << in[n].x << ", " << in[n].y << ") = " 
               << out[n].x << ", " << out[n].y 
#ifdef __CPU_CHECK
               << " vs. " << out_cpu[n].x << ", " << out_cpu[n].y 
#endif
               << std::endl;
   }
#endif
   img -= GCF_DIM + IMG_SIZE*GCF_DIM;
   free(img);
   free(gcf);
}
