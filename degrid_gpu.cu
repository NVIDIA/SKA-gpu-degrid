#include "Defines.h"
#include "cucommon.cuh"
#include <iostream>

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
template <class CmplxType>
__global__ void degrid_kernel(CmplxType* out, CmplxType* in, CmplxType* img, CmplxType* gcf, size_t npts) {
   
   __shared__ CmplxType shm[1024/GCF_DIM][GCF_DIM+1];
   for (int n = blockIdx.x; n<npts; n+= gridDim.x) {
      int sub_x = floorf(8*(in[n].x-floorf(in[n].x)));
      int sub_y = floorf(8*(in[n].y-floorf(in[n].y)));
      int main_x = floorf(in[n].x); 
      int main_y = floorf(in[n].y); 
      auto sum_r = img[0].x * 0.0;
      auto sum_i = sum_r;
      int a = threadIdx.x-GCF_DIM/2;
      for(int b = threadIdx.y-GCF_DIM/2;b<GCF_DIM/2;b+=blockDim.y)
      {
         auto r1 = img[main_x+a+IMG_SIZE*(main_y+b)].x; 
         auto i1 = img[main_x+a+IMG_SIZE*(main_y+b)].y; 
         auto r2 = gcf[GCF_DIM*GCF_DIM*(8*sub_y+sub_x) + 
                        GCF_DIM*b+a].x;
         auto i2 = gcf[GCF_DIM*GCF_DIM*(8*sub_y+sub_x) + 
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
      CmplxType tmp = shm[0][threadIdx.x];
      for(int s = blockDim.x < 16 ? blockDim.x : 16; s>0;s/=2) {
         tmp.x += __shfl_down(tmp.x,s);
         tmp.y += __shfl_down(tmp.y,s);
      }
         
      if (threadIdx.x == 0) {
         out[n] = tmp;
      }
   }
}

template <class CmplxType>
void degridGPU(CmplxType* out, CmplxType* in, CmplxType *img, CmplxType *gcf) {
//degrid on the CPU
//  out (inout) - the locations to be interpolated
//  img (in) - the image
//  gcf (in) - the gridding convolution function
   CmplxType *d_out, *d_in, *d_img, *d_gcf;

   cudaEvent_t start, stop;
   cudaEventCreate(&start); cudaEventCreate(&stop);

   CUDA_CHECK_ERR(__LINE__,__FILE__);
   //img is padded to avoid overruns. Subtract to find the real head
   img -= IMG_SIZE*GCF_DIM+GCF_DIM;

   //Allocate GPU memory
   std::cout << "img size = " << (IMG_SIZE*IMG_SIZE+2*IMG_SIZE*GCF_DIM+2*GCF_DIM)*sizeof(CmplxType) << std::endl;
   cudaMalloc(&d_img, sizeof(CmplxType)*(IMG_SIZE*IMG_SIZE+2*IMG_SIZE*GCF_DIM+2*GCF_DIM));
   cudaMalloc(&d_gcf, sizeof(CmplxType)*64*GCF_DIM*GCF_DIM);
   cudaMalloc(&d_out, sizeof(CmplxType)*NPOINTS);
   cudaMalloc(&d_in, sizeof(CmplxType)*NPOINTS);
   std::cout << "out size = " << sizeof(CmplxType)*NPOINTS << std::endl;
   CUDA_CHECK_ERR(__LINE__,__FILE__);

   //Copy in img, gcf and out
   cudaEventRecord(start);
   cudaMemcpy(d_img, img, 
              sizeof(CmplxType)*(IMG_SIZE*IMG_SIZE+2*IMG_SIZE*GCF_DIM+2*GCF_DIM), 
              cudaMemcpyHostToDevice);
   CUDA_CHECK_ERR(__LINE__,__FILE__);
   cudaMemcpy(d_gcf, gcf, sizeof(CmplxType)*64*GCF_DIM*GCF_DIM, 
              cudaMemcpyHostToDevice);
   CUDA_CHECK_ERR(__LINE__,__FILE__);
   cudaMemcpy(d_in, in, sizeof(CmplxType)*NPOINTS,
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

   cudaMemcpy(out, d_out, sizeof(CmplxType)*NPOINTS, cudaMemcpyDeviceToHost);
   CUDA_CHECK_ERR(__LINE__,__FILE__);

   //Restore d_img and d_gcf for deallocation
   d_img -= IMG_SIZE*GCF_DIM+GCF_DIM;
   d_gcf -= GCF_DIM*(GCF_DIM+1)/2;
   cudaFree(d_out);
   cudaFree(d_img);
   cudaEventDestroy(start); cudaEventDestroy(stop);
   CUDA_CHECK_ERR(__LINE__,__FILE__);
}
template void degridGPU<double2>(double2* out, double2* in, double2 *img, double2 *gcf); 
template void degridGPU<float2>(float2* out, float2* in, float2 *img, float2 *gcf); 
