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
__device__ int2 convert(int asize, int Qpx, float pin) {

   float frac; float round;
   //TODO add the 1 afterward?
   frac = modf((pin+1)*asize, &round);
   return make_int2(int(round), int(frac*Qpx));
}

__device__ double make_zero(double2* in) { return (double)0.0;}
__device__ float make_zero(float2* in) { return (float)0.0;}

template <int gcf_dim, class CmplxType>
__global__ void degrid_kernel(CmplxType* out, CmplxType* in, size_t npts, CmplxType* img, 
                              size_t img_dim, CmplxType* gcf) {
   
   __shared__ CmplxType shm[1024/gcf_dim][gcf_dim+1];
   __shared__ CmplxType inbuff[32];
   for (int n = 32*blockIdx.x; n<npts; n+= 32*gridDim.x) {
   if (threadIdx.y == 0 && threadIdx.x<32) inbuff[threadIdx.x] = in[n+threadIdx.x];
   __syncthreads();
   for (int q=0;q<32;q++) {
      CmplxType inn = inbuff[q];
      int sub_x = floorf(GCF_GRID*(inn.x-floorf(inn.x)));
      int sub_y = floorf(GCF_GRID*(inn.y-floorf(inn.y)));
      int main_x = floorf(inn.x); 
      int main_y = floorf(inn.y); 
      auto sum_r = make_zero(img);
      auto sum_i = make_zero(img);
      int a = threadIdx.x-gcf_dim/2;
      for(int b = threadIdx.y-gcf_dim/2;b<gcf_dim/2;b+=blockDim.y)
      {
         auto r1 = img[main_x+a+img_dim*(main_y+b)].x; 
         auto i1 = img[main_x+a+img_dim*(main_y+b)].y; 
         auto r2 = __ldg(&gcf[gcf_dim*gcf_dim*(GCF_GRID*sub_y+sub_x) + 
                        gcf_dim*b+a].x);
         auto i2 = __ldg(&gcf[gcf_dim*gcf_dim*(GCF_GRID*sub_y+sub_x) + 
                        gcf_dim*b+a].y);
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
      }

      //Reduce the top row
      for(int s = blockDim.x/2;s>16;s/=2) {
         if (0 == threadIdx.y && threadIdx.x < s) 
                    shm[0][threadIdx.x].x += shm[0][threadIdx.x+s].x;
         if (0 == threadIdx.y && threadIdx.x < s) 
                    shm[0][threadIdx.x].y += shm[0][threadIdx.x+s].y;
         __syncthreads();
      }
      if (threadIdx.y == 0) {
         //Reduce the final warp using shuffle
         CmplxType tmp = shm[0][threadIdx.x];
         for(int s = blockDim.x < 16 ? blockDim.x : 16; s>0;s/=2) {
            tmp.x += __shfl_down(tmp.x,s);
            tmp.y += __shfl_down(tmp.y,s);
         }
         
         if (threadIdx.x == 0) {
            out[n+q] = tmp;
         }
      }
   }
   }
}

template <class CmplxType>
void degridGPU(CmplxType* out, CmplxType* in, size_t npts, CmplxType *img, size_t img_dim, 
               CmplxType *gcf, size_t gcf_dim) {
//degrid on the GPU
//  out (out) - the output values for each location
//  in  (in)  - the locations to be interpolated 
//  npts (in) - number of locations
//  img (in) - the image
//  img_dim (in) - dimension of the image
//  gcf (in) - the gridding convolution function
//  gcf_dim (in) - dimension of the GCF

   CmplxType *d_out, *d_in, *d_img, *d_gcf;

   cudaEvent_t start, stop;
   cudaEventCreate(&start); cudaEventCreate(&stop);

   CUDA_CHECK_ERR(__LINE__,__FILE__);
#ifdef __MANAGED
   d_img = img;
   d_gcf = gcf;
   d_out = out;
   d_in = in;
   std::cout << "img size = " << (img_dim*img_dim+2*img_dim*gcf_dim+2*gcf_dim)*
                                                                 sizeof(CmplxType) << std::endl;
   std::cout << "out size = " << sizeof(CmplxType)*npts << std::endl;
#else
   //img is padded to avoid overruns. Subtract to find the real head
   img -= img_dim*gcf_dim+gcf_dim;

   //Allocate GPU memory
   std::cout << "img size = " << (img_dim*img_dim+2*img_dim*gcf_dim+2*gcf_dim)*
                                                                 sizeof(CmplxType) << std::endl;
   cudaMalloc(&d_img, sizeof(CmplxType)*(img_dim*img_dim+2*img_dim*gcf_dim+2*gcf_dim));
   cudaMalloc(&d_gcf, sizeof(CmplxType)*64*gcf_dim*gcf_dim);
   cudaMalloc(&d_out, sizeof(CmplxType)*npts);
   cudaMalloc(&d_in, sizeof(CmplxType)*npts);
   std::cout << "out size = " << sizeof(CmplxType)*npts << std::endl;
   CUDA_CHECK_ERR(__LINE__,__FILE__);

   //Copy in img, gcf and out
   cudaEventRecord(start);
   cudaMemcpy(d_img, img, 
              sizeof(CmplxType)*(img_dim*img_dim+2*img_dim*gcf_dim+2*gcf_dim), 
              cudaMemcpyHostToDevice);
   cudaMemcpy(d_gcf, gcf, sizeof(CmplxType)*64*gcf_dim*gcf_dim, 
              cudaMemcpyHostToDevice);
   cudaMemcpy(d_in, in, sizeof(CmplxType)*npts,
              cudaMemcpyHostToDevice);
   CUDA_CHECK_ERR(__LINE__,__FILE__);
   std::cout << "memcpy time: " << getElapsed(start, stop) << std::endl;

   //move d_img and d_gcf to remove padding
   d_img += img_dim*gcf_dim+gcf_dim;
#endif
   //offset gcf to point to the middle of the first GCF for cleaner code later
   d_gcf += gcf_dim*(gcf_dim+1)/2;

   cudaEventRecord(start);
   degrid_kernel<128>
            <<<npts/32,dim3(gcf_dim,512/gcf_dim)>>>(d_out,d_in,npts,d_img,img_dim,d_gcf); 
   float kernel_time = getElapsed(start,stop);
   std::cout << "Processed " << npts << " complex points in " << kernel_time << " ms." << std::endl;
   std::cout << npts / 1000000.0 / kernel_time * gcf_dim * gcf_dim * 8 << "Gflops" << std::endl;
   CUDA_CHECK_ERR(__LINE__,__FILE__);

#ifdef __MANAGED
   cudaDeviceSynchronize();
#else
   cudaMemcpy(out, d_out, sizeof(CmplxType)*npts, cudaMemcpyDeviceToHost);
   CUDA_CHECK_ERR(__LINE__,__FILE__);

   //Restore d_img and d_gcf for deallocation
   d_img -= img_dim*gcf_dim+gcf_dim;
   d_gcf -= gcf_dim*(gcf_dim+1)/2;
   cudaFree(d_out);
   cudaFree(d_img);
#endif
   cudaEventDestroy(start); cudaEventDestroy(stop);
   CUDA_CHECK_ERR(__LINE__,__FILE__);
}
template void degridGPU<double2>(double2* out, double2* in, size_t npts, double2 *img, 
                                 size_t img_dim, double2 *gcf, size_t gcf_dim); 
template void degridGPU<float2>(float2* out, float2* in, size_t npts, float2 *img, 
                                size_t img_dim, float2 *gcf, size_t gcf_dim); 
