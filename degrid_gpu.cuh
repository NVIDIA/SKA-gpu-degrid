#ifndef __DEGRID_CUH
#define __DEGRID_CUH
template <class CmplxType>
void degridGPU(CmplxType* out, CmplxType* in, size_t npts, CmplxType *img, size_t img_dim, 
               CmplxType *gcf, size_t gcf_dim); 
#endif

