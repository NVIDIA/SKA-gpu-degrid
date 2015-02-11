#include "cucommon.cuh"
#ifndef __DEGRID_CUH
#define __DEGRID_CUH
template <class CmplxType>
void degridGPU(CmplxType* out, CmplxType* in, CmplxType *img, CmplxType *gcf); 
#endif

