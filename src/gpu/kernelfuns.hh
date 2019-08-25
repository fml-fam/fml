#ifndef FML_GPU_KERNELFUNS_H
#define FML_GPU_KERNELFUNS_H


#include <cuda_runtime.h>

#include "../matrix.hh"


namespace kernelfuns
{
  template <typename REAL>
  __global__ void kernel_fill_eye(len_t m, len_t n, REAL *data)
  {
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    int j = blockDim.y*blockIdx.y + threadIdx.y;
    
    if (i < m && j < n)
    {
      if (i == j)
        data[i + m*j] = (REAL) 1;
      else
        data[i + m*j] = (REAL) 0;
    }
  }
  
  
  
  template <typename REAL>
  __global__ void kernel_fill_val(const REAL v, len_t m, len_t n, REAL *data)
  {
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    int j = blockDim.y*blockIdx.y + threadIdx.y;
    
    if (i < m && j < n)
      data[i + m*j] = v;
  }
}


#endif
