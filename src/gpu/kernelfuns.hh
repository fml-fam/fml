#ifndef FML_GPU_KERNELFUNS_H
#define FML_GPU_KERNELFUNS_H


#include <cuda_runtime.h>

#include "../types.hh"


namespace kernelfuns
{
  template <typename REAL>
  __device__ void kernel_rev_vec(len_t n, REAL *data)
  {
    __shared__ REAL shmem[64];
    
    int ind = threadIdx.x;
    int ind_rev = n - ind - 1;
    
    shmem[ind] = data[ind];
    __syncthreads();
    data[ind] = shmem[ind_rev];
  }
  
  
  
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
  
  
  
  static __global__ void kernel_fill_linspace(const __half start, const __half stop, len_t m, len_t n, __half *data)
  {
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    int j = blockDim.y*blockIdx.y + threadIdx.y;
    
    if (i < m && j < n)
    {
      float v_f = ((float)(stop-start))/((float) m*n-1);
      __half v = (__half) v_f;
      len_t ind = i + m*j;
      data[ind] = v*__int2half_rz(ind) + start;
    }
  }
  
  template <typename REAL>
  __global__ void kernel_fill_linspace(const REAL start, const REAL stop, len_t m, len_t n, REAL *data)
  {
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    int j = blockDim.y*blockIdx.y + threadIdx.y;
    
    if (i < m && j < n)
    {
      REAL v = (stop-start)/((REAL) m*n-1);
      len_t ind = i + m*j;
      data[ind] = v*((REAL) ind) + start;
    }
  }
  
  
  
  template <typename REAL>
  __global__ void kernel_scale(const REAL s, len_t m, len_t n, REAL *data)
  {
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    int j = blockDim.y*blockIdx.y + threadIdx.y;
    
    if (i < m && j < n)
      data[i + m*j] *= s;
  }
}


#endif
