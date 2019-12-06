// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_GPU_KERNELFUNS_H
#define FML_GPU_KERNELFUNS_H
#pragma once


#include <cuda_runtime.h>

#include "../_internals/types.hh"


namespace kernelfuns
{
  template <typename REAL>
  __global__ void kernel_rev_rows(const len_t m, const len_t n, REAL *data)
  {
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    int j = blockDim.y*blockIdx.y + threadIdx.y;
    
    if (i < m/2 && j < n)
    {
      REAL tmp = data[i + m*j];
      data[i + m*j] = data[m-i-1 + m*j];
      data[m-i-1 + m*j] = tmp;
    }
  }
  
  
  
  template <typename REAL>
  __global__ void kernel_rev_cols(const len_t m, const len_t n, REAL *data)
  {
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    int j = blockDim.y*blockIdx.y + threadIdx.y;
    
    if (i < m && j < n/2)
    {
      REAL tmp = data[i + m*j];
      data[i + m*j] = data[i + m*(n-j-1)];
      data[i + m*(n-j-1)] = tmp;
    }
  }
  
  
  
  template <typename REAL>
  __global__ void kernel_fill_eye(const len_t m, const len_t n, REAL *data)
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
  __global__ void kernel_fill_diag(const len_t size, const REAL *v, const len_t m, const len_t n, REAL *data)
  {
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    int j = blockDim.y*blockIdx.y + threadIdx.y;
    
    if (i < m && j < n)
    {
      if (i == j)
        data[i + m*j] = v[i % size];
      else
        data[i + m*j] = (REAL) 0;
    }
  }
  
  
  
  template <typename REAL>
  __global__ void kernel_fill_val(const REAL v, const len_t m, const len_t n, REAL *data)
  {
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    int j = blockDim.y*blockIdx.y + threadIdx.y;
    
    if (i < m && j < n)
      data[i + m*j] = v;
  }
  
  
  
  static __global__ void kernel_fill_linspace(const __half start, const __half stop, const len_t m, const len_t n, __half *data)
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
  __global__ void kernel_fill_linspace(const REAL start, const REAL stop, const len_t m, const len_t n, REAL *data)
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
  __global__ void kernel_scale(const REAL s, const len_t m, const len_t n, REAL *data)
  {
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    int j = blockDim.y*blockIdx.y + threadIdx.y;
    
    if (i < m && j < n)
      data[i + m*j] *= s;
  }
  
  
  
  template <typename REAL>
  __global__ void kernel_any_inf(const len_t m, const len_t n, const REAL *data, int *has_inf)
  {
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    int j = blockDim.y*blockIdx.y + threadIdx.y;
    
    if (i < m && j < n)
    {
      if (isinf(data[i + m*j]))
        atomicMax(has_inf, 1);
    }
  }
  
  
  
  template <typename REAL>
  __global__ void kernel_any_nan(const len_t m, const len_t n, const REAL *data, int *has_nan)
  {
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    int j = blockDim.y*blockIdx.y + threadIdx.y;
    
    if (i < m && j < n)
    {
      if (isnan(data[i + m*j]))
        atomicMax(has_nan, 1);
    }
  }
  
  
  
  template <typename REAL>
  __global__ void kernel_all_eq(const len_t m, const len_t n, const REAL *x, const REAL *y, int *all_eq)
  {
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    int j = blockDim.y*blockIdx.y + threadIdx.y;
    
    if (i < m && j < n)
    {
      if (x[i + m*j] != y[i + m*j])
        atomicMin(all_eq, 0);
    }
  }
  
  
  
  template <typename REAL>
  __global__ void kernel_trace(const len_t m, const len_t n, const REAL *data, REAL *tr)
  {
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    int j = blockDim.y*blockIdx.y + threadIdx.y;
    
    if (i < m && j < n && i == j)
      atomicAdd(tr, data[i + m*i]);
  }
}


#endif
