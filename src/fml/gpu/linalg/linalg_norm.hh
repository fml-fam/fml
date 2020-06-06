// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_GPU_LINALG_LINALG_NORM_H
#define FML_GPU_LINALG_LINALG_NORM_H
#pragma once


#include <stdexcept>

#include "../arch/arch.hh"

#include "../internals/gpuscalar.hh"
#include "../internals/kernelfuns.hh"

#include "../gpumat.hh"


namespace fml
{
namespace linalg
{
  namespace
  {
    // Based on https://forums.developer.nvidia.com/t/atomicmax-with-floats/8791/10
    static __device__ float atomicMaxf(float *address, float val)
    {
      int *address_int = (int*) address;
      int old = *address_int, assumed;
      
      while (val > __int_as_float(old))
      {
        assumed = old;
        old = atomicCAS(address_int, assumed, __float_as_int(val));
      }
      
      return __int_as_float(old);
    }
    
    static __device__ double atomicMaxf(double *address, double val)
    {
      unsigned long long *address_ull = (unsigned long long*) address;
      unsigned long long old = *address_ull, assumed;
      
      while (val > __longlong_as_double(old))
      {
        assumed = old;
        old = atomicCAS(address_ull, assumed, __double_as_longlong(val));
      }
      
      return __longlong_as_double(old);
    }
  }
  
  
  
  namespace
  {
    template <typename REAL>
    __global__ void kernel_norm_1(const len_t m, const len_t n, const REAL *x, REAL *macs, REAL *norm)
    {
      int i = blockDim.x*blockIdx.x + threadIdx.x;
      int j = blockDim.y*blockIdx.y + threadIdx.y;
      
      if (i < m && j < n)
      {
        REAL tmp = fabs(x[i + m*j]);
        atomicAdd(macs + j, tmp);
        atomicMaxf(norm, macs[j]);
      }
    }
  }
  
  /**
    @brief Computes the 1 matrix norm, the maximum absolute column sum.
    
    @param[in] x Input data matrix, replaced by its LU factorization.
    
    @return Returns the norm.
    
    @allocs Allocates temporary storage to store the col sums.
    
    @except If an allocation is triggered and fails, a `bad_alloc` exception
    will be thrown.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  REAL norm_1(const gpumat<REAL> &x)
  {
    REAL norm = 0;
    gpuscalar<REAL> norm_gpu(x.get_card(), norm);
    gpuvec<REAL> macs(x.get_card(), x.ncols());
    macs.fill_zero();
    
    kernel_norm_1<<<x.get_griddim(), x.get_blockdim()>>>(x.nrows(),
      x.ncols(), x.data_ptr(), macs.data_ptr(), norm_gpu.data_ptr());
    
    norm_gpu.get_val(&norm);
    
    return norm;
  }
  
  
  
  namespace
  {
    template <typename REAL>
    __global__ void kernel_norm_I(const len_t m, const len_t n, const REAL *x, REAL *mars, REAL *norm)
    {
      int i = blockDim.x*blockIdx.x + threadIdx.x;
      int j = blockDim.y*blockIdx.y + threadIdx.y;
      
      if (i < m && j < n)
      {
        REAL tmp = fabs(x[i + m*j]);
        atomicAdd(mars + i, tmp);
        atomicMaxf(norm, mars[i]);
      }
    }
  }
  
  /**
    @brief Computes the infinity matrix norm, the maximum absolute row sum.
    
    @param[in] x Input data matrix, replaced by its LU factorization.
    
    @return Returns the norm.
    
    @allocs Allocates temporary storage to store the row sums.
    
    @except If an allocation is triggered and fails, a `bad_alloc` exception
    will be thrown.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  REAL norm_I(const gpumat<REAL> &x)
  {
    REAL norm = 0;
    gpuscalar<REAL> norm_gpu(x.get_card(), norm);
    gpuvec<REAL> mars(x.get_card(), x.nrows());
    mars.fill_zero();
    
    kernel_norm_I<<<x.get_griddim(), x.get_blockdim()>>>(x.nrows(),
      x.ncols(), x.data_ptr(), mars.data_ptr(), norm_gpu.data_ptr());
    
    norm_gpu.get_val(&norm);
    
    return norm;
  }
  
  
  
  namespace
  {
    template <typename REAL>
    __global__ void kernel_norm_F_sq(const len_t m, const len_t n, const REAL *x, REAL *norm)
    {
      int i = blockDim.x*blockIdx.x + threadIdx.x;
      int j = blockDim.y*blockIdx.y + threadIdx.y;
      
      if (i < m && j < n)
      {
        REAL tmp = x[i + m*j] * x[i + m*j];
        atomicAdd(norm, tmp);
      }
    }
  }
  
  /**
    @brief Computes the Frobenius/Euclidean matrix norm.
    
    @param[in] x Input data matrix, replaced by its LU factorization.
    
    @return Returns the norm.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  REAL norm_F(const gpumat<REAL> &x)
  {
    REAL norm = 0;
    gpuscalar<REAL> norm_gpu(x.get_card(), norm);
    
    kernel_norm_F_sq<<<x.get_griddim(), x.get_blockdim()>>>(x.nrows(),
      x.ncols(), x.data_ptr(), norm_gpu.data_ptr());
    
    norm_gpu.get_val(&norm);
    
    return sqrt(norm);
  }
  
  
  
  namespace
  {
    template <typename REAL>
    __global__ void kernel_norm_M(const len_t m, const len_t n, const REAL *x, REAL *norm)
    {
      int i = blockDim.x*blockIdx.x + threadIdx.x;
      int j = blockDim.y*blockIdx.y + threadIdx.y;
      
      if (i < m && j < n)
      {
        REAL tmp = fabs(x[i + m*j]);
        atomicMaxf(norm, tmp);
      }
    }
  }
  
  /**
    @brief Computes the maximum modulus matrix norm.
    
    @param[in] x Input data matrix, replaced by its LU factorization.
    
    @return Returns the norm.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  REAL norm_M(const gpumat<REAL> &x)
  {
    REAL norm = 0;
    gpuscalar<REAL> norm_gpu(x.get_card(), norm);
    
    kernel_norm_M<<<x.get_griddim(), x.get_blockdim()>>>(x.nrows(),
      x.ncols(), x.data_ptr(), norm_gpu.data_ptr());
    
    norm_gpu.get_val(&norm);
    
    return norm;
  }
}
}


#endif
