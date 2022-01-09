// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_GPU_LINALG_DET_H
#define FML_GPU_LINALG_DET_H
#pragma once


#include <stdexcept>

#include "../arch/arch.hh"

#include "../internals/gpuscalar.hh"

#include "../gpuvec.hh"
#include "../gpumat.hh"

#include "lu.hh"


namespace fml
{
namespace linalg
{
  namespace
  {
    static __global__ void kernel_lu_pivot_sgn(const len_t n, int *ipiv, int *sgn)
    {
      int i = blockDim.x*blockIdx.x + threadIdx.x;
      
      if (i < n)
      {
        ipiv[i] = (ipiv[i] != (i+1) ? -1 : 1);
        atomicAdd(sgn, ipiv[i]);
        
        if (threadIdx.x == 0)
          (*sgn) = ((*sgn)%2 == 0 ? 1 : -1);
      }
    }
    
    template <typename REAL>
    __global__ void kernel_det_mod(const len_t m, const len_t n, const REAL *x, REAL *mod, int *sgn)
    {
      int i = blockDim.x*blockIdx.x + threadIdx.x;
      int j = blockDim.y*blockIdx.y + threadIdx.y;
      
      if (i < m && j < n && i == j)
      {
        REAL d = x[i + m*j];
        int s = 0;
        
        if (d < 0)
        {
          d = (*mod) + log(-d);
          s++;
        }
        else
          d = (*mod) + log(d);
        
        atomicAdd(mod, d);
        int s_g = (*sgn);
        (*sgn) = 0;
        atomicAdd(sgn, s);
        
        if (threadIdx.x == 0 && threadIdx.y == 0)
        {
          (*sgn) = ((*sgn)%2 == 0 ? 1 : -1);
          (*sgn) *= s_g;
        }
      }
    }
  }
  
  /**
    @brief Computes the determinant in logarithmic form.
    
    @details The input is replaced by its LU factorization.
    
    @param[inout] x Input data matrix, replaced by its LU factorization.
    @param[out] sign The sign of the determinant.
    @param[out] modulus Log of the modulus.
    
    @impl Uses `lu()`.
    
    @allocs Allocates temporary storage to compute the LU.
    
    @except If an allocation is triggered and fails, a `bad_alloc` exception
    will be thrown.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  void det(gpumat<REAL> &x, int &sign, REAL &modulus)
  {
    if (!x.is_square())
      throw std::runtime_error("'x' must be a square matrix");
    
    auto c = x.get_card();
    
    gpuvec<int> p(c);
    int info;
    lu(x, p, info);
    
    if (info != 0)
    {
      if (info > 0)
      {
        sign = 1;
        modulus = -INFINITY;
        return;
      }
      else
        return;
    }
    
    // get determinant
    modulus = 0.0;
    sign = 1;
    
    gpuscalar<int> sign_gpu(c, sign);
    gpuscalar<REAL> modulus_gpu(c, modulus);
    
    kernel_lu_pivot_sgn<<<p.get_griddim(), p.get_blockdim()>>>(p.size(),
      p.data_ptr(), sign_gpu.data_ptr());
    kernel_det_mod<<<x.get_griddim(), x.get_blockdim()>>>(x.nrows(), x.ncols(),
      x.data_ptr(), modulus_gpu.data_ptr(), sign_gpu.data_ptr());
    
    sign_gpu.get_val(&sign);
    modulus_gpu.get_val(&modulus);
  }
}
}


#endif
