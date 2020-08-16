// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_GPU_LINALG_LINALG_NORM_H
#define FML_GPU_LINALG_LINALG_NORM_H
#pragma once


#include <stdexcept>

#include "../arch/arch.hh"

#include "../internals/atomics.hh"
#include "../internals/gpu_utils.hh"
#include "../internals/gpuscalar.hh"
#include "../internals/kernelfuns.hh"

#include "../gpumat.hh"

#include "linalg_qr.hh"
#include "linalg_svd.hh"


namespace fml
{
namespace linalg
{
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
        atomics::atomicMaxf(norm, macs[j]);
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
        atomics::atomicMaxf(norm, mars[i]);
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
        atomics::atomicMaxf(norm, tmp);
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
  
  
  
  /**
    @brief Computes the 2/spectral matrix norm.
    
    @details Returns the largest singular value.
    
    @param[inout] x Input data matrix. Values are overwritten.
    
    @return Returns the norm.
    
    @impl Uses `linalg::cpsvd()`.
    
    @allocs Allocates temporary storage to compute the singular values.
    
    @except If an allocation is triggered and fails, a `bad_alloc` exception
    will be thrown.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  REAL norm_2(gpumat<REAL> &x)
  {
    REAL ret;
    gpuvec<REAL> s(x.get_card());
    cpsvd(x, s);
    ret = s.get(0);
    
    return ret;
  }
  
  
  
  namespace
  {
    template <typename REAL>
    REAL cond_square_internals(const char norm, gpumat<REAL> &x)
    {
      const len_t n = x.nrows();
      
      REAL ret;
      
      if (norm == '1')
        ret = norm_1(x);
      else //if (norm == 'I')
        ret = norm_I(x);
      
      invert(x);
      
      if (norm == '1')
        ret *= norm_1(x);
      else //if (norm == 'I')
        ret *= norm_I(x);
      
      return ret;
    }
    
    template <typename REAL>
    REAL cond_nonsquare_internals(const char norm, gpumat<REAL> &x)
    {
      const len_t m = x.nrows();
      const len_t n = x.ncols();
      
      auto c = x.get_card();
      
      REAL ret;
      gpublas_status_t check;
      
      gpuvec<REAL> aux(c);
      
      if (m > n)
      {
        gpumat<REAL> R(c);
        qr(false, x, aux);
        qr_R(x, R);
        
        if (norm == '1')
          ret = norm_1(R);
        else //if (norm == 'I')
          ret = norm_I(R);
        
        x.fill_eye();
        
        check = gpublas::trsm(c->blas_handle(), GPUBLAS_SIDE_LEFT,
          GPUBLAS_FILL_U, GPUBLAS_OP_N, GPUBLAS_DIAG_NON_UNIT, n, n,
          (REAL)1, R.data_ptr(), n, x.data_ptr(), m);
        gpublas::err::check_ret(check, "trsm");
        
        gpu_utils::lacpy(GPUBLAS_FILL_U, n, n, x.data_ptr(), m, R.data_ptr(), n);
        
        if (norm == '1')
          ret *= norm_1(R);
        else //if (norm == 'I')
          ret *= norm_I(R);
      }
      else
      {
        gpumat<REAL> L(c);
        lq(x, aux);
        lq_L(x, L);
        
        if (norm == '1')
          ret = norm_1(L);
        else //if (norm == 'I')
          ret = norm_I(L);
        
        x.fill_eye();
        
        check = gpublas::trsm(c->blas_handle(), GPUBLAS_SIDE_LEFT,
          GPUBLAS_FILL_L, GPUBLAS_OP_N, GPUBLAS_DIAG_NON_UNIT, m, m,
          (REAL)1, L.data_ptr(), m, x.data_ptr(), m);
        gpublas::err::check_ret(check, "trsm");
        
        gpu_utils::lacpy(GPUBLAS_FILL_L, m, m, x.data_ptr(), m, L.data_ptr(), m);
        
        if (norm == '1')
          ret *= norm_1(L);
        else //if (norm == 'I')
          ret *= norm_I(L);
      }
      
      return ret;
    }
  }
  
  /**
    @brief Estimates the condition number under the 1-norm.
    
    @param[in] x Input data matrix.
    
    @param[inout] x Input data matrix. The data is overwritten.
    
    @impl Computes L or R (whichever is smaller) if the input is not square, and
    the inverse otherwise.
    
    @allocs Allocates temporary storage to compute the QR/LQ/LU, as well as
    workspace arrays for the LAPACK condition number function.
    
    @except If an allocation is triggered and fails, a `bad_alloc` exception
    will be thrown.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  REAL cond_1(gpumat<REAL> &x)
  {
    if (x.is_square())
      return cond_square_internals('1', x);
    else
      return cond_nonsquare_internals('1', x);
  }
  
  
  
  /**
    @brief Estimates the condition number under the infinity norm.
    
    @param[in] x Input data matrix.
    
    @param[inout] x Input data matrix. The data is overwritten.
    
    @impl Computes L or R (whichever is smaller) if the input is not square, and
    the inverse otherwise.
    
    @allocs Allocates temporary storage to compute the QR/LQ/LU, as well as
    workspace arrays for the LAPACK condition number function.
    
    @except If an allocation is triggered and fails, a `bad_alloc` exception
    will be thrown.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  REAL cond_I(gpumat<REAL> &x)
  {
    if (x.is_square())
      return cond_square_internals('I', x);
    else
      return cond_nonsquare_internals('I', x);
  }
  
  
  
  /**
    @brief Estimates the condition number under the 2 norm.
    
    @param[in] x Input data matrix.
    
    @param[inout] x Input data matrix. The data is overwritten.
    
    @impl Uses `linalg::cpsvd()`.
    
    @allocs Allocates temporary storage to compute the QR/LQ/LU, as well as
    workspace arrays for the LAPACK condition number function.
    
    @except If an allocation is triggered and fails, a `bad_alloc` exception
    will be thrown.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  REAL cond_2(gpumat<REAL> &x)
  {
    gpuvec<REAL> s(x.get_card());
    cpsvd(x, s);
    
    REAL max = s.max();
    
    if (max == 0)
      return 0;
    
    REAL min;
    fml::gpuscalar<REAL> min_gpu(x.get_card());
    kernelfuns::kernel_min_nz<<<s.get_griddim(), s.get_blockdim()>>>(s.size(),
      s.data_ptr(), min_gpu.data_ptr());
    min_gpu.get_val(&min);
    
    return max/min;
  }
}
}


#endif
