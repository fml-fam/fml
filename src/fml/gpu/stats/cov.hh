// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_GPU_STATS_COV_H
#define FML_GPU_STATS_COV_H
#pragma once


#include <stdexcept>

#include "../internals/gpuscalar.hh"

#include "../gpumat.hh"
#include "../gpuvec.hh"

#include "../dimops.hh"

#include "../linalg/linalg_blas.hh"


namespace fml
{
namespace stats
{
  /**
    @brief Covariance.
    
    @details Computes the lower triangle of the variance-covariance matrix.
    Centering is done in-place.
    
    @param[inout] x,y Input data. For the matrix variants, data is mean-centered
    on return.
    @param[out] cov The covariance matrix.
    
    @except In the matrix-matrix and vector-vector variants, if the object
    dimensions/sizes are non-conformable, a `runtime_error` exception is thrown.
    
    @tparam REAL should be 'float' or 'double'.
  */
  template <typename REAL>
  void cov(gpumat<REAL> &x, gpumat<REAL> &cov)
  {
    dimops::scale(true, false, x);
    
    const REAL alpha = 1. / ((REAL) (x.nrows()-1));
    linalg::crossprod(alpha, x, cov);
  }
  
  
  
  /// \overload
  template <typename REAL>
  void cov(gpumat<REAL> &x, gpumat<REAL> &y, gpumat<REAL> &cov)
  {
    if (x.nrows() != y.nrows())
      throw std::runtime_error("non-conformable arguments");
    
    dimops::scale(true, false, x);
    dimops::scale(true, false, y);
    
    const REAL alpha = 1. / ((REAL) (x.nrows()-1));
    linalg::matmult(true, false, alpha, x, y, cov);
  }
  
  
  
  namespace internals
  {
    template <typename REAL>
    __global__ void kernel_cov_vecvec(const len_t n, const REAL *x,
    const REAL *y, REAL *sxy, REAL *sx, REAL *sy)
    {
      int i = blockDim.x*blockIdx.x + threadIdx.x;
      int j = blockDim.y*blockIdx.y + threadIdx.y;
      
      if (i < n && j < n)
      {
        atomicAdd(sxy, x[i]*y[i]);
        atomicAdd(sx, x[i]);
        atomicAdd(sy, y[i]);
      }
    }
  }
  
  /// \overload
  template <typename REAL>
  REAL cov(const gpuvec<REAL> &x, const gpuvec<REAL> &y)
  {
    const len_t n = x.size();
    if (n != y.size())
      throw std::runtime_error("non-conformal arguments");
    
    REAL sxy, sx, sy;
    sxy = sx = sy = 0;
    gpuscalar<REAL> sxy_gpu(x.get_card(), sxy);
    gpuscalar<REAL> sx_gpu(x.get_card(), sx);
    gpuscalar<REAL> sy_gpu(x.get_card(), sy);
    
    internals::kernel_cov_vecvec<<<x.get_griddim(), x.get_blockdim()>>>(n,
      x.data_ptr(), y.data_ptr(), sxy_gpu.data_ptr(), sx_gpu.data_ptr(),
      sy_gpu.data_ptr());
    
    sxy_gpu.get_val(&sxy);
    sx_gpu.get_val(&sx);
    sy_gpu.get_val(&sy);
    
    return (sxy - (sx*sy*((REAL) 1./n))) * ((REAL) 1./(n-1));
  }
}
}


#endif
