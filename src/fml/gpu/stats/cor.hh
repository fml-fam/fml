// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_GPU_STATS_COR_H
#define FML_GPU_STATS_COR_H
#pragma once


#include <stdexcept>

#include "../../_internals/omp.hh"

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
    
    @details Computes the lower triangle of the Pearson correlation matrix.
    Centering is done in-place.
    
    @param[inout] x,y Input data. For the matrix variants, data is mean-centered
    on return.
    @param[out] cov The correlation matrix.
    
    @except In the matrix-matrix and vector-vector variants, if the object
    dimensions/sizes are non-conformable, a `runtime_error` exception is thrown.
    
    @tparam REAL should be 'float' or 'double'.
  */
  template <typename REAL>
  void cor(gpumat<REAL> &x, gpumat<REAL> &cov)
  {
    dimops::scale(true, true, x);
    
    const REAL alpha = 1. / ((REAL) (x.nrows()-1));
    linalg::crossprod(alpha, x, cov);
  }
  
  
  
  /// \overload
  template <typename REAL>
  void cor(gpumat<REAL> &x, gpumat<REAL> &y, gpumat<REAL> &cov)
  {
    if (x.nrows() != y.nrows())
      throw std::runtime_error("non-conformable arguments");
    
    dimops::scale(true, true, x);
    dimops::scale(true, true, y);
    
    const REAL alpha = 1. / ((REAL) (x.nrows()-1));
    linalg::matmult(true, false, alpha, x, y, cov);
  }
  
  
  
  namespace internals
  {
    template <typename REAL>
    __global__ void kernel_cor_vecvec(const len_t n, const REAL *x,
    const REAL *y, const REAL *meanx, const REAL *meany, REAL *cp, REAL *normx,
    REAL *normy)
    {
      int i = blockDim.x*blockIdx.x + threadIdx.x;
      int j = blockDim.y*blockIdx.y + threadIdx.y;
      
      if (i < n && j < n)
      {
        const REAL xi_mm = x[i] - (*meanx);
        const REAL yi_mm = y[i] - (*meany);
      
        atomicAdd(cp, xi_mm * yi_mm);
        atomicAdd(normx, xi_mm * xi_mm);
        atomicAdd(normy, yi_mm * yi_mm);
      }
    }
  }
  
  /// \overload
  template <typename REAL>
  REAL cor(const gpuvec<REAL> &x, const gpuvec<REAL> &y)
  {
    const len_t n = x.size();
    if (n != y.size())
      throw std::runtime_error("non-conformal arguments");
    
    const REAL meanx = x.sum() / n;
    const REAL meany = y.sum() / n;
    gpuscalar<REAL> meanx_gpu(x.get_card(), meanx);
    gpuscalar<REAL> meany_gpu(x.get_card(), meany);
    
    REAL cp = 0, normx = 0, normy = 0;
    gpuscalar<REAL> cp_gpu(x.get_card(), cp);
    gpuscalar<REAL> normx_gpu(x.get_card(), normx);
    gpuscalar<REAL> normy_gpu(x.get_card(), normy);
    
    internals::kernel_cor_vecvec<<<x.get_griddim(), x.get_blockdim()>>>(n,
      x.data_ptr(), y.data_ptr(), meanx_gpu.data_ptr(), meany_gpu.data_ptr(),
      cp_gpu.data_ptr(), normx_gpu.data_ptr(), normy_gpu.data_ptr());
    
    cp_gpu.get_val(&cp);
    normx_gpu.get_val(&normx);
    normy_gpu.get_val(&normy);
    
    return cp / sqrt(normx * normy);
  }
}
}


#endif
