// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_MPI_LINALG_MATMULT_H
#define FML_MPI_LINALG_MATMULT_H
#pragma once


#include <stdexcept>

#include "../../_internals/linalgutils.hh"

#include "../mpimat.hh"

#include "internals/err.hh"
#include "internals/pblas.hh"


namespace fml
{
namespace linalg
{
  /**
    @brief Returns alpha*op(x)*op(y) where op(A) is A or A^T
    
    @param[in] transx Should x^T be used?
    @param[in] transy Should y^T be used?
    @param[in] alpha Scalar.
    @param[in] x Left multiplicand.
    @param[in] y Right multiplicand.
    
    @except If x and y are inappropriately sized for a matrix product, the
     method will throw a 'runtime_error' exception. If the inputs are
     distributed on different grids, a `runtime_exception` is thrown.
    
    @impl Uses the PBLAS function `pXgemm()`.
    
    @comm The method will communicate across all processes in the BLACS grid.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  mpimat<REAL> matmult(const bool transx, const bool transy, const REAL alpha, const mpimat<REAL> &x, const mpimat<REAL> &y)
  {
    err::check_grid(x, y);
    
    len_t m, n, k;
    fml::linalgutils::matmult_params(transx, transy, x.nrows(), x.ncols(), y.nrows(), y.ncols(), &m, &n, &k);
    
    const grid g = x.get_grid();
    mpimat<REAL> ret(g, m, n, x.bf_rows(), x.bf_cols());
    
    const char ctransx = transx ? 'T' : 'N';
    const char ctransy = transy ? 'T' : 'N';
    
    fml::pblas::gemm(ctransx, ctransy, m, n, k, alpha,
      x.data_ptr(), x.desc_ptr(), y.data_ptr(), y.desc_ptr(),
      (REAL)0, ret.data_ptr(), ret.desc_ptr());
    
    return ret;
  }
  
  
  
  /**
    @brief Computes ret = alpha*op(x)*op(y) where op(A) is A or A^T
    
    @param[in] transx Should x^T be used?
    @param[in] transy Should y^T be used?
    @param[in] alpha Scalar.
    @param[in] x Left multiplicand.
    @param[in] y Right multiplicand.
    @param[out] ret The product.
    
    @except If x and y are inappropriately sized for a matrix product, the
     method will throw a 'runtime_error' exception. If the inputs are
     distributed on different grids, a `runtime_exception` is thrown.
    
    @impl Uses the PBLAS function `pXgemm()`.
    
    @comm The method will communicate across all processes in the BLACS grid.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  void matmult(const bool transx, const bool transy, const REAL alpha, const mpimat<REAL> &x, const mpimat<REAL> &y, mpimat<REAL> &ret)
  {
    err::check_grid(x, y, ret);
    
    len_t m, n, k;
    fml::linalgutils::matmult_params(transx, transy, x.nrows(), x.ncols(), y.nrows(), y.ncols(), &m, &n, &k);
    
    if (m != ret.nrows() || n != ret.ncols())
      ret.resize(m, n);
    
    const char ctransx = transx ? 'T' : 'N';
    const char ctransy = transy ? 'T' : 'N';
    
    fml::pblas::gemm(ctransx, ctransy, m, n, k, alpha,
      x.data_ptr(), x.desc_ptr(), y.data_ptr(), y.desc_ptr(),
      (REAL)0, ret.data_ptr(), ret.desc_ptr());
  }
}
}


#endif
