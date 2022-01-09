// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_GPU_LINALG_ADD_H
#define FML_GPU_LINALG_ADD_H
#pragma once


#include <stdexcept>

#include "../../_internals/linalgutils.hh"

#include "../arch/arch.hh"

#include "../gpumat.hh"

#include "internals/err.hh"


namespace fml
{
namespace linalg
{
  /**
    @brief Returns alpha*op(x) + beta*op(y) where op(A) is A or A^T
    
    @param[in] transx Should x^T be used?
    @param[in] transy Should y^T be used?
    @param[in] alpha,beta Scalars.
    @param[in] x,y The inputs to the sum.
    @param[out] ret The sum.
    
    @except If x and y are inappropriately sized for the sum, the method will
    throw a 'runtime_error' exception.
    
    @impl Uses the cuBLAS function `cublasXgeam()`.
    
    @tparam REAL should be '__half', 'float', or 'double'.
   */
  template <typename REAL>
  void add(const bool transx, const bool transy, const REAL alpha,
    const REAL beta, const gpumat<REAL> &x, const gpumat<REAL> &y,
    gpumat<REAL> &ret)
  {
    err::check_card(x, y, ret);
    
    len_t m, n;
    fml::linalgutils::matadd_params(transx, transy, x.nrows(), x.ncols(),
      y.nrows(), y.ncols(), &m, &n);
    
    if (ret.nrows() != m || ret.ncols() != n)
      ret.resize(m, n);
    
    auto c = x.get_card();
    gpublas_operation_t cbtransx = transx ? GPUBLAS_OP_T : GPUBLAS_OP_N;
    gpublas_operation_t cbtransy = transy ? GPUBLAS_OP_T : GPUBLAS_OP_N;
    
    gpublas_status_t check = gpublas::geam(c->blas_handle(), cbtransx, cbtransy,
      m, n, alpha, x.data_ptr(), x.nrows(), beta, y.data_ptr(), y.nrows(),
      ret.data_ptr(), m);
    gpublas::err::check_ret(check, "geam");
  }
  
  
  
  /// \overload
  template <typename REAL>
  gpumat<REAL> add(const bool transx, const bool transy, const REAL alpha,
    const REAL beta, const gpumat<REAL> &x, const gpumat<REAL> &y)
  {
    err::check_card(x, y);
    
    len_t m, n;
    fml::linalgutils::matadd_params(transx, transy, x.nrows(), x.ncols(),
      y.nrows(), y.ncols(), &m, &n);
    
    auto c = x.get_card();
    gpumat<REAL> ret(c, m, n);
    add(transx, transy, alpha, beta, x, y, ret);
    return ret;
  }
}
}


#endif
