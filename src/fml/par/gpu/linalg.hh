// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_PAR_GPU_LINALG_H
#define FML_PAR_GPU_LINALG_H
#pragma once


#include "parmat.hh"
#include "../../gpu/linalg.hh"


namespace fml
{
namespace linalg
{
  /// @brief Computes lower triangle of alpha*x^T*x
  template <typename REAL>
  void crossprod(const REAL alpha, const parmat_gpu<REAL> &x, gpumat<REAL> &ret)
  {
    err::check_card(x, ret);
    
    const len_t n = x.ncols();
    if (n != ret.nrows() || n != ret.ncols())
      ret.resize(n, n);
    
    linalg::crossprod(alpha, x.data_obj(), ret);
    
    comm r = x.get_comm();
    r.allreduce(n*n, ret.data_ptr());
  }
  
  /// \overload
  template <typename REAL>
  gpumat<REAL> crossprod(const REAL alpha, const parmat_gpu<REAL> &x)
  {
    const len_t n = x.ncols();
    gpumat<REAL> ret(x.get_card(), n, n);
    
    crossprod(alpha, x, ret);
    return ret;
  }
}
}


#endif
