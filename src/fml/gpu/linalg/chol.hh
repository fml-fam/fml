// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_GPU_LINALG_CHOL_H
#define FML_GPU_LINALG_CHOL_H
#pragma once


#include "../../_internals/linalgutils.hh"

#include "../arch/arch.hh"

#include "../internals/gpu_utils.hh"
#include "../internals/gpuscalar.hh"

#include "../gpumat.hh"


namespace fml
{
namespace linalg
{
  /**
    @brief Compute the Choleski factorization.
    
    @details The matrix should be 1. square, 2. symmetric, 3. positive-definite.
    Failure of any of these conditions can lead to a runtime exception. The
    input is replaced by its lower-triangular Choleski factor.
    
    @param[inout] x Input data matrix, replaced by its lower-triangular Choleski
    factor.
    
    Uses the cuSOLVER function `cusolverDnXpotrf()`.
    
    @allocs Some temporary work storage is needed.
    
    @except If a (re-)allocation is triggered and fails, a `bad_alloc`
    exception will be thrown.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  void chol(gpumat<REAL> &x)
  {
    const len_t n = x.nrows();
    if (n != x.ncols())
      throw std::runtime_error("'x' must be a square matrix");
    
    auto c = x.get_card();
    const auto fill = GPUBLAS_FILL_L;
    
    int lwork;
    gpulapack_status_t check = gpulapack::potrf_buflen(c->lapack_handle(), fill, n,
      x.data_ptr(), n, &lwork);
    gpulapack::err::check_ret(check, "potrf_bufferSize");
    
    gpuvec<REAL> work(c, lwork);
    
    int info = 0;
    gpuscalar<int> info_device(c, info);
    check = gpulapack::potrf(c->lapack_handle(), fill, n, x.data_ptr(), n,
      work.data_ptr(), lwork, info_device.data_ptr());
    
    info_device.get_val(&info);
    gpulapack::err::check_ret(check, "potrf");
    if (info < 0)
      fml::linalgutils::check_info(info, "potrf");
    else if (info > 0)
      throw std::runtime_error("chol: leading minor of order " + std::to_string(info) + " is not positive definite");
    
    fml::gpu_utils::tri2zero('U', false, n, n, x.data_ptr(), n);
  }
}
}


#endif
