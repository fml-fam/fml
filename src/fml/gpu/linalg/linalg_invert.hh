// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_GPU_LINALG_LINALG_INVERT_H
#define FML_GPU_LINALG_LINALG_INVERT_H
#pragma once


#include <stdexcept>

#include "../../_internals/linalgutils.hh"

#include "../arch/arch.hh"

#include "../internals/gpuscalar.hh"

#include "../copy.hh"
#include "../gpumat.hh"
#include "../gpuvec.hh"

#include "linalg_err.hh"
#include "linalg_blas.hh"
#include "linalg_lu.hh"


namespace fml
{
namespace linalg
{
  /**
    @brief Compute the matrix inverse.
    
    @details The input is replaced by its inverse, computed via a PLU.
    
    @param[inout] x Input data matrix. Should be square.
    
    @impl Uses the cuSOLVER functions `cusolverDnXgetrf()` (LU) and
    `cusolverDnXgetrs()` (solve).
    
    @allocs LU pivot data is allocated internally. The inverse is computed in
    a copy before copying back to the input.
    
    @except If the matrix is non-square, a `runtime_error` exception is thrown.
    If an allocation fails, a `bad_alloc` exception will be thrown.
    
    @tparam REAL should be '__half', 'float', or 'double'.
   */
  template <typename REAL>
  void invert(gpumat<REAL> &x)
  {
    if (!x.is_square())
      throw std::runtime_error("'x' must be a square matrix");
    
    // Factor x = LU
    auto c = x.get_card();
    gpuvec<int> p(c);
    int info;
    lu(x, p, info);
    fml::linalgutils::check_info(info, "getrf");
    
    // Invert
    const len_t n = x.nrows();
    const len_t nrhs = n;
    gpumat<REAL> inv(c, n, nrhs);
    inv.fill_eye();
    
    gpuscalar<int> info_device(c, info);
    
    gpulapack_status_t check = gpulapack::getrs(c->lapack_handle(), GPUBLAS_OP_N, n,
      nrhs, x.data_ptr(), n, p.data_ptr(), inv.data_ptr(), n, info_device.data_ptr());
    
    info_device.get_val(&info);
    gpulapack::err::check_ret(check, "getrs");
    fml::linalgutils::check_info(info, "getrs");
    
    copy::gpu2gpu(inv, x);
  }
  
  
  
  /**
    @brief Compute the matrix inverse of a triangular matrix.
    
    @details The input is replaced by its inverse.
    
    @param[in] upper Should the upper triangle be used? Otherwise the lower
    triangle will be used.
    @param[in] unit_diag Is the input matrix unit diagonal?
    @param[inout] x Input data matrix. Should be square.
    
    @impl Uses the cuBLAS functions `cublasXtrsm()`.
    
    @allocs The inverse is computed in a copy.
    
    @except If the matrix is non-square, a `runtime_error` exception is thrown.
    If an allocation fails, a `bad_alloc` exception will be thrown.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  void trinv(const bool upper, const bool unit_diag, gpumat<REAL> &x)
  {
    if (!x.is_square())
      throw std::runtime_error("'x' must be a square matrix");
    
    const len_t n = x.nrows();
    gpumat<REAL> inv(x.get_card(), n, n);
    inv.fill_eye();
    
    gpublas_fillmode_t uplo = (upper ? GPUBLAS_FILL_U : GPUBLAS_FILL_L);
    gpublas_diagtype_t diag = (unit_diag ? GPUBLAS_DIAG_UNIT : GPUBLAS_DIAG_NON_UNIT);
    
    gpublas_status_t check =  gpublas::trsm(x.get_card()->blas_handle(),
      GPUBLAS_SIDE_LEFT, uplo, GPUBLAS_OP_N, diag, n, n, (REAL)1, x.data_ptr(),
      n, inv.data_ptr(), n);
      
    gpublas::err::check_ret(check, "trsm");
    copy::gpu2gpu(inv, x);
  }
}
}


#endif
