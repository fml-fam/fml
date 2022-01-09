// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_CPU_LINALG_LINALG_INVERT_H
#define FML_CPU_LINALG_LINALG_INVERT_H
#pragma once


#include <cmath>
#include <stdexcept>

#include "../../_internals/linalgutils.hh"
#include "../../_internals/omp.hh"

#include "../internals/cpu_utils.hh"

#include "../cpumat.hh"
#include "../cpuvec.hh"

#include "internals/lapack.hh"
#include "lu.hh"


namespace fml
{
namespace linalg
{
  /**
    @brief Compute the matrix inverse.
    
    @details The input is replaced by its inverse, computed via a PLU.
    
    @param[inout] x Input data matrix. Should be square.
    
    @impl Uses the LAPACK functions `Xgetrf()` (LU) and `Xgetri()` (solve).
    
    @allocs LU pivot data is allocated internally.
    
    @except If the matrix is non-square, a `runtime_error` exception is thrown.
    If an allocation fails, a `bad_alloc` exception will be thrown.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  void invert(cpumat<REAL> &x)
  {
    const len_t n = x.nrows();
    if (!x.is_square())
      throw std::runtime_error("'x' must be a square matrix");
    
    // Factor x = LU
    cpuvec<int> p;
    int info;
    lu(x, p, info);
    linalgutils::check_info(info, "getrf");
    
    // Invert
    REAL tmp;
    lapack::getri(n, x.data_ptr(), n, p.data_ptr(), &tmp, -1, &info);
    int lwork = (int) tmp;
    cpuvec<REAL> work(lwork);
    
    lapack::getri(n, x.data_ptr(), n, p.data_ptr(), work.data_ptr(), lwork, &info);
    linalgutils::check_info(info, "getri");
  }
  
  
  
  /**
    @brief Compute the matrix inverse of a triangular matrix.
    
    @details The input is replaced by its inverse.
    
    @param[in] upper Should the upper triangle be used? Otherwise the lower
    triangle will be used.
    @param[in] unit_diag Is the input matrix unit diagonal?
    @param[inout] x Input data matrix. Should be square.
    
    @impl Uses the LAPACK functions `Xtrtri()`.
    
    @except If the matrix is non-square, a `runtime_error` exception is thrown.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  void trinv(const bool upper, const bool unit_diag, cpumat<REAL> &x)
  {
    if (!x.is_square())
      throw std::runtime_error("'x' must be a square matrix");
    
    const len_t n = x.nrows();
    
    int info;
    char uplo = (upper ? 'U' : 'L');
    char diag = (unit_diag ? 'U' : 'N');
    lapack::trtri(uplo, diag, x.nrows(), x.data_ptr(), n, &info);
    linalgutils::check_info(info, "trtri");
    
    uplo = (uplo == 'U' ? 'L' : 'U');
    cpu_utils::tri2zero(uplo, false, n, n, x.data_ptr(), n);
  }
}
}


#endif
