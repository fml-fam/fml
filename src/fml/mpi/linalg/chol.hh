// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_MPI_LINALG_CHOL_H
#define FML_MPI_LINALG_CHOL_H
#pragma once


#include <stdexcept>

#include "../../_internals/linalgutils.hh"

#include "../internals/bcutils.hh"
#include "../internals/mpi_utils.hh"

#include "../mpimat.hh"

#include "internals/scalapack.hh"


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
    
    @impl Uses the ScaLAPACK function `pXpotrf()`.
    
    @allocs Some temporary work storage is needed.
    
    @except If a (re-)allocation is triggered and fails, a `bad_alloc`
    exception will be thrown.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  void chol(mpimat<REAL> &x)
  {
    const len_t n = x.nrows();
    if (n != x.ncols())
      throw std::runtime_error("'x' must be a square matrix");
    
    int info = 0;
    fml::scalapack::potrf('L', n, x.data_ptr(), x.desc_ptr(), &info);
    
    if (info < 0)
      fml::linalgutils::check_info(info, "potrf");
    else if (info > 0)
      throw std::runtime_error("chol: leading minor of order " + std::to_string(info) + " is not positive definite");
    
    fml::mpi_utils::tri2zero('U', false, x.get_grid(), n, n, x.data_ptr(), x.desc_ptr());
  }
}
}


#endif
