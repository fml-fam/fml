// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_MPI_LINALG_LINALG_INVERT_H
#define FML_MPI_LINALG_LINALG_INVERT_H
#pragma once


#include <stdexcept>

#include "../../_internals/linalgutils.hh"
#include "../../cpu/cpuvec.hh"

#include "../internals/bcutils.hh"
#include "../internals/mpi_utils.hh"

#include "../mpimat.hh"

#include "linalg_err.hh"
#include "linalg_lu.hh"
#include "scalapack.hh"


namespace fml
{
namespace linalg
{
  /**
    @brief Compute the matrix inverse.
    
    @details The input is replaced by its inverse, computed via a PLU.
    
    @param[inout] x Input data matrix. Should be square.
    
    @impl Uses the ScaLAPACK functions `pXgetrf()` (LU) and `pXgetri()`
    (solve).
    
    @allocs LU pivot data is allocated internally.
    
    @except If the matrix is non-square, a `runtime_error` exception is thrown.
    If an allocation fails, a `bad_alloc` exception will be thrown.
    
    @comm The method will communicate across all processes in the BLACS grid.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  void invert(mpimat<REAL> &x)
  {
    if (!x.is_square())
      throw std::runtime_error("'x' must be a square matrix");
    
    // Factor x = LU
    cpuvec<int> p;
    int info;
    lu(x, p, info);
    fml::linalgutils::check_info(info, "getrf");
    
    // Invert
    const len_t n = x.nrows();
    REAL tmp;
    int liwork;
    fml::scalapack::getri(n, x.data_ptr(), x.desc_ptr(), p.data_ptr(), &tmp, -1, &liwork, -1, &info);
    int lwork = std::max(1, (int)tmp);
    cpuvec<REAL> work(lwork);
    cpuvec<int> iwork(liwork);
    
    fml::scalapack::getri(n, x.data_ptr(), x.desc_ptr(), p.data_ptr(), work.data_ptr(), lwork, iwork.data_ptr(), liwork, &info);
    fml::linalgutils::check_info(info, "getri");
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
  void trinv(const bool upper, const bool unit_diag, mpimat<REAL> &x)
  {
    if (!x.is_square())
      throw std::runtime_error("'x' must be a square matrix");
    
    int info;
    char uplo = (upper ? 'U' : 'L');
    char diag = (unit_diag ? 'U' : 'N');
    fml::scalapack::trtri(uplo, diag, x.nrows(), x.data_ptr(), x.desc_ptr(), &info);
    fml::linalgutils::check_info(info, "trtri");
  }
}
}


#endif
