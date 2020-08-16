// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_MPI_LINALG_LINALG_EIGEN_H
#define FML_MPI_LINALG_LINALG_EIGEN_H
#pragma once


#include <stdexcept>

#include "../../_internals/linalgutils.hh"
#include "../../cpu/cpuvec.hh"

#include "../internals/bcutils.hh"
#include "../internals/mpi_utils.hh"

#include "../copy.hh"
#include "../mpimat.hh"

#include "linalg_blas.hh"
#include "linalg_err.hh"
#include "scalapack.hh"


namespace fml
{
namespace linalg
{
  namespace
  {
    template <typename REAL>
    int eig_sym_internals(const bool only_values, mpimat<REAL> &x,
      cpuvec<REAL> &values, mpimat<REAL> &vectors)
    {
      if (!x.is_square())
        throw std::runtime_error("'x' must be a square matrix");
      
      int info = 0;
      int val_found, vec_found;
      char jobz;
      
      len_t n = x.nrows();
      values.resize(n);
      
      if (only_values)
        jobz = 'N';
      else
      {
        jobz = 'V';
        vectors.resize(n, n, x.bf_rows(), x.bf_cols());
      }
      
      REAL worksize;
      int lwork, liwork;
      
      fml::scalapack::syevr(jobz, 'A', 'L', n, x.data_ptr(), x.desc_ptr(),
        (REAL) 0.f, (REAL) 0.f, 0, 0, &val_found, &vec_found,
        values.data_ptr(), vectors.data_ptr(), vectors.desc_ptr(),
        &worksize, -1, &liwork, -1, &info);
      
      lwork = (int) worksize;
      cpuvec<REAL> work(lwork);
      cpuvec<int> iwork(liwork);
      
      fml::scalapack::syevr(jobz, 'A', 'L', n, x.data_ptr(), x.desc_ptr(),
        (REAL) 0.f, (REAL) 0.f, 0, 0, &val_found, &vec_found,
        values.data_ptr(), vectors.data_ptr(), vectors.desc_ptr(),
        work.data_ptr(), lwork, iwork.data_ptr(), liwork, &info);
      
      return info;
    }
  }
  
  /**
    @brief Compute the eigenvalues and optionally the eigenvectors for a
    symmetric matrix.
    
    @details The input data is overwritten.
    
    @param[inout] x Input data matrix. Should be square.
    @param[out] values Eigenvalues.
    @param[out] vectors Eigenvectors.
    
    @impl Uses the ScaLAPACK functions `pXsyevr()`.
    
    @allocs If any output's dimension is inappropriately sized, it will
    automatically be re-allocated.
    
    @except If the matrix is non-square, a `runtime_error` exception is thrown.
    If an allocation fails, a `bad_alloc` exception will be thrown.
    
    @comm The method will communicate across all processes in the BLACS grid.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  void eigen_sym(mpimat<REAL> &x, cpuvec<REAL> &values)
  {
    mpimat<REAL> ignored(x.get_grid());
    
    int info = eig_sym_internals(true, x, values, ignored);
    fml::linalgutils::check_info(info, "syevr");
  }
  
  /// \overload
  template <typename REAL>
  void eigen_sym(mpimat<REAL> &x, cpuvec<REAL> &values, mpimat<REAL> &vectors)
  {
    err::check_grid(x, vectors);
    
    int info = eig_sym_internals(false, x, values, vectors);
    fml::linalgutils::check_info(info, "syevr");
  }
}
}


#endif
