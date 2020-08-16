// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_CPU_LINALG_LINALG_EIGEN_H
#define FML_CPU_LINALG_LINALG_EIGEN_H
#pragma once


#include <cmath>
#include <stdexcept>

#include "../../_internals/linalgutils.hh"
#include "../../_internals/omp.hh"

#include "../cpumat.hh"
#include "../cpuvec.hh"

#include "lapack.hh"


namespace fml
{
namespace linalg
{
  namespace
  {
    template <typename REAL>
    int eig_sym_internals(const bool only_values, cpumat<REAL> &x,
      cpuvec<REAL> &values, cpumat<REAL> &vectors)
    {
      if (!x.is_square())
        throw std::runtime_error("'x' must be a square matrix");
      
      int info = 0;
      int nfound;
      char jobz;
      
      len_t n = x.nrows();
      values.resize(n);
      cpuvec<int> support;
      
      if (only_values)
        jobz = 'N';
      else
      {
        jobz = 'V';
        vectors.resize(n, n);
        support.resize(2*n);
      }
      
      REAL worksize;
      int lwork, liwork;
      fml::lapack::syevr(jobz, 'A', 'L', n, x.data_ptr(), n, (REAL) 0.f, (REAL) 0.f,
        0, 0, (REAL) 0.f, &nfound, values.data_ptr(), vectors.data_ptr(), n,
        support.data_ptr(), &worksize, -1, &liwork, -1,
        &info);
      
      lwork = (int) worksize;
      cpuvec<REAL> work(lwork);
      cpuvec<int> iwork(liwork);
      
      fml::lapack::syevr(jobz, 'A', 'L', n, x.data_ptr(), n, (REAL) 0.f, (REAL) 0.f,
        0, 0, (REAL) 0.f, &nfound, values.data_ptr(), vectors.data_ptr(), n,
        support.data_ptr(), work.data_ptr(), lwork, iwork.data_ptr(), liwork,
        &info);
      
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
    
    @impl Uses the LAPACK functions `Xsyevr()`.
    
    @allocs If any output's dimension is inappropriately sized, it will
    automatically be re-allocated.
    
    @except If the matrix is non-square, a `runtime_error` exception is thrown.
    If an allocation fails, a `bad_alloc` exception will be thrown.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  void eigen_sym(cpumat<REAL> &x, cpuvec<REAL> &values)
  {
    cpumat<REAL> ignored;
    
    int info = eig_sym_internals(true, x, values, ignored);
    fml::linalgutils::check_info(info, "syevr");
  }
  
  /// \overload
  template <typename REAL>
  void eigen_sym(cpumat<REAL> &x, cpuvec<REAL> &values, cpumat<REAL> &vectors)
  {
    int info = eig_sym_internals(false, x, values, vectors);
    fml::linalgutils::check_info(info, "syevr");
  }
}
}


#endif
