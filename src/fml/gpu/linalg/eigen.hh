// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_GPU_LINALG_EIGEN_H
#define FML_GPU_LINALG_EIGEN_H
#pragma once


#include <stdexcept>

#include "../../_internals/linalgutils.hh"

#include "../arch/arch.hh"

#include "../internals/gpuscalar.hh"

#include "../copy.hh"
#include "../gpumat.hh"
#include "../gpuvec.hh"

#include "internals/err.hh"


namespace fml
{
namespace linalg
{
  namespace
  {
    template <typename REAL>
    int eig_sym_internals(const bool only_values, gpumat<REAL> &x,
      gpuvec<REAL> &values, gpumat<REAL> &vectors)
    {
      if (!x.is_square())
        throw std::runtime_error("'x' must be a square matrix");
      
      auto c = x.get_card();
      
      len_t n = x.nrows();
      values.resize(n);
      
      cusolverEigMode_t jobz;
      if (only_values)
        jobz = CUSOLVER_EIG_MODE_NOVECTOR;
      else
        jobz = CUSOLVER_EIG_MODE_VECTOR;
      
      int lwork;
      gpulapack_status_t check = gpulapack::syevd_buflen(c->lapack_handle(), jobz,
        GPUBLAS_FILL_L, n, x.data_ptr(), n, values.data_ptr(), &lwork);
      gpulapack::err::check_ret(check, "syevd_bufferSize");
      
      gpuvec<REAL> work(c, lwork);
      
      int info = 0;
      gpuscalar<int> info_device(c, info);
      
      check = gpulapack::syevd(c->lapack_handle(), jobz, GPUBLAS_FILL_L,
        n, x.data_ptr(), n, values.data_ptr(), work.data_ptr(), lwork,
        info_device.data_ptr());
      
      info_device.get_val(&info);
      gpulapack::err::check_ret(check, "syevd");
      fml::linalgutils::check_info(info, "syevd");
      
      if (!only_values)
      {
        vectors.resize(n, n);
        copy::gpu2gpu(x, vectors);
      }
      
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
    
    @impl Uses the cuSOLVER functions `cusolverDnXsyevd()`.
    
    @allocs If any output's dimension is inappropriately sized, it will
    automatically be re-allocated.
    
    @except If the matrix is non-square, a `runtime_error` exception is thrown.
    If an allocation fails, a `bad_alloc` exception will be thrown.
    
    @tparam REAL should be '__half', 'float', or 'double'.
   */
  template <typename REAL>
  void eigen_sym(gpumat<REAL> &x, gpuvec<REAL> &values)
  {
    err::check_card(x, values);
    gpumat<REAL> ignored(x.get_card());
    
    int info = eig_sym_internals(true, x, values, ignored);
    fml::linalgutils::check_info(info, "syevd");
  }
  
  /// \overload
  template <typename REAL>
  void eigen_sym(gpumat<REAL> &x, gpuvec<REAL> &values, gpumat<REAL> &vectors)
  {
    err::check_card(x, values, values);
    
    int info = eig_sym_internals(false, x, values, vectors);
    fml::linalgutils::check_info(info, "syevd");
  }
}
}


#endif
