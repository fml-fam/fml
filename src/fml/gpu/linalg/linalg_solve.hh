// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_GPU_LINALG_LINALG_SOLVE_H
#define FML_GPU_LINALG_LINALG_SOLVE_H
#pragma once


#include <stdexcept>

#include "../../_internals/linalgutils.hh"

#include "../arch/arch.hh"

#include "../internals/gpuscalar.hh"

#include "../gpumat.hh"
#include "../gpuvec.hh"

#include "linalg_err.hh"
#include "linalg_lu.hh"


namespace fml
{
namespace linalg
{
  namespace
  {
    template <typename REAL>
    void solver(gpumat<REAL> &x, len_t ylen, len_t nrhs, REAL *y_d)
    {
      const len_t n = x.nrows();
      if (!x.is_square())
        throw std::runtime_error("'x' must be a square matrix");
      if (n != ylen)
        throw std::runtime_error("rhs 'y' must be compatible with data matrix 'x'");
      
      // Factor x = LU
      auto c = x.get_card();
      gpuvec<int> p(c);
      int info;
      lu(x, p, info);
      fml::linalgutils::check_info(info, "getrf");
      
      // Solve xb = y
      gpuscalar<int> info_device(c, info);
      
      gpulapack_status_t check = gpulapack::getrs(c->lapack_handle(), GPUBLAS_OP_N,
        n, nrhs, x.data_ptr(), n, p.data_ptr(), y_d, n, info_device.data_ptr());
      
      info_device.get_val(&info);
      gpulapack::err::check_ret(check, "getrs");
      fml::linalgutils::check_info(info, "getrs");
    }
  }
  
  /**
    @brief Solve a system of equations.
    
    @details The input is replaced by its PLU factorization.
    
    @param[inout] x Input LHS. Should be square. Overwritten by LU.
    @param[inout] y Input RHS. Overwritten by solution.
    
    @impl Uses the cuSOLVER functions `cusolverDnXgetrf()` (LU) and
    `cusolverDnXgetrs()` (solve).
    
    @allocs LU pivot data is allocated internally.
    
    @except If the matrix is non-square or if the RHS is incompatible with the
    LHS, a `runtime_error` exception is thrown. If an allocation fails, a
    `bad_alloc` exception will be thrown.
    
    @tparam REAL should be '__half', 'float', or 'double'.
   */
  template <typename REAL>
  void solve(gpumat<REAL> &x, gpuvec<REAL> &y)
  {
    err::check_card(x, y);
    solver(x, y.size(), 1, y.data_ptr());
  }
  
  /// \overload
  template <typename REAL>
  void solve(gpumat<REAL> &x, gpumat<REAL> &y)
  {
    err::check_card(x, y);
    solver(x, y.nrows(), y.ncols(), y.data_ptr());
  }
}
}


#endif
