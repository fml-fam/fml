// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_GPU_LINALG_LINALG_LU_H
#define FML_GPU_LINALG_LINALG_LU_H
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
  /**
    @brief Computes the PLU factorization with partial pivoting.
    
    @details The input is replaced by its LU factorization, with L
    unit-diagonal.
    
    @param[inout] x Input data matrix, replaced by its LU factorization.
    @param[out] p Vector of pivots, representing the diagonal matrix P in the
    PLU.
    @param[out] info The LAPACK return number.
    
    @impl Uses the cuSOLVER function `cusolverDnXgetrf()`.
    
    @allocs If the pivot vector is inappropriately sized, it will automatically
    be re-allocated.
    
    @except If a reallocation is triggered and fails, a `bad_alloc` exception
    will be thrown.
    
    @tparam REAL should be '__half', 'float', or 'double'.
   */
  template <typename REAL>
  void lu(gpumat<REAL> &x, gpuvec<int> &p, int &info)
  {
    err::check_card(x, p);
    
    info = 0;
    const len_t m = x.nrows();
    const len_t n = x.ncols();
    auto c = x.get_card();
    
    const len_t lipiv = std::min(m, n);
    if (!p.get_card()->valid_card())
      p.inherit(c);
    
    p.resize(lipiv);
    
    #if defined(FML_GPULAPACK_VENDOR)
      int lwork;
      gpulapack_status_t check = gpulapack::getrf_buflen(c->lapack_handle(), m,
        n, x.data_ptr(), m, &lwork);
      gpulapack::err::check_ret(check, "getrf_bufferSize");
      
      gpuvec<REAL> work(c, lwork);
      gpuscalar<int> info_device(c, info);
      
      check = gpulapack::getrf(c->lapack_handle(), m, n, x.data_ptr(), m,
        work.data_ptr(), p.data_ptr(), info_device.data_ptr());
      
      info_device.get_val(&info);
      gpulapack::err::check_ret(check, "getrf");
    #elif defined(FML_GPULAPACK_MAGMA)
      cpuvec<int> p_cpu(lipiv);
      gpulapack::getrf(m, n, x.data_ptr(), m, p_cpu.data_ptr(), &info);
      copy::cpu2gpu(p_cpu, p);
    #else
      #error "Unsupported GPU lapack"
    #endif
  }
  
  /// \overload
  template <typename REAL>
  void lu(gpumat<REAL> &x)
  {
    gpuvec<int> p(x.get_card());
    int info;
    
    lu(x, p, info);
    
    fml::linalgutils::check_info(info, "getrf");
  }
}
}


#endif
