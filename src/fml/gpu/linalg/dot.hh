// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_GPU_LINALG_DOT_H
#define FML_GPU_LINALG_DOT_H
#pragma once


#include <stdexcept>

#include "../../_internals/linalgutils.hh"

#include "../arch/arch.hh"

#include "../gpumat.hh"

#include "internals/err.hh"


namespace fml
{
namespace linalg
{
  /**
    @brief Computes the dot product of two vectors, i.e. the sum of the product
    of the elements.
    
    @details NOTE: if the vectors are of different length, the dot product will
    use only the indices of the smaller-sized vector.
    
    @param[in] x,y Vectors.
    
    @return The dot product.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  REAL dot(const gpuvec<REAL> &x, const gpuvec<REAL> &y)
  {
    err::check_card(x, y);
    const len_t len = std::min(x.size(), y.size());
    
    len_t m, n, k;
    fml::linalgutils::matmult_params(true, false, len, 1, len, 1, &m, &n, &k);
    
    REAL d;
    gpuscalar<REAL> d_device(x.get_card());
    gpublas_status_t check = gpublas::gemm(x.get_card()->blas_handle(),
      GPUBLAS_OP_T, GPUBLAS_OP_N, m, n, k, (REAL)1, x.data_ptr(), len,
      y.data_ptr(), len, (REAL)0, d_device.data_ptr(), 1);
    gpublas::err::check_ret(check, "gemm");
    
    d_device.get_val(&d);
    
    return d;
  }
  
  
  
  /// \overload
  template <typename REAL>
  REAL dot(const gpuvec<REAL> &x)
  {
    return dot(x, x);
  }
}
}


#endif
