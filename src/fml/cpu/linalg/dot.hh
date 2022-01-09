// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_CPU_LINALG_DOT_H
#define FML_CPU_LINALG_DOT_H
#pragma once


#include "../cpuvec.hh"


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
    
    @tparam REAL should be 'float' or 'double' ('int' is also ok).
   */
  template <typename REAL>
  REAL dot(const cpuvec<REAL> &x, const cpuvec<REAL> &y)
  {
    const len_t n = std::min(x.size(), y.size());
    const REAL *x_d = x.data_ptr();
    const REAL *y_d = y.data_ptr();
    
    REAL d = 0;
    #pragma omp simd reduction(+:d)
    for (len_t i=0; i<n; i++)
      d += x_d[i] * y_d[i];
    
    return d;
  }
  
  
  
  /// \overload
  template <typename REAL>
  REAL dot(const cpuvec<REAL> &x)
  {
    return dot(x, x);
  }
}
}


#endif
