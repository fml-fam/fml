// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_CPU_LINALG_TRACE_H
#define FML_CPU_LINALG_TRACE_H
#pragma once


#include "../../_internals/omp.hh"

#include "../cpumat.hh"


namespace fml
{
namespace linalg
{
  /**
    @brief Computes the trace, i.e. the sum of the diagonal.
    
    @param[in] x Input data matrix.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  REAL trace(const cpumat<REAL> &x)
  {
    const REAL *x_d = x.data_ptr();
    const len_t m = x.nrows();
    const len_t minmn = std::min(m, x.ncols());
    
    REAL tr = 0;
    #pragma omp parallel for simd if(minmn > fml::omp::OMP_MIN_SIZE) reduction(+:tr)
    for (len_t i=0; i<minmn; i++)
      tr += x_d[i + i*m];
    
    return tr;
  }
}
}


#endif
