// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_GPU_LINALG_TRACE_H
#define FML_GPU_LINALG_TRACE_H
#pragma once


#include "../arch/arch.hh"

#include "../internals/gpuscalar.hh"

#include "../gpumat.hh"

#include "lu.hh"


namespace fml
{
namespace linalg
{
  namespace
  {
    template <typename REAL>
    __global__ void kernel_trace(const len_t m, const len_t n, const REAL *data, REAL *tr)
    {
      int i = blockDim.x*blockIdx.x + threadIdx.x;
      int j = blockDim.y*blockIdx.y + threadIdx.y;
      
      if (i < m && j < n && i == j)
        atomicAdd(tr, data[i + m*i]);
    }
  }
  
  
  
  /**
    @brief Computes the trace, i.e. the sum of the diagonal.
    
    @param[in] x Input data matrix.
    
    @tparam REAL should be '__half', 'float', or 'double'.
   */
  template <typename REAL>
  REAL trace(const gpumat<REAL> &x)
  {
    const len_t m = x.nrows();
    const len_t n = x.ncols();
    auto c = x.get_card();
    
    REAL tr = 0;
    gpuscalar<REAL> tr_gpu(c, tr);
    
    kernel_trace<<<x.get_griddim(), x.get_blockdim()>>>(m, n, x.data_ptr(),
      tr_gpu.data_ptr());
    
    tr_gpu.get_val(&tr);
    c->check();
    
    return tr;
  }
}
}


#endif
