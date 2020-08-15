// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_PAR_GPU_COPY_H
#define FML_PAR_GPU_COPY_H
#pragma once


#include <stdexcept>

#include "../comm.hh"
#include "parmat.hh"

#include "../../gpu/cpumat.hh"
#include "../../gpu/copy.hh"


namespace fml
{
namespace copy
{
  /**
    @brief Copy data from a PAR object to another.
    
    @param[in] par_in Input data.
    @param[out] par_out Output.
    
    @allocs If the output dimensions do not match those of the input, the
    internal data will automatically be re-allocated.
    
    @except If a reallocation is triggered and fails, a `bad_alloc` exception
    will be thrown.
    
    @tparam REAL_IN,REAL_OUT Should be `float` or `double`. They do not have to
    be the same type.
  */
  template <typename REAL_IN, typename REAL_OUT>
  void par2par(const parmat_gpu<REAL_IN> &par_in, parmat_gpu<REAL_OUT> &par_out)
  {
    par_out.resize(par_in.nrows(), par_in.ncols());
    gpu2gpu(par_in.data_obj(), par_out.data_obj());
  }
}
}


#endif
