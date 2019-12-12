// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_CPU_CPUHELPERS_H
#define FML_CPU_CPUHELPERS_H
#pragma once


#include <stdexcept>

#include "../arraytools/src/arraytools.hpp"

#include "cpumat.hh"
#include "cpuvec.hh"


/// @brief CPU class helpers.
namespace cpuhelpers
{
  /**
    @brief Copy data from a CPU object to another.
    
    @param[in] cpu_in Input data.
    @param[out] cpu_out Output. Dimensions should match those of the input
    data. If not, the matrix will automatically be resized.
    
    @allocs If the output dimensions do not match those of the input, the
    internal data will automatically be re-allocated.
    
    @except If a reallocation is triggered and fails, a `bad_alloc` exception
    will be thrown.
    
    @tparam REAL_IN,REAL_OUT Should be `float` or `double`. They do not have to
    be the same type.
  */
  template <typename REAL_IN, typename REAL_OUT>
  void cpu2cpu(const cpuvec<REAL_IN> &cpu_in, cpuvec<REAL_OUT> &cpu_out)
  {
    cpu_out.resize(cpu_in.size());
    
    arraytools::copy(cpu_in.size(), cpu_in.data_ptr(), cpu_out.data_ptr());
  }
  
  /// \overload
  template <typename REAL>
  cpuvec<REAL> cpu2cpu(const cpuvec<REAL> &cpu_in)
  {
    cpuvec<REAL> cpu_out;
    cpu2cpu(cpu_in, cpu_out);
    
    return cpu_out;
  }
  
  /// \overload
  template <typename REAL_IN, typename REAL_OUT>
  void cpu2cpu(const cpumat<REAL_IN> &cpu_in, cpumat<REAL_OUT> &cpu_out)
  {
    cpu_out.resize(cpu_in.nrows(), cpu_in.ncols());
    
    size_t len = (size_t) cpu_in.nrows() * cpu_in.ncols();
    arraytools::copy(len, cpu_in.data_ptr(), cpu_out.data_ptr());
  }
  
  /// \overload
  template <typename REAL>
  cpumat<REAL> cpu2cpu(const cpumat<REAL> &cpu_in)
  {
    cpumat<REAL> cpu_out;
    cpu2cpu(cpu_in, cpu_out);
    
    return cpu_out;
  }
}


#endif
