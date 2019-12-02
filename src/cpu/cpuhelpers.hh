// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_CPU_CPUHELPERS_H
#define FML_CPU_CPUHELPERS_H
#pragma once


#include <stdexcept>

#include "../arraytools/src/arraytools.hpp"

#include "../cpu/cpumat.hh"
#include "../cpu/cpuvec.hh"


namespace cpuhelpers
{
  template <typename REAL_IN, typename REAL_OUT>
  void cpu2cpu(const cpuvec<REAL_IN> &cpu_in, cpuvec<REAL_OUT> &cpu_out)
  {
    cpu_out.resize(cpu_in.size());
    
    arraytools::copy(cpu_in.size(), cpu_in.data_ptr(), cpu_out.data_ptr());
  }
  
  template <typename REAL_IN, typename REAL_OUT>
  void cpu2cpu(const cpumat<REAL_IN> &cpu_in, cpumat<REAL_OUT> &cpu_out)
  {
    cpu_out.resize(cpu_in.nrows(), cpu_in.ncols());
    
    size_t len = (size_t) cpu_in.nrows() * cpu_in.ncols();
    arraytools::copy(len, cpu_in.data_ptr(), cpu_out.data_ptr());
  }
}


#endif
