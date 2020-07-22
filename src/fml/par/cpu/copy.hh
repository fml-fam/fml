// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_PAR_CPU_COPY_H
#define FML_PAR_CPU_COPY_H
#pragma once


#include <stdexcept>

#include "../comm.hh"
#include "parmat.hh"

#include "../../cpu/cpumat.hh"
#include "../../cpu/copy.hh"


namespace fml
{
namespace copy
{
  /**
    @brief Copy data from an PAR object to a CPU object.
    
    @details Every process receives a copy of the data. The number of rows of
    the input should fit in the storage of a `len_t`.
    
    @param[in] par Input data.
    @param[out] cpu Output.
    
    @allocs If the output dimensions do not match those of the input, the
    internal data will automatically be re-allocated.
    
    @except If a reallocation is triggered and fails, a `bad_alloc` exception
    will be thrown.
    
    @comm The method will communicate across all processes in the communicator.
    
    @tparam REAL_IN,REAL_OUT Should be `float` or `double`. They do not have to
    be the same type.
  */
  template <typename REAL_IN, typename REAL_OUT>
  void par2cpu(const parmat_cpu<REAL_IN> &par, cpumat<REAL_OUT> &cpu)
  {
    const len_t m = (len_t) par.nrows();
    const len_t n = par.ncols();
    
    cpu.resize(m, n);
    cpu.fill_zero();
    
    const len_t m_local = par.nrows_local();
    const len_t nb4 = (len_t) par.nrows_before();
    REAL_IN *par_d = par.data_obj().data_ptr();
    REAL_OUT *cpu_d = cpu.data_ptr();
    
    for (len_t j=0; j<n; j++)
    {
      for (len_t i=0; i<m_local; i++)
        (REAL_OUT) cpu_d[i+nb4 + m*j] = par_d[i + m_local*j];
    }
    
    par.get_comm().allreduce(m*n, cpu.data_ptr());
  }
  
  
  
  /**
    @brief Copy data from an CPU object to a PAR object.
    
    @details Every process should have a copy of the data on input.
    
    @param[in] par Input data.
    @param[out] cpu Output.
    
    @allocs If the output dimensions do not match those of the input, the
    internal data will automatically be re-allocated.
    
    @except If a reallocation is triggered and fails, a `bad_alloc` exception
    will be thrown.
    
    @tparam REAL_IN,REAL_OUT Should be `float` or `double`. They do not have to
    be the same type.
  */
  template <typename REAL_IN, typename REAL_OUT>
  void cpu2par(const cpumat<REAL_IN> &cpu, parmat_cpu<REAL_OUT> &par)
  {
    const len_t m = cpu.nrows();
    const len_t n = cpu.ncols();
    
    par.resize(m, n);
    
    const len_t m_local = par.nrows_local();
    const len_t nb4 = (len_t) par.nrows_before();
    REAL_IN *cpu_d = cpu.data_ptr();
    REAL_OUT *par_d = par.data_obj().data_ptr();
    
    for (len_t j=0; j<n; j++)
    {
      for (len_t i=0; i<m_local; i++)
        par_d[i + m_local*j] = (REAL_OUT) cpu_d[i+nb4 + m*j];
    }
  }
  
  
  
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
  void par2par(const parmat_cpu<REAL_IN> &par_in, parmat_cpu<REAL_OUT> &par_out)
  {
    par_out.resize(par_in.nrows(), par_in.ncols());
    cpu2cpu(par_in.data_obj(), par_out.data_obj());
  }
}
}


#endif
