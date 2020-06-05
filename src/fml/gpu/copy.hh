// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_GPU_COPY_H
#define FML_GPU_COPY_H
#pragma once


#include <stdexcept>

#include "../_internals/arraytools/src/arraytools.hpp"

#include "../cpu/cpumat.hh"
#include "../cpu/cpuvec.hh"

#include "internals/kernelfuns.hh"

#include "card.hh"
#include "gpumat.hh"
#include "gpuvec.hh"


namespace fml
{
namespace copy
{
  namespace internals
  {
    static const size_t CPLEN = 1024;
    
    template <typename REAL_IN, typename REAL_OUT>
    void copy_gpu2gpu(const len_t m, const len_t n, std::shared_ptr<card> c, dim3 griddim, dim3 blockdim, const REAL_IN *in, REAL_OUT *out)
    {
      if (std::is_same<REAL_IN, REAL_OUT>::value)
      {
        const size_t len = (size_t) m*n*sizeof(REAL_IN);
        c->mem_gpu2gpu((void*)out, (void*)in, len);
      }
      else
        fml::kernelfuns::kernel_copy<<<griddim, blockdim>>>(m, n, in, out);
    }
    
    template <typename REAL_IN, typename REAL_OUT>
    void copy_gpu2cpu(const len_t m, const len_t n, std::shared_ptr<card> c, const REAL_IN *in, REAL_OUT *out)
    {
      if (std::is_same<REAL_IN, REAL_OUT>::value)
      {
        const size_t len = (size_t) m*n*sizeof(REAL_IN);
        c->mem_gpu2cpu((void*)out, (void*)in, len);
      }
      else
      {
        size_t top = (size_t) m*n;
        size_t tmplen = std::min(top, CPLEN);
        cpuvec<REAL_IN> tmp(tmplen);
        REAL_IN *tmp_d = tmp.data_ptr();
        
        for (size_t i=0; i<top; i+=tmplen)
        {
          const size_t rem = top - i;
          const size_t copylen = std::min(tmplen, rem);
          c->mem_gpu2cpu((void*)tmp_d, (void*)(in + i), copylen*sizeof(*in));
          arraytools::copy(copylen, tmp_d, out + i);
        }
      }
    }
    
    template <typename REAL_IN, typename REAL_OUT>
    void copy_cpu2gpu(const len_t m, const len_t n, std::shared_ptr<card> c, const REAL_IN *in, REAL_OUT *out)
    {
      if (std::is_same<REAL_IN, REAL_OUT>::value)
      {
        const size_t len = (size_t) m*n*sizeof(REAL_IN);
        c->mem_cpu2gpu((void*)out, (void*)in, len);
      }
      else
      {
        size_t top = (size_t) m*n;
        size_t tmplen = std::min(top, CPLEN);
        cpuvec<REAL_OUT> tmp(tmplen);
        REAL_OUT *tmp_d = tmp.data_ptr();
        
        for (size_t i=0; i<top; i+=tmplen)
        {
          const size_t rem = top - i;
          const size_t copylen = std::min(tmplen, rem);
          arraytools::copy(copylen, in + i, tmp_d);
          c->mem_cpu2gpu((void*)(out + i), (void*)tmp_d, copylen*sizeof(*out));
        }
      }
    }
  }
  
  
  
  /**
    @brief Copy data from a GPU object to a CPU object.
    
    @param[in] gpu Input data.
    @param[out] cpu Output. Dimensions should match those of the input
    data. If not, the matrix will automatically be resized.
    
    @allocs If the output dimensions do not match those of the input, the
    internal data will automatically be re-allocated.
    
    @except If a reallocation is triggered and fails, a `bad_alloc` exception
    will be thrown.
    
    @tparam REAL_IN,REAL_OUT Should be `float` or `double`. They do not have to
    be the same type. Additionally, `REAL_IN` can be `__half`.
  */
  template <typename REAL_IN, typename REAL_OUT>
  void gpu2cpu(const gpuvec<REAL_IN> &gpu, cpuvec<REAL_OUT> &cpu)
  {
    cpu.resize(gpu.size());
    internals::copy_gpu2cpu(gpu.size(), (len_t)1, gpu.get_card(), gpu.data_ptr(), cpu.data_ptr());
  }
  
  /// \overload
  template <typename REAL>
  gpuvec<REAL> gpu2cpu(const gpuvec<REAL> &gpu)
  {
    cpuvec<REAL> cpu;
    gpu2cpu(gpu, cpu);
    
    return cpu;
  }
  
  /// \overload
  template <typename REAL_IN, typename REAL_OUT>
  void gpu2cpu(const gpumat<REAL_IN> &gpu, cpumat<REAL_OUT> &cpu)
  {
    cpu.resize(gpu.nrows(), gpu.ncols());
    internals::copy_gpu2cpu(gpu.nrows(), gpu.ncols(), gpu.get_card(), gpu.data_ptr(), cpu.data_ptr());
  }
  
  /// \overload
  template <typename REAL>
  cpumat<REAL> gpu2cpu(const gpumat<REAL> &gpu)
  {
    cpumat<REAL> cpu;
    gpu2cpu(gpu, cpu);
    
    return cpu;
  }
  
  
  
  /**
    @brief Copy data from a CPU object to a GPU object.
    
    @param[in] cpu Input data.
    @param[out] gpu Output. Dimensions should match those of the input
    data. If not, the matrix will automatically be resized.
    
    @allocs If the output dimensions do not match those of the input, the
    internal data will automatically be re-allocated.
    
    @except If a reallocation is triggered and fails, a `bad_alloc` exception
    will be thrown.
    
    @tparam REAL_IN,REAL_OUT Should be `float` or `double`. They do not have to
    be the same type. Additionally, `REAL_OUT` can be `__half`.
  */
  template <typename REAL_IN, typename REAL_OUT>
  void cpu2gpu(const cpuvec<REAL_IN> &cpu, gpuvec<REAL_OUT> &gpu)
  {
    gpu.resize(cpu.size());
    internals::copy_cpu2gpu(cpu.size(), (len_t)1, gpu.get_card(), cpu.data_ptr(), gpu.data_ptr());
  }
  
  /// \overload
  template <typename REAL_IN, typename REAL_OUT>
  void cpu2gpu(const cpumat<REAL_IN> &cpu, gpumat<REAL_OUT> &gpu)
  {
    gpu.resize(cpu.nrows(), cpu.ncols());
    internals::copy_cpu2gpu(cpu.nrows(), cpu.ncols(), gpu.get_card(), cpu.data_ptr(), gpu.data_ptr());
  }
  
  
  
  /**
    @brief Copy data from a GPU object to another.
    
    @details The GPU objects should be on the same card.
    
    @param[in] gpu_in Input data.
    @param[out] gpu_out Output. Dimensions should match those of the input
    data. If not, the matrix will automatically be resized.
    
    @allocs If the output dimensions do not match those of the input, the
    internal data will automatically be re-allocated.
    
    @except If a reallocation is triggered and fails, a `bad_alloc` exception
    will be thrown. If the objects are on different cards, a `logic_error`
    exception will be thrown.
    
    @tparam REAL_IN,REAL_OUT should be `__half`, `float`, or `double`. They do
    not have to be the same type.
  */
  template <typename REAL_IN, typename REAL_OUT>
  void gpu2gpu(const gpuvec<REAL_IN> &gpu_in, gpuvec<REAL_OUT> &gpu_out)
  {
    auto c = gpu_in.get_card();
    if (c->get_id() != gpu_out.get_card()->get_id())
      throw std::logic_error("input/output data must be on the same gpu");
    
    gpu_out.resize(gpu_in.size());
    internals::copy_gpu2gpu(gpu_in.size(), (len_t)1, c, gpu_in.get_griddim(), gpu_in.get_blockdim(), gpu_in.data_ptr(), gpu_out.data_ptr());
  }
  
  /// \overload
  template <typename REAL>
  gpuvec<REAL> gpu2gpu(const gpuvec<REAL> &gpu_in)
  {
    gpuvec<REAL> gpu_out(gpu_in.get_card());
    gpu2gpu(gpu_in, gpu_out);
    
    return gpu_out;
  }
  
  /// \overload
  template <typename REAL_IN, typename REAL_OUT>
  void gpu2gpu(const gpumat<REAL_IN> &gpu_in, gpumat<REAL_OUT> &gpu_out)
  {
    auto c = gpu_in.get_card();
    if (c->get_id() != gpu_out.get_card()->get_id())
      throw std::logic_error("input/output data must be on the same gpu");
    
    gpu_out.resize(gpu_in.nrows(), gpu_in.ncols());
    internals::copy_gpu2gpu(gpu_in.nrows(), gpu_in.ncols(), c, gpu_in.get_griddim(), gpu_in.get_blockdim(), gpu_in.data_ptr(), gpu_out.data_ptr());
  }
  
  /// \overload
  template <typename REAL>
  gpumat<REAL> gpu2gpu(const gpumat<REAL> &gpu_in)
  {
    gpumat<REAL> gpu_out(gpu_in.get_card());
    gpu2gpu(gpu_in, gpu_out);
    
    return gpu_out;
  }
}
}


#endif
