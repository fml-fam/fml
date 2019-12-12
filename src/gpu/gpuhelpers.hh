// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_GPU_GPUHELPERS_H
#define FML_GPU_GPUHELPERS_H
#pragma once


#include <stdexcept>

#include "../arraytools/src/arraytools.hpp"

#include "../cpu/cpumat.hh"
#include "../cpu/cpuvec.hh"

#include "internals/kernelfuns.hh"

#include "card.hh"
#include "gpumat.hh"
#include "gpuvec.hh"


/// @brief GPU class helpers.
namespace gpuhelpers
{
  /**
    @brief Initialize a new card.
    
    @param[in] id GPU id number.
    @return Shared pointer to initialized card object.
    
    @except If the GPU can not be initialized, or if the allocation of one of the
    handles fails, the method will throw a 'runtime_error' exception.
  */
  inline std::shared_ptr<card> new_card(int id)
  {
    return std::make_shared<card>(id);
  }
  
  
  
  namespace
  {
    template <typename REAL_IN, typename REAL_OUT>
    void copy_gpu2gpu(const len_t m, const len_t n, std::shared_ptr<card> c, dim3 griddim, dim3 blockdim, const REAL_IN *in, REAL_OUT *out)
    {
      if (std::is_same<REAL_IN, REAL_OUT>::value)
      {
        const size_t len = (size_t) m*n*sizeof(REAL_IN);
        c->mem_gpu2gpu((void*)out, (void*)in, len);
      }
      else
        kernelfuns::kernel_copy<<<griddim, blockdim>>>(m, n, in, out);
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
        cpuvec<REAL_OUT> cp(kernelfuns::CPLEN);
        REAL_OUT *cp_d = cp.data_ptr();
        
        size_t top = (size_t) m*n;
        for (size_t i=0; i<top; i+=kernelfuns::CPLEN)
        {
          const size_t start = top - i;
          const size_t copylen = std::min(kernelfuns::CPLEN, start);
          c->mem_cpu2gpu((void*)cp_d, (void*)(in + start), copylen);
          arraytools::copy(copylen, cp_d, out + start);
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
        cpuvec<REAL_OUT> cp(kernelfuns::CPLEN);
        REAL_OUT *cp_d = cp.data_ptr();
        
        size_t top = (size_t) m*n;
        for (size_t i=0; i<top; i+=kernelfuns::CPLEN)
        {
          const size_t start = top - i;
          const size_t copylen = std::min(kernelfuns::CPLEN, start);
          arraytools::copy(copylen, in + start, cp_d);
          c->mem_cpu2gpu((void*)(out + start), (void*)cp_d, copylen);
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
    copy_gpu2cpu(gpu.size(), (len_t)1, gpu.get_card(), gpu.data_ptr(), cpu.data_ptr());
  }
  
  /// \overload
  template <typename REAL_IN, typename REAL_OUT>
  void gpu2cpu(const gpumat<REAL_IN> &gpu, cpumat<REAL_OUT> &cpu)
  {
    cpu.resize(gpu.nrows(), gpu.ncols());
    copy_gpu2cpu(gpu.nrows(), gpu.ncols(), gpu.get_card(), gpu.data_ptr(), cpu.data_ptr());
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
    
    copy_cpu2gpu(cpu.size(), (len_t)1, gpu.get_card(), cpu.data_ptr(), gpu.data_ptr());
  }
  
  /// \overload
  template <typename REAL_IN, typename REAL_OUT>
  void cpu2gpu(const cpumat<REAL_IN> &cpu, gpumat<REAL_OUT> &gpu)
  {
    gpu.resize(cpu.nrows(), cpu.ncols());
    copy_cpu2gpu(cpu.nrows(), cpu.ncols(), gpu.get_card(), cpu.data_ptr(), gpu.data_ptr());
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
    copy_gpu2gpu(gpu_in.size(), (len_t)1, c, gpu_in.get_griddim(), gpu_in.get_blockdim(), gpu_in.data_ptr(), gpu_out.data_ptr());
  }
  
  /// \overload
  template <typename REAL_IN, typename REAL_OUT>
  void gpu2gpu(const gpumat<REAL_IN> &gpu_in, gpumat<REAL_OUT> &gpu_out)
  {
    auto c = gpu_in.get_card();
    if (c->get_id() != gpu_out.get_card()->get_id())
      throw std::logic_error("input/output data must be on the same gpu");
    
    gpu_out.resize(gpu_in.nrows(), gpu_in.ncols());
    copy_gpu2gpu(gpu_in.nrows(), gpu_in.ncols(), c, gpu_in.get_griddim(), gpu_in.get_blockdim(), gpu_in.data_ptr(), gpu_out.data_ptr());
  }
}


#endif
