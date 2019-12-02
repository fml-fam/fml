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

#include "gpumat.hh"
#include "gpuvec.hh"


namespace gpuhelpers
{
  inline std::shared_ptr<card> new_card(int id)
  {
    return std::make_shared<card>(id);
  }
  
  
  
  namespace
  {
    static const size_t CPLEN = 1024;
    
    
    
    static __global__ void kernel_copy(len_t m, len_t n, __half *in, float *out)
    {
      int i = blockDim.x*blockIdx.x + threadIdx.x;
      int j = blockDim.y*blockIdx.y + threadIdx.y;
      
      if (i < m && j < n)
        out[i + m*j] = __half2float(in[i + m*j]);
    }
    
    static __global__ void kernel_copy(len_t m, len_t n, float *in, __half *out)
    {
      int i = blockDim.x*blockIdx.x + threadIdx.x;
      int j = blockDim.y*blockIdx.y + threadIdx.y;
      
      if (i < m && j < n)
        out[i + m*j] = __float2half(in[i + m*j]);
    }
    
    template <typename REAL_IN, typename REAL_OUT>
    __global__ void kernel_copy(len_t m, len_t n, REAL_IN *in, REAL_OUT *out)
    {
      int i = blockDim.x*blockIdx.x + threadIdx.x;
      int j = blockDim.y*blockIdx.y + threadIdx.y;
      
      if (i < m && j < n)
        out[i + m*j] = (REAL_OUT) in[i + m*j];
    }
    
    
    
    template <typename REAL_IN, typename REAL_OUT>
    void copy_gpu2gpu(const len_t m, const len_t n, std::shared_ptr<card> c, dim3 griddim, dim3 blockdim, const REAL_IN *in, REAL_OUT *out)
    {
      if (std::is_same<REAL_IN, REAL_OUT>::value)
      {
        const size_t len = (size_t) m*n*sizeof(REAL_IN);
        c->mem_gpu2gpu((void*)out, (void*)in, len);
      }
      else
        kernel_copy<<<griddim, blockdim>>>(m, n, in, out);
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
        cpuvec<REAL_OUT> cp(CPLEN);
        REAL_OUT *cp_d = cp.data_ptr();
        
        size_t top = (size_t) m*n;
        for (size_t i=0; i<top; i+=CPLEN)
        {
          const size_t start = top - i;
          const size_t copylen = std::min(CPLEN, start);
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
        cpuvec<REAL_OUT> cp(CPLEN);
        REAL_OUT *cp_d = cp.data_ptr();
        
        size_t top = (size_t) m*n;
        for (size_t i=0; i<top; i+=CPLEN)
        {
          const size_t start = top - i;
          const size_t copylen = std::min(CPLEN, start);
          arraytools::copy(copylen, in + start, cp_d);
          c->mem_cpu2gpu((void*)(out + start), (void*)cp_d, copylen);
        }
      }
    }
  }
  
  
  
  template <typename REAL_IN, typename REAL_OUT>
  void gpu2cpu(const gpuvec<REAL_IN> &gpu, cpuvec<REAL_OUT> &cpu)
  {
    cpu.resize(gpu.size());
    copy_gpu2cpu(gpu.size(), (len_t)1, gpu.get_card(), gpu.data_ptr(), cpu.data_ptr());
  }
  
  template <typename REAL_IN, typename REAL_OUT>
  void gpu2cpu(const gpumat<REAL_IN> &gpu, cpumat<REAL_OUT> &cpu)
  {
    cpu.resize(gpu.nrows(), gpu.ncols());
    copy_gpu2cpu(gpu.nrows(), gpu.ncols(), gpu.get_card(), gpu.data_ptr(), cpu.data_ptr());
  }
  
  
  
  template <typename REAL_IN, typename REAL_OUT>
  void cpu2gpu(const cpuvec<REAL_IN> &cpu, gpuvec<REAL_OUT> &gpu)
  {
    gpu.resize(cpu.size());
    
    copy_cpu2gpu(cpu.size(), (len_t)1, gpu.get_card(), cpu.data_ptr(), gpu.data_ptr());
  }
  
  template <typename REAL_IN, typename REAL_OUT>
  void cpu2gpu(const cpumat<REAL_IN> &cpu, gpumat<REAL_OUT> &gpu)
  {
    gpu.resize(cpu.nrows(), cpu.ncols());
    copy_cpu2gpu(cpu.nrows(), cpu.ncols(), gpu.get_card(), cpu.data_ptr(), gpu.data_ptr());
  }
  
  
  
  template <typename REAL_IN, typename REAL_OUT>
  void gpu2gpu(const gpuvec<REAL_IN> &gpu_in, gpuvec<REAL_OUT> &gpu_out)
  {
    auto c = gpu_in.get_card();
    if (c->get_id() != gpu_out.get_card()->get_id())
      throw std::logic_error("input/output data must be on the same gpu");
    
    gpu_out.resize(gpu_in.size());
    copy_gpu2gpu(gpu_in.size(), (len_t)1, c, gpu_in.get_griddim(), gpu_in.get_blockdim(), gpu_in.data_ptr(), gpu_out.data_ptr());
  }
  
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
