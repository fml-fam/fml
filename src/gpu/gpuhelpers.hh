#ifndef FML_GPU_GPUHELPERS_H
#define FML_GPU_GPUHELPERS_H


#include <stdexcept>

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
  void copy(const gpumat<REAL_IN> &in, gpumat<REAL_OUT> &out)
  {
    len_t m = in.nrows();
    len_t n = in.ncols();
    auto c = in.get_card();
    out.resize(c, m, n);
    
    REAL_IN *in_d = in.data_ptr();
    REAL_OUT *out_d = out.data_ptr();
    
    if (std::is_same<REAL_IN, REAL_OUT>::value)
    {
      const size_t len = (size_t) m*n*sizeof(REAL_IN);
      c->mem_gpu2gpu(out_d, in_d, len);
    }
    else
    {
      dim3 dim_block(16, 16);
      dim3 dim_grid((m + 16 - 1) / 16, (n + 16 - 1) / 16);
      kernel_copy<<<dim_grid, dim_block>>>(m, n, in_d, out_d);
    }
  }
  
  
  
  // gpu2cpu
  template <typename REAL>
  void gpu2cpu(const gpuvec<REAL> &gpu, cpuvec<REAL> &cpu)
  {
    cpu.resize(gpu.size());
    
    size_t len = gpu.size() * sizeof(REAL);
    gpu.get_card()->mem_gpu2cpu(cpu.data_ptr(), gpu.data_ptr(), len);
  }
  
  template <typename REAL>
  void gpu2cpu(const gpumat<REAL> &gpu, cpumat<REAL> &cpu)
  {
    cpu.resize(gpu.nrows(), gpu.ncols());
    
    size_t len = (size_t) gpu.nrows() * gpu.ncols() * sizeof(REAL);
    gpu.get_card()->mem_gpu2cpu(cpu.data_ptr(), gpu.data_ptr(), len);
  }
  
  
  
  // cpu2gpu
  template <typename REAL>
  void cpu2gpu(const cpuvec<REAL> &cpu, gpuvec<REAL> &gpu)
  {
    gpu.resize(cpu.size());
    
    size_t len = cpu.size() * sizeof(REAL);
    gpu.get_card()->mem_cpu2gpu(gpu.data_ptr(), cpu.data_ptr(), len);
  }
  
  template <typename REAL>
  void cpu2gpu(const cpumat<REAL> &cpu, gpumat<REAL> &gpu)
  {
    gpu.resize(cpu.nrows(), cpu.ncols());
    
    size_t len = (size_t) gpu.nrows() * gpu.ncols() * sizeof(REAL);
    gpu.get_card()->mem_cpu2gpu(gpu.data_ptr(), cpu.data_ptr(), len);
  }
}


#endif
