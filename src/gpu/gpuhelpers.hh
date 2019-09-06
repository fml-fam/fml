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
  
  
  
  inline __device__ void kernel_copy(len_t m, len_t n, __half *in, float *out)
  {
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    int j = blockDim.y*blockIdx.y + threadIdx.y;
    
    if (i < m && j < n)
      out[i + m*j] = __half2float(in[i + m*j]);
  }
  
  inline __device__ void kernel_copy(len_t m, len_t n, float *in, __half *out)
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
  void gpu2cpu(gpuvec<REAL> &gpu, cpuvec<REAL> &cpu)
  {
    if (gpu.size() != cpu.size())
      throw std::runtime_error("non-conformable arguments");
    
    size_t len = gpu.size() * sizeof(REAL);
    gpu.get_card()->mem_gpu2cpu(cpu.data_ptr(), gpu.data_ptr(), len);
  }
  
  template <typename REAL>
  cpuvec<REAL> gpu2cpu(gpuvec<REAL> &gpu)
  {
    cpuvec<REAL> cpu(gpu.size());
    gpu2cpu(gpu, cpu);
    return cpu;
  }
  
  
  
  template <typename REAL>
  void gpu2cpu(gpumat<REAL> &gpu, cpumat<REAL> &cpu)
  {
    if (gpu.nrows() != cpu.nrows() || gpu.ncols() != cpu.ncols())
      throw std::runtime_error("non-conformable arguments");
    
    size_t len = (size_t) gpu.nrows() * gpu.ncols() * sizeof(REAL);
    gpu.get_card()->mem_gpu2cpu(cpu.data_ptr(), gpu.data_ptr(), len);
  }
  
  template <typename REAL>
  cpumat<REAL> gpu2cpu(gpumat<REAL> &gpu)
  {
    cpumat<REAL> cpu(gpu.nrows(), gpu.ncols());
    gpu2cpu(gpu, cpu);
    return cpu;
  }
  
  
  
  // cpu2gpu
  template <typename REAL>
  void cpu2gpu(cpuvec<REAL> &cpu, gpuvec<REAL> &gpu)
  {
    if (gpu.size() != cpu.size())
      throw std::runtime_error("non-conformable arguments");
    
    size_t len = cpu.size() * sizeof(REAL);
    gpu.get_card()->mem_cpu2gpu(gpu.data_ptr(), cpu.data_ptr(), len);
  }
  
  template <typename REAL>
  gpuvec<REAL> cpu2gpu(std::shared_ptr<card> c, cpuvec<REAL> &cpu)
  {
    gpuvec<REAL> gpu(c, cpu.size());
    cpu2gpu(cpu, gpu);
    return gpu;
  }
  
  
  
  template <typename REAL>
  void cpu2gpu(cpumat<REAL> &cpu, gpumat<REAL> &gpu)
  {
    if (gpu.nrows() != cpu.nrows() || gpu.ncols() != cpu.ncols())
      throw std::runtime_error("non-conformable arguments");
    
    size_t len = (size_t) gpu.nrows() * gpu.ncols() * sizeof(REAL);
    gpu.get_card()->mem_cpu2gpu(gpu.data_ptr(), cpu.data_ptr(), len);
  }
  
  template <typename REAL>
  gpumat<REAL> cpu2gpu(cpumat<REAL> &cpu, std::shared_ptr<card> c)
  {
    gpumat<REAL> gpu(c, cpu.nrows(), cpu.ncols());
    cpu2gpu(cpu, gpu);
    return gpu;
  }
}


#endif
