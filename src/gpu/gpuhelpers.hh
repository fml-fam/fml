#ifndef FML_GPU_GPUHELPERS_H
#define FML_GPU_GPUHELPERS_H


#include "../cpu/cpumat.hh"
#include "gpumat.hh"


namespace gpuhelpers
{
  template <typename REAL>
  void gpu2cpu_noalloc(gpumat<REAL> &gpu, cpumat<REAL> &cpu)
  {
    size_t len = gpu.nrows() * gpu.ncols() * sizeof(REAL);
    gpu.get_card()->mem_gpu2cpu(cpu.data_ptr(), gpu.data_ptr(), len);
  }
  
  template <typename REAL>
  cpumat<REAL> gpu2cpu(gpumat<REAL> &gpu)
  {
    cpumat<REAL> cpu(gpu.nrows(), gpu.ncols());
    gpu2cpu_noalloc(gpu, cpu);
    return cpu;
  }
  
  
  
  template <typename REAL>
  void cpu2gpu_noalloc(cpumat<REAL> &cpu, gpumat<REAL> &gpu)
  {
    size_t len = cpu.nrows() * cpu.ncols() * sizeof(REAL);
    gpu.get_card()->mem_cpu2gpu(gpu.data_ptr(), cpu.data_ptr(), len);
  }
  
  template <typename REAL>
  gpumat<REAL> cpu2gpu(cpumat<REAL> &cpu, std::shared_ptr<card> c)
  {
    gpumat<REAL> gpu(c, cpu.nrows(), cpu.ncols());
    cpu2gpu_noalloc(cpu, gpu);
    return gpu;
  }
}


#endif
