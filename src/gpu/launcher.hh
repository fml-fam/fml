#ifndef FML_GPU_LAUNCHER_H
#define FML_GPU_LAUNCHER_H


// TODO
#define FML_USE_CUDA


#include <cstdarg>

#if defined(FML_USE_CUDA)
  #define FML_LAUNCH_KERNEL(FML_KERNEL, FML_GRIDSIZE, FML_BLOCKSIZE, ...) \
  FML_KERNEL<<<FML_GRIDSIZE, FML_BLOCKSIZE>>>(__VA_ARGS__)
#elif defined(FML_USE_HIP)
  #define FML_LAUNCH_KERNEL(FML_KERNEL, FML_GRIDSIZE, FML_BLOCKSIZE, ...) \
  hipLaunchKernelGGL(FML_KERNEL, FML_GRIDSIZE, FML_BLOCKSIZE, 0, 0, __VA_ARGS__)
#else
  #error "Unsupported kernel launcher"
#endif


#endif
