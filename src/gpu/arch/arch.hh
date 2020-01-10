// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_GPU_ARCH_ARCH_H
#define FML_GPU_ARCH_ARCH_H
#pragma once


#include <cstdarg>
#include "../../_internals/types.hh"

#if (!defined(FML_USE_CUDA) && !defined(FML_USE_HIP))
  #define FML_USE_CUDA
#endif

#if (defined(FML_USE_CUDA))
  #include "cuda/gpulapack.hh"
  #include "cuda/gpuprims.hh"
  #include "cuda/gpurand.hh"
  #include "cuda/nvml.hh"
  #include "cuda/types.hh"
#elif defined(FML_USE_HIP)
  #error "HIP is currently unsupported"
  // #include "hip/gpulapack.hh"
  // #include "hip/gpuprims.hh"
  // #include "hip/gpurand.hh"
  // #include "hip/rocm_smi.hh"
  // #include "hip/types.hh"
#else
  #error "Unsupported kernel launcher"
#endif


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
