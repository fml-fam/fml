// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_GPU_ARCH_CUDA_GPUPRIMS_H
#define FML_GPU_ARCH_CUDA_GPUPRIMS_H
#pragma once


#include <cublas.h>
#include <cusolverDn.h>


namespace fml
{
  namespace gpuprims
  {
    // device management
    inline cudaError_t gpu_set_device(int device)
    {
      return cudaSetDevice(device);
    }
    
    inline cudaError_t gpu_synch()
    {
      return cudaDeviceSynchronize();
    }
    
    inline cudaError_t gpu_device_reset()
    {
      return cudaDeviceReset();
    }
    
    
    
    // memory management
    inline cudaError_t gpu_malloc(void **x, size_t size)
    {
      return cudaMalloc(x, size);
    }
    
    inline cudaError_t gpu_memset(void *x, int value, size_t count)
    {
      return cudaMemset(x, value, count);
    }
    
    inline cudaError_t gpu_free(void *x)
    {
      return cudaFree(x);
    }
    
    inline cudaError_t gpu_memcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind)
    {
      return cudaMemcpy(dst, src, count, kind);
    }
    
    
    
    // error handling
    inline std::string gpu_error_string(cudaError_t code)
    {
      return cudaGetErrorString(code);
    }
    
    inline cudaError_t gpu_last_error()
    {
      return cudaGetLastError();
    }
    
    
    
    // cublas and cusolver
    inline cublasStatus_t gpu_blas_init(cublasHandle_t *handle)
    {
      return cublasCreate(handle);
    }
    
    inline cublasStatus_t gpu_blas_free(cublasHandle_t handle)
    {
      return cublasDestroy(handle);
    }
    
    inline cusolverStatus_t gpu_lapack_init(cusolverDnHandle_t *handle)
    {
      return cusolverDnCreate(handle);
    }
    
    inline cusolverStatus_t gpu_lapack_free(cusolverDnHandle_t handle)
    {
      return cusolverDnDestroy(handle);
    }
  }
}


#endif
