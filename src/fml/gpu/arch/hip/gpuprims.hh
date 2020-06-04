// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_GPU_ARCH_HIP_GPUPRIMS_H
#define FML_GPU_ARCH_HIP_GPUPRIMS_H
#pragma once


#include <rocblas.h>
#include <rocsolver.h>


namespace fml
{
  namespace gpuprims
  {
    // device management
    inline hipError_t gpu_set_device(int device)
    {
      return hipSetDevice(device);
    }
    
    inline hipError_t gpu_synch()
    {
      return hipDeviceSynchronize();
    }
    
    inline hipError_t gpu_device_reset()
    {
      return hipDeviceReset();
    }
    
    
    
    // memory management
    inline hipError_t gpu_malloc(void **x, size_t size)
    {
      return hipMalloc(x, size);
    }
    
    inline hipError_t gpu_memset(void *x, int value, size_t count)
    {
      return hipMemset(x, value, count);
    }
    
    inline hipError_t gpu_free(void *x)
    {
      return hipFree(x);
    }
    
    inline hipError_t gpu_memcpy(void *dst, const void *src, size_t count, hipMemcpyKind kind)
    {
      return hipMemcpy(dst, src, count, kind);
    }
    
    
    
    // error handling
    inline std::string gpu_error_string(hipError_t code)
    {
      return hipGetErrorString(code);
    }
    
    inline hipError_t gpu_last_error()
    {
      return hipGetLastError();
    }
    
    
    
    // rocblas and rocsolver
    inline rocblas_status gpu_blas_init(rocblas_handle *handle)
    {
      return rocblas_create_handle(handle);
    }
    
    inline rocblas_status gpu_blas_free(rocblas_handle handle)
    {
      return rocblas_destroy_handle(handle);
    }
    
    inline rocsolver_status gpu_lapack_init(rocsolver_handle *handle)
    {
      return rocsolver_create_handle(handle);
    }
    
    inline rocsolver_status gpu_lapack_free(rocsolver_handle handle)
    {
      return rocsolver_destroy_handle(handle);
    }
  }
}


#endif
