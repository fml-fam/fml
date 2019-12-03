// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_GPU_INTERNALS_CUDA_GPUPRIMS_H
#define FML_GPU_INTERNALS_CUDA_GPUPRIMS_H
#pragma once


#include <cublas.h>
#include <cusolverDn.h>


namespace gpuprims
{
  inline cudaError_t gpu_device_reset()
  {
    return cudaDeviceReset();
  }
  
  inline std::string gpu_error_string(cudaError_t code)
  {
    return cudaGetErrorString(code);
  }
  
  
  
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


#endif
