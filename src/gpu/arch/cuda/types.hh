// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_GPU_ARCH_CUDA_TYPES_H
#define FML_GPU_ARCH_CUDA_TYPES_H
#pragma once


#include <cublas.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

typedef cudaError_t gpu_error_t;
#define GPU_SUCCESS cudaSuccess

#define GPU_MEMCPY_HOST_TO_DEVICE cudaMemcpyHostToDevice
#define GPU_MEMCPY_DEVICE_TO_HOST cudaMemcpyDeviceToHost
#define GPU_MEMCPY_DEVICE_TO_DEVICE cudaMemcpyDeviceToDevice

#define GPUBLAS_STATUS_SUCCESS CUBLAS_STATUS_SUCCESS
typedef cublasStatus_t blas_status_t;
typedef cublasHandle_t blas_handle_t;

#define GPULAPACK_STATUS_SUCCESS CUSOLVER_STATUS_SUCCESS
typedef cusolverStatus_t lapack_status_t;
typedef cusolverDnHandle_t lapack_handle_t;


#endif
