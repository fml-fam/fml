// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_GPU_ARCH_CUDA_TYPES_H
#define FML_GPU_ARCH_CUDA_TYPES_H
#pragma once


#include <cublas.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>


// core
typedef cudaError_t gpu_error_t;
#define GPU_SUCCESS cudaSuccess

#define GPU_MEMCPY_HOST_TO_DEVICE cudaMemcpyHostToDevice
#define GPU_MEMCPY_DEVICE_TO_HOST cudaMemcpyDeviceToHost
#define GPU_MEMCPY_DEVICE_TO_DEVICE cudaMemcpyDeviceToDevice

// blas
#define GPUBLAS_STATUS_SUCCESS CUBLAS_STATUS_SUCCESS
typedef cublasStatus_t gpublas_status_t;
typedef cublasHandle_t gpublas_handle_t;

typedef cublasOperation_t gpublas_operation_t;
#define GPUBLAS_OP_T CUBLAS_OP_T
#define GPUBLAS_OP_N CUBLAS_OP_N

typedef cublasFillMode_t gpublas_fillmode_t;
#define GPUBLAS_FILL_L CUBLAS_FILL_MODE_LOWER
#define GPUBLAS_FILL_U CUBLAS_FILL_MODE_UPPER

typedef cublasSideMode_t gpublas_sidemode_t;
#define GPUBLAS_SIDE_LEFT CUBLAS_SIDE_LEFT
#define GPUBLAS_SIDE_RIGHT CUBLAS_SIDE_RIGHT

typedef cublasDiagType_t gpublas_diagtype_t;
#define GPUBLAS_DIAG_NON_UNIT CUBLAS_DIAG_NON_UNIT
#define GPUBLAS_DIAG_UNIT CUBLAS_DIAG_UNIT

typedef cublasMath_t gpublas_mathmode_t;
#define GPUBLAS_MATH_DEFAULT CUBLAS_DEFAULT_MATH
#define GPUBLAS_MATH_ACCELERATE CUBLAS_TF32_TENSOR_OP_MATH
#define GPUBLAS_MATH_PEDANTIC CUBLAS_PEDANTIC_MATH

// lapack/"solver"
#define GPULAPACK_STATUS_SUCCESS CUSOLVER_STATUS_SUCCESS
typedef cusolverStatus_t gpulapack_status_t;
typedef cusolverDnHandle_t gpulapack_handle_t;


#endif
