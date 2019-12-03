// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_GPU_INTERNALS_CUDA_TYPES_H
#define FML_GPU_INTERNALS_CUDA_TYPES_H
#pragma once


#include <cublas.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

typedef cudaError_t GPUError_t;
#define GPU_SUCCESS cudaSuccess

#define GPUBLAS_STATUS_SUCCESS CUBLAS_STATUS_SUCCESS
typedef cublasStatus_t BLASStatus_t;
typedef cublasHandle_t BLASHandle_t;

#define GPULAPACK_STATUS_SUCCESS CUSOLVER_STATUS_SUCCESS
typedef cusolverStatus_t LAPACKStatus_t;
typedef cusolverDnHandle_t LAPACKHandle_t;


#endif
