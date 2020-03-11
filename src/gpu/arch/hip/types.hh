// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_GPU_ARCH_HIP_TYPES_H
#define FML_GPU_ARCH_HIP_TYPES_H
#pragma once


#include <hip_runtime_api.h>
#include <rocblas.h>
#include <rocsolver.h>


// core
typedef hipError_t gpu_error_t;
#define GPU_SUCCESS hipSuccess

#define GPU_MEMCPY_HOST_TO_DEVICE hipMemcpyHostToDevice
#define GPU_MEMCPY_DEVICE_TO_HOST hipMemcpyDeviceToHost
#define GPU_MEMCPY_DEVICE_TO_DEVICE hipMemcpyDeviceToDevice

// blas
#define GPUBLAS_STATUS_SUCCESS rocblas_status_success
typedef rocblas_status gpublas_status_t;
typedef rocblas_handle gpublas_handle_t;
typedef rocblas_operation gpublas_operation_t;
typedef rocblas_fill gpublas_fillmode_t;
#define GPUBLAS_OP_T rocblas_operation_transpose
#define GPUBLAS_OP_N rocblas_operation_none
#define GPUBLAS_FILL_L rocblas_fill_lower
#define GPUBLAS_FILL_U rocblas_fill_upper
#define GPUBLAS_SIDE_LEFT rocblas_side_left
#define GPUBLAS_SIDE_RIGHT rocblas_side_right

// lapack/"solver"
#define GPULAPACK_STATUS_SUCCESS CUSOLVER_STATUS_SUCCESS
typedef rocsolver_status gpulapack_status_t;
typedef rocsolver_handle gpulapack_handle_t;


#endif
