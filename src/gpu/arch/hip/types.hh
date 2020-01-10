// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_GPU_ARCH_HIP_TYPES_H
#define FML_GPU_ARCH_HIP_TYPES_H
#pragma once


#include <cublas.h>       // FIXME
#include <cuda_runtime.h> // FIXME
#include <rocsolver.h>

typedef hipError_t GPUError_t;
#define GPU_SUCCESS hipSuccess

#define GPU_MEMCPY_HOST_TO_DEVICE hipMemcpyHostToDevice
#define GPU_MEMCPY_DEVICE_TO_HOST hipMemcpyDeviceToHost
#define GPU_MEMCPY_DEVICE_TO_DEVICE hipMemcpyDeviceToDevice

#define GPUBLAS_STATUS_SUCCESS rocblas_status_success
typedef rocblas_status BLASStatus_t;
typedef rocblas_handle BLASHandle_t;

#define GPULAPACK_STATUS_SUCCESS CUSOLVER_STATUS_SUCCESS
typedef rocsolver_status LAPACKStatus_t;
typedef rocsolver_handle LAPACKHandle_t;


#endif
