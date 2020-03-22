// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_GPU_INTERNALS_GPU_UTILS_H
#define FML_GPU_INTERNALS_GPU_UTILS_H
#pragma once


#include "../arch/arch.hh"
#include "launcher.hh"


namespace fml
{
  namespace gpu_utils
  {
    namespace internals
    {
      template <typename REAL>
      __global__ void kernel_lacpy(const gpublas_fillmode_t uplo, const int m, const int n, const REAL *A, const int lda,
        REAL *B, const int ldb)
      {
        int i = blockDim.x*blockIdx.x + threadIdx.x;
        int j = blockDim.y*blockIdx.y + threadIdx.y;
        
        if ((i < m && j < n) && (uplo == GPUBLAS_FILL_U && i <= j) || (uplo == GPUBLAS_FILL_L && i >= j))
            B[i + ldb*j] = A[i + lda*j];
      }
    }
    
    template <typename REAL>
    void lacpy(const gpublas_fillmode_t uplo, const int m, const int n,
      const REAL *A, const int lda, REAL *B, const int ldb)
    {
      auto dim_block = fml::kernel_launcher::dim_block2();
      auto dim_grid = fml::kernel_launcher::dim_grid(m, n);
      internals::kernel_lacpy<<<dim_grid, dim_block>>>(uplo, m, n, A, lda, B, ldb);
    }
    
    
    
    namespace internals
    {
      template <typename REAL>
      __global__ void kernel_tri2zero(const char uplo, const bool diag, const len_t m, const len_t n, REAL *A, const len_t lda)
      {
        int i = blockDim.x*blockIdx.x + threadIdx.x;
        int j = blockDim.y*blockIdx.y + threadIdx.y;
        
        if ((i < m && j < n) && ((diag && i == j) || (uplo == 'U' && i < j) || (uplo == 'L' && i > j)))
          A[i + lda*j] = (REAL) 0.0;
      }
    }
    
    template <typename REAL>
    void tri2zero(const char uplo, const bool diag, const int m, const int n,
      REAL *A, const int lda)
    {
      auto dim_block = fml::kernel_launcher::dim_block2();
      auto dim_grid = fml::kernel_launcher::dim_grid(m, n);
      internals::kernel_tri2zero<<<dim_grid, dim_block>>>(uplo, diag, m, n, A, lda);
    }
  }
}


#endif
