// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_CPU_LINALG_BLAS_H
#define FML_CPU_LINALG_BLAS_H
#pragma once


#include "_blas_prototypes.h"


namespace fml
{
  namespace blas
  {
    inline void gemm(const char transa, const char transb, const int m,
      const int n, const int k, const float alpha,
      const float *restrict a, const int lda, const float *restrict b,
      const int ldb, const float beta, float *restrict c, const int ldc)
    {
      sgemm_(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c,
        &ldc);
    }
    
    inline void gemm(const char transa, const char transb, const int m,
      const int n, const int k, const double alpha,
      const double *restrict a, const int lda, const double *restrict b,
      const int ldb, const double beta, double *restrict c, const int ldc)
    {
      dgemm_(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c,
        &ldc);
    }
    
    
    
    inline void syrk(const char uplo, const char trans, const int n, const int k, 
      const float alpha, const float *restrict a, const int lda,
      const float beta, float *restrict c, const int ldc)
    {
      ssyrk_(&uplo, &trans, &n, &k, &alpha, a, &lda, &beta, c, &ldc);
    }
    
    inline void syrk(const char uplo, const char trans, const int n, const int k, 
      const double alpha, const double *restrict a, const int lda,
      const double beta, double *restrict c, const int ldc)
    {
      dsyrk_(&uplo, &trans, &n, &k, &alpha, a, &lda, &beta, c, &ldc);
    }
  }
}


#endif
