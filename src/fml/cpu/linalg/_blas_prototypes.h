// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_CPU_LINALG__BLAS_PROTOTYPES_H
#define FML_CPU_LINALG__BLAS_PROTOTYPES_H
#pragma once


#ifndef restrict
#define restrict __restrict__
#endif

#ifdef __cplusplus
extern "C" {
#endif


extern void sgemm_(const char *transa, const char *transb, const int *m,
  const int *n, const int *k, const float *restrict alpha,
  const float *restrict a, const int *lda, const float *restrict b,
  const int *ldb, const float *beta, float *restrict c, const int *ldc);

extern void dgemm_(const char *transa, const char *transb, const int *m,
  const int *n, const int *k, const double *restrict alpha,
  const double *restrict a, const int *lda, const double *restrict b,
  const int *ldb, const double *beta, double *restrict c, const int *ldc);



extern void ssyrk_(const char *uplo, const char *trans, const int *n,
  const int *k, const float *restrict alpha, const float *restrict a,
  const int *lda, const float *restrict beta, float *restrict c,
  const int *ldc);

extern void dsyrk_(const char *uplo, const char *trans, const int *n,
  const int *k, const double *restrict alpha, const double *restrict a,
  const int *lda, const double *restrict beta, double *restrict c,
  const int *ldc);


#ifdef __cplusplus
}
#endif


#endif
