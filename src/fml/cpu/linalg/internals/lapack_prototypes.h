// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_CPU_INTERNALS_INTERNALS_LAPACK_PROTOTYPES_H
#define FML_CPU_INTERNALS_INTERNALS_LAPACK_PROTOTYPES_H
#pragma once


#ifndef restrict
#define restrict __restrict__
#endif

#ifdef __cplusplus
extern "C" {
#endif


extern void sgetrf_(const int *const restrict m, const int *const restrict n,
  float *const restrict a, const int *const lda,
  int *const restrict ipiv, int *const restrict info);

extern void dgetrf_(const int *const restrict m, const int *const restrict n,
  double *const restrict a, const int *const lda,
  int *const restrict ipiv, int *const restrict info);



extern void sgesdd_(const char *const restrict jobz, const int *const restrict m,
  const int *const restrict n, float *const restrict a,
  const int *const restrict lda, float *const restrict s,
  float *const restrict u, const int *const restrict ldu,
  float *const restrict vt, const int *const restrict ldvt,
  float *const restrict work, const int *const restrict lwork,
  int *const restrict iwork, int *const restrict info);

extern void dgesdd_(const char *const restrict jobz, const int *const restrict m,
  const int *const restrict n, double *const restrict a,
  const int *const restrict lda, double *const restrict s,
  double *const restrict u, const int *const restrict ldu,
  double *const restrict vt, const int *const restrict ldvt,
  double *const restrict work, const int *const restrict lwork,
  int *const restrict iwork, int *const restrict info);



extern void ssyevr_(const char *restrict jobz, const char *restrict range,
  const char *restrict uplo, const int *restrict n, float *restrict a,
  const int *restrict lda, const float *restrict vl, const float *restrict vu,
  const int *restrict il, const int *restrict iu, const float *restrict abstol,
  const int *restrict m, float *w, float *z, const int *restrict ldz,
  int *isuppz, float *work, const int *lwork, int *iwork, const int *liwork,
  int *info);

extern void dsyevr_(const char *restrict jobz, const char *restrict range,
  const char *restrict uplo, const int *restrict n, double *restrict a,
  const int *restrict lda, const double *restrict vl, const double *restrict vu,
  const int *restrict il, const int *restrict iu, const double *restrict abstol,
  const int *restrict m, double *w, double *z, const int *restrict ldz,
  int *isuppz, double *work, const int *lwork, int *iwork, const int *liwork,
  int *info);



extern void sgeev_(const char *restrict jobvl, const char *restrict jobvr,
  const int *n, float *restrict a, const int *lda, float *restrict wr,
  float *restrict wi, float *restrict vl, const int *ldvl,
  float *restrict vr, const int *ldvr, float *restrict work,
  const int *restrict lwork, int *restrict info);

extern void dgeev_(const char *restrict jobvl, const char *restrict jobvr,
  const int *n, double *restrict a, const int *lda, double *restrict wr,
  double *restrict wi, double *restrict vl, const int *ldvl,
  double *restrict vr, const int *ldvr, double *restrict work,
  const int *restrict lwork, int *restrict info);



extern void sgetri_(const int *n, float *restrict a, const int *lda, 
  int *restrict ipiv, float *restrict work, const int *restrict lwork,
  int *info);

extern void dgetri_(const int *n, double *restrict a, const int *lda, 
  int *restrict ipiv, double *restrict work, const int *restrict lwork,
  int *info);



extern void sgesv_(const int *n, const int *restrict nrhs, float *restrict a,
  const int *lda, int *restrict ipiv, float *restrict b, const int *ldb,
  int *restrict info);

extern void dgesv_(const int *n, const int *restrict nrhs, double *restrict a,
  const int *lda, int *restrict ipiv, double *restrict b, const int *ldb,
  int *restrict info);



extern void slacpy_(const char *uplo, const int *m, const int *n,
  const float *restrict a, const int *lda, float *restrict b, const int *ldb);

extern void dlacpy_(const char *uplo, const int *m, const int *n,
  const double *restrict a, const int *lda, double *restrict b, const int *ldb);



extern void sgeqp3_(const int *const m, const int *const n,
  float *const restrict A, const int *const lda, int *const restrict jpvt,
  float *const restrict tau, float *const restrict work,
  const int *const restrict lwork, int *const restrict info);

extern void dgeqp3_(const int *const m, const int *const n,
  double *const restrict A, const int *const lda, int *const restrict jpvt,
  double *const restrict tau, double *const restrict work,
  const int *const restrict lwork, int *const restrict info);



extern void sgeqrf_(const int *const m, const int *const n,
  float *const restrict A, const int *const lda, float *const restrict tau,
  float *const restrict work, const int *const restrict lwork,
  int *const restrict info);

extern void dgeqrf_(const int *const m, const int *const n,
  double *const restrict A, const int *const lda, double *const restrict tau,
  double *const restrict work, const int *const restrict lwork,
  int *const restrict info);



extern void sormqr_(const char *side, const char *trans, const int *m,
  const int *n, const int *k, const float *const restrict A, const int *lda,
  const float *const restrict tau, float *const restrict c, const int *ldc,
  float *const restrict work, const int *lwork, int *const restrict info);

extern void dormqr_(const char *side, const char *trans, const int *m,
  const int *n, const int *k, const double *const restrict A, const int *lda,
  const double *const restrict tau, double *const restrict c, const int *ldc,
  double *const restrict work, const int *lwork, int *const restrict info);



extern void sorgqr_(const int *m, const int *n, const int *k, float *A,
  const int *lda, const float *tau, float *work, const int *ldwork, int *info);

extern void dorgqr_(const int *m, const int *n, const int *k, double *A,
  const int *lda, const double *tau, double *work, const int *ldwork, int *info);



extern void sorglq_(const int *m, const int *n, const int *k, float *A,
  const int *lda, const float *tau, float *work, const int *ldwork, int *info);

extern void dorglq_(const int *m, const int *n, const int *k, double *A,
  const int *lda, const double *tau, double *work, const int *ldwork, int *info);



extern void sgelqf_(const int *const m, const int *const n,
  float *const restrict A, const int *const lda, float *const restrict tau,
  float *const restrict work, const int *const restrict lwork,
  int *const restrict info);

extern void dgelqf_(const int *const m, const int *const n,
  double *const restrict A, const int *const lda, double *const restrict tau,
  double *const restrict work, const int *const restrict lwork,
  int *const restrict info);



extern void sormlq_(const char *side, const char *trans, const int *m,
  const int *n, const int *k, const float *const restrict A, const int *lda,
  const float *const restrict tau, float *const restrict c, const int *ldc,
  float *const restrict work, const int *lwork, int *const restrict info);

extern void dormlq_(const char *side, const char *trans, const int *m,
  const int *n, const int *k, const double *const restrict A, const int *lda,
  const double *const restrict tau, double *const restrict c, const int *ldc,
  double *const restrict work, const int *lwork, int *const restrict info);



extern void spotrf_(const char *uplo, const int *n, float *const restrict A,
  const int *lda, int *info);

extern void dpotrf_(const char *uplo, const int *n, double *const restrict A,
  const int *lda, int *info);



extern void slassq_(const int *n, const float *restrict x, const int *incx,
  float *scale, float *sumsq);

extern void dlassq_(const int *n, const double *restrict x, const int *incx,
  double *scale, double *sumsq);



extern void sgecon_(const char *norm, const int *n,
  const float *const restrict A, const int *lda, const float *anorm,
  float *rcond, float *work, float *work2, int *info);

extern void dgecon_(const char *norm, const int *n,
  const double *const restrict A, const int *lda, const double *anorm,
  double *rcond, double *work, double *work2, int *info);



extern void strcon_(const char *norm, const char *uplo, const char *diag,
  const int *n, const float *const restrict A, const int *lda, float *rcond,
  float *work, float *work2, int *info);

extern void dtrcon_(const char *norm, const char *uplo, const char *diag,
  const int *n, const double *const restrict A, const int *lda, double *rcond,
  double *work, double *work2, int *info);



extern void strtri_(const char *uplo, const char *diag, const int *n,
  float *A, const int *lda, int *info);

extern void dtrtri_(const char *uplo, const char *diag, const int *n,
  double *A, const int *lda, int *info);


#ifdef __cplusplus
}
#endif


#endif
