// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_CPU_LINALG_LAPACK_H
#define FML_CPU_LINALG_LAPACK_H
#pragma once


#include "_lapack_prototypes.h"


namespace fml
{
  namespace lapack
  {
    inline void getrf(const int m, const int n, float *a, const int lda,
      int *ipiv, int *info)
    {
      sgetrf_(&m, &n, a, &lda, ipiv, info);
    }
    
    inline void getrf(const int m, const int n, double *a, const int lda,
      int *ipiv, int *info)
    {
      dgetrf_(&m, &n, a, &lda, ipiv, info);
    }
    
    
    
    inline void gesdd(const char jobz, const int m, const int n, float *a,
      const int lda, float *s, float *u, const int ldu, float *vt,
      const int ldvt, float *work, const int lwork, int *iwork, int *info)
    {
      sgesdd_(&jobz, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, iwork,
        info);
    }
    
    inline void gesdd(const char jobz, const int m, const int n, double *a,
      const int lda, double *s, double *u, const int ldu, double *vt,
      const int ldvt, double *work, const int lwork, int *iwork, int *info)
    {
      dgesdd_(&jobz, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, iwork,
        info);
    }
    
    
    
    inline void syevr(const char jobz, const char range, const char uplo,
      const int n, float *a, const int lda, const float vl, const float vu,
      const int il, const int iu, const float abstol, int *m, float *w,
      float *z, const int ldz, int *isuppz, float *work, const int lwork,
      int *iwork, const int liwork, int *info)
    {
      ssyevr_(&jobz, &range, &uplo, &n, a, &lda, &vl, &vu, &il, &iu, &abstol, m,
        w, z, &ldz, isuppz, work, &lwork, iwork, &liwork, info);
    }
    
    inline void syevr(const char jobz, const char range, const char uplo,
      const int n, double *a, const int lda, const double vl, const double vu,
      const int il, const int iu, const double abstol, int *m, double *w,
      double *z, const int ldz, int *isuppz, double *work, const int lwork,
      int *iwork, const int liwork, int *info)
    {
      dsyevr_(&jobz, &range, &uplo, &n, a, &lda, &vl, &vu, &il, &iu, &abstol, m,
        w, z, &ldz, isuppz, work, &lwork, iwork, &liwork, info);
    }
    
    
    
    inline void geev(const char jobvl, const char jobvr, const int n, float *a,
      const int lda, float *wr, float *wi, float *vl, const int ldvl, float *vr,
      const int ldvr, float *work, const int lwork, int *info)
      {
        sgeev_(&jobvl, &jobvr, &n, a, &lda, wr, wi, vl, &ldvl, vr, &ldvr, work,
          &lwork, info);
      }
    
    inline void geev(const char jobvl, const char jobvr, const int n, double *a,
      const int lda, double *wr, double *wi, double *vl, const int ldvl, double *vr,
      const int ldvr, double *work, const int lwork, int *info)
      {
        dgeev_(&jobvl, &jobvr, &n, a, &lda, wr, wi, vl, &ldvl, vr, &ldvr, work,
          &lwork, info);
      }
    
    
    
    inline void getri(const int n, float *a, const int lda, int *ipiv,
      float *work, const int lwork, int *info)
    {
      sgetri_(&n, a, &lda, ipiv, work, &lwork, info);
    }
    
    inline void getri(const int n, double *a, const int lda, int *ipiv,
      double *work, const int lwork, int *info)
    {
      dgetri_(&n, a, &lda, ipiv, work, &lwork, info);
    }
    
    
    
    inline void gesv(const int n, const int nrhs, float *a, const int lda,
      int *ipiv, float *b, const int ldb, int *info)
    {
      sgesv_(&n, &nrhs, a, &lda, ipiv, b, &ldb, info);
    }
    
    inline void gesv(const int n, const int nrhs, double *a, const int lda,
      int *ipiv, double *b, const int ldb, int *info)
    {
      dgesv_(&n, &nrhs, a, &lda, ipiv, b, &ldb, info);
    }
    
    
    
    inline void lacpy(const char uplo, const int m, const int n, const float *a,
      const int lda, float *b, const int ldb)
    {
      slacpy_(&uplo, &m, &n, a, &lda, b, &ldb);
    }
    
    inline void lacpy(const char uplo, const int m, const int n, const double *a,
      const int lda, double *b, const int ldb)
    {
      dlacpy_(&uplo, &m, &n, a, &lda, b, &ldb);
    }
    
    
    
    inline void geqp3(const int m, const int n, float *x, const int lda,
      int *pivot, float *qraux, float *work, const int lwork, int *info)
    {
      sgeqp3_(&m, &n, x, &lda, pivot, qraux, work, &lwork, info);
    }
    
    inline void geqp3(const int m, const int n, double *x, const int lda,
      int *pivot, double *qraux, double *work, const int lwork, int *info)
    {
      dgeqp3_(&m, &n, x, &lda, pivot, qraux, work, &lwork, info);
    }
    
    
    
    inline void geqrf(const int m, const int n, float *x, const int lda,
      float *tau, float *work, const int lwork, int *info)
    {
      sgeqrf_(&m, &n, x, &lda, tau, work, &lwork, info);
    }
    
    inline void geqrf(const int m, const int n, double *x, const int lda,
      double *tau, double *work, const int lwork, int *info)
    {
      dgeqrf_(&m, &n, x, &lda, tau, work, &lwork, info);
    }
    
    
    
    inline void ormqr(const char side, const char trans, const int m,
      const int n, const int k, const float *x, const int lda,
      const float *tau, float *c, const int ldc, float *work,
      const int lwork, int *info)
    {
      sormqr_(&side, &trans, &m, &n, &k, x, &lda, tau, c, &ldc, work, &lwork, info);
    }
    
    inline void ormqr(const char side, const char trans, const int m,
      const int n, const int k, const double *x, const int lda,
      const double *tau, double *c, const int ldc, double *work,
      const int lwork, int *info)
    {
      dormqr_(&side, &trans, &m, &n, &k, x, &lda, tau, c, &ldc, work, &lwork, info);
    }
    
    
    
    inline void gelqf(const int m, const int n, float *x, const int lda,
      float *tau, float *work, const int lwork, int *info)
    {
      sgelqf_(&m, &n, x, &lda, tau, work, &lwork, info);
    }
    
    inline void gelqf(const int m, const int n, double *x, const int lda,
      double *tau, double *work, const int lwork, int *info)
    {
      dgelqf_(&m, &n, x, &lda, tau, work, &lwork, info);
    }
    
    
    
    inline void ormlq(const char side, const char trans, const int m,
      const int n, const int k, const float *x, const int lda,
      const float *tau, float *c, const int ldc, float *work,
      const int lwork, int *info)
    {
      sormlq_(&side, &trans, &m, &n, &k, x, &lda, tau, c, &ldc, work, &lwork, info);
    }
    
    inline void ormlq(const char side, const char trans, const int m,
      const int n, const int k, const double *x, const int lda,
      const double *tau, double *c, const int ldc, double *work,
      const int lwork, int *info)
    {
      dormlq_(&side, &trans, &m, &n, &k, x, &lda, tau, c, &ldc, work, &lwork, info);
    }
    
    
    
    inline void potrf(const char uplo, const int n, float *A, const int lda,
      int *info)
    {
      spotrf_(&uplo, &n, A, &lda, info);
    }
    
    inline void potrf(const char uplo, const int n, double *A, const int lda,
      int *info)
    {
      dpotrf_(&uplo, &n, A, &lda, info);
    }
    
    
    
    inline void lassq(const int n, const float *x, const int incx,
      float *scale, float *sumsq)
    {
      slassq_(&n, x, &incx, scale, sumsq);
    }
    
    inline void lassq(const int n, const double *x, const int incx,
      double *scale, double *sumsq)
    {
      dlassq_(&n, x, &incx, scale, sumsq);
    }
  }
}


#endif
