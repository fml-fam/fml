// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_MPI_LINALG_SCALAPACK_H
#define FML_MPI_LINALG_SCALAPACK_H
#pragma once


#include "_scalapack_prototypes.h"


namespace fml
{
  namespace scalapack
  {
    inline void getrf(const int m, const int n, float *a, int *desca, int *ipiv,
      int *info)
    {
      int ij = 1;
      psgetrf_(&m, &n, a, &ij, &ij, desca, ipiv, info);
    }
    
    inline void getrf(const int m, const int n, double *a, int *desca, int *ipiv,
      int *info)
    {
      int ij = 1;
      pdgetrf_(&m, &n, a, &ij, &ij, desca, ipiv, info);
    }
    
    
    
    inline void gesvd(const char jobu, const char jobvt, const int m, const int n,
      float *a, int *desca, float *s, float *u, int *descu, float *vt,
      int *descvt, float *work, int lwork, int *info)
    {
      int ij = 1;
      psgesvd_(&jobu, &jobvt, &m, &n, a, &ij, &ij, desca, s, u, &ij, &ij, descu,
        vt, &ij, &ij, descvt, work, &lwork, info);
    }
    
    inline void gesvd(const char jobu, const char jobvt, const int m, const int n,
      double *a, int *desca, double *s, double *u, int *descu, double *vt,
      int *descvt, double *work, int lwork, int *info)
    {
      int ij = 1;
      pdgesvd_(&jobu, &jobvt, &m, &n, a, &ij, &ij, desca, s, u, &ij, &ij, descu,
        vt, &ij, &ij, descvt, work, &lwork, info);
    }
    
    
    
    inline void syevr(const char jobz, const char range, const char uplo,
      const int n, float *a, const int *desca, const float vl, const float vu,
      const int il, const int iu, int *m, int *nz, float *w, float *z,
      const int *descz, float *work, const int lwork, int *iwork,
      const int liwork, int *info)
    {
      int ij = 1;
      pssyevr_(&jobz, &range, &uplo, &n, a, &ij, &ij, desca, &vl, &vu, &il, &iu,
        m, nz, w, z, &ij, &ij, descz, work, &lwork, iwork, &liwork, info);
    }
    
    inline void syevr(const char jobz, const char range, const char uplo,
      const int n, double *a, const int *desca, const double vl, const double vu,
      const int il, const int iu, int *m, int *nz, double *w, double *z,
      const int *descz, double *work, const int lwork, int *iwork,
      const int liwork, int *info)
    {
      int ij = 1;
      pdsyevr_(&jobz, &range, &uplo, &n, a, &ij, &ij, desca, &vl, &vu, &il, &iu,
        m, nz, w, z, &ij, &ij, descz, work, &lwork, iwork, &liwork, info);
    }
    
    
    
    inline void getri(const int n, float *a, const int *desca, int *ipiv,
      float *work, const int lwork, int *iwork, const int liwork, int *info)
    {
      int ij = 1;
      psgetri_(&n, a, &ij, &ij, desca, ipiv, work, &lwork, iwork, &liwork, info);
    }
    
    inline void getri(const int n, double *a, const int *desca, int *ipiv,
      double *work, const int lwork, int *iwork, const int liwork, int *info)
    {
      int ij = 1;
      pdgetri_(&n, a, &ij, &ij, desca, ipiv, work, &lwork, iwork, &liwork, info);
    }
    
    
    
    inline void gesv(const int n, const int nrhs, float *a, const int *desca,
      int *ipvt, float *b, const int *descb, int *info)
    {
      int ij = 1;
      psgesv_(&n, &nrhs, a, &ij, &ij, desca, ipvt, b, &ij, &ij, descb, info);
    }
    
    inline void gesv(const int n, const int nrhs, double *a, const int *desca,
      int *ipvt, double *b, const int *descb, int *info)
    {
      int ij = 1;
      pdgesv_(&n, &nrhs, a, &ij, &ij, desca, ipvt, b, &ij, &ij, descb, info);
    }
    
    
    
    inline void lacpy(const char uplo, const int m, const int n,
      const float *a, const int *desca, float *b, const int *descb)
    {
      int ij = 1;
      pslacpy_(&uplo, &m, &n, a, &ij, &ij, desca, b, &ij, &ij, descb);
    }
    
    inline void lacpy(const char uplo, const int m, const int n,
      const double *a, const int *desca, double *b, const int *descb)
    {
      int ij = 1;
      pdlacpy_(&uplo, &m, &n, a, &ij, &ij, desca, b, &ij, &ij, descb);
    }
    
    
    
    inline void geqpf(const int m, const int n, float *a, const int *desca,
      int *ipiv, float *tau, float *work, const int lwork, int *info)
    {
      int ij = 1;
      psgeqpf_(&m, &n, a, &ij, &ij, desca, ipiv, tau, work, &lwork, info);
    }
    
    inline void geqpf(const int m, const int n, double *a, const int *desca,
      int *ipiv, double *tau, double *work, const int lwork, int *info)
    {
      int ij = 1;
      pdgeqpf_(&m, &n, a, &ij, &ij, desca, ipiv, tau, work, &lwork, info);
    }
    
    
    
    inline void geqrf(const int m, const int n, float *a, const int *desca,
      float *tau, float *work, const int lwork, int *info)
    {
      int ij = 1;
      psgeqrf_(&m, &n, a, &ij, &ij, desca, tau, work, &lwork, info);
    }
    
    inline void geqrf(const int m, const int n, double *a, const int *desca,
      double *tau, double *work, const int lwork, int *info)
    {
      int ij = 1;
      pdgeqrf_(&m, &n, a, &ij, &ij, desca, tau, work, &lwork, info);
    }
    
    
    
    inline void ormqr(const char side, const char trans, const int m,
      const int n, const int k, const float *a, const int *desca, float *tau,
      float *c, const int *descc, float *work, const int lwork, int *info)
    {
      int ij = 1;
      psormqr_(&side, &trans, &m, &n, &k, a, &ij, &ij, desca, tau, c, &ij, &ij,
        descc, work, &lwork, info);
    }
    
    inline void ormqr(const char side, const char trans, const int m,
      const int n, const int k, const double *a, const int *desca, double *tau,
      double *c, const int *descc, double *work, const int lwork, int *info)
    {
      int ij = 1;
      pdormqr_(&side, &trans, &m, &n, &k, a, &ij, &ij, desca, tau, c, &ij, &ij,
        descc, work, &lwork, info);
    }
    
    
    
    inline void gelqf(const int m, const int n, float *a, const int *desca,
      float *tau, float *work, const int lwork, int *info)
    {
      int ij = 1;
      psgelqf_(&m, &n, a, &ij, &ij, desca, tau, work, &lwork, info);
    }
    
    inline void gelqf(const int m, const int n, double *a, const int *desca,
      double *tau, double *work, const int lwork, int *info)
    {
      int ij = 1;
      pdgelqf_(&m, &n, a, &ij, &ij, desca, tau, work, &lwork, info);
    }
    
    
    
    inline void ormlq(const char side, const char trans, const int m,
      const int n, const int k, const float *a, const int *desca, float *tau,
      float *c, const int *descc, float *work, const int lwork, int *info)
    {
      int ij = 1;
      psormlq_(&side, &trans, &m, &n, &k, a, &ij, &ij, desca, tau, c, &ij, &ij,
        descc, work, &lwork, info);
    }
    
    inline void ormlq(const char side, const char trans, const int m,
      const int n, const int k, const double *a, const int *desca, double *tau,
      double *c, const int *descc, double *work, const int lwork, int *info)
    {
      int ij = 1;
      pdormlq_(&side, &trans, &m, &n, &k, a, &ij, &ij, desca, tau, c, &ij, &ij,
        descc, work, &lwork, info);
    }
    
    
    
    inline void potrf(const char uplo, const int n, float *a, const int *desca,
      int *info)
    {
      int ij = 1;
      pspotrf_(&uplo, &n, a, &ij, &ij, desca, info);
    }
    
    inline void potrf(const char uplo, const int n, double *a, const int *desca,
      int *info)
    {
      int ij = 1;
      pdpotrf_(&uplo, &n, a, &ij, &ij, desca, info);
    }
    
    
    
    inline void lassq(const int n, const float *x, const int *descx,
      const int incx, float *scale, float *sumsq)
    {
      int ij = 1;
      pslassq_(&n, x, &ij, &ij, descx, &incx, scale, sumsq);
    }
    
    inline void lassq(const int n, const double *x, const int *descx,
      const int incx, double *scale, double *sumsq)
    {
      int ij = 1;
      pdlassq_(&n, x, &ij, &ij, descx, &incx, scale, sumsq);
    }
    
    
    
    inline void gecon(const char norm, const int n,
      const float *const restrict A, const int *desca, const float anorm,
      float *rcond, float *work, const int lwork, int *iwork, const int liwork,
      int *info)
    {
      int ij = 1;
      psgecon_(&norm, &n, A, &ij, &ij, desca, &anorm, rcond, work, &lwork,
        iwork, &liwork, info);
    }
    
    inline void gecon(const char norm, const int n,
      const double *const restrict A, const int *desca, const double anorm,
      double *rcond, double *work, const int lwork, int *iwork, const int liwork,
      int *info)
    {
      int ij = 1;
      pdgecon_(&norm, &n, A, &ij, &ij, desca, &anorm, rcond, work, &lwork,
        iwork, &liwork, info);
    }
    
    
    
    inline void trcon(const char norm, const char uplo, const char diag,
      const int n, const float *const restrict A, const int *desca,
      float *rcond, float *work, const int lwork, int *iwork,
      const int liwork, int *info)
    {
      int ij = 1;
      pstrcon_(&norm, &uplo, &diag, &n, A, &ij, &ij, desca, rcond, work, &lwork,
        iwork, &liwork, info);
    }
    
    inline void trcon(const char norm, const char uplo, const char diag,
      const int n, const double *const restrict A, const int *desca,
      double *rcond, double *work, const int lwork, int *iwork,
      const int liwork, int *info)
    {
      int ij = 1;
      pdtrcon_(&norm, &uplo, &diag, &n, A, &ij, &ij, desca, rcond, work, &lwork,
        iwork, &liwork, info);
    }
  }
}


#endif
