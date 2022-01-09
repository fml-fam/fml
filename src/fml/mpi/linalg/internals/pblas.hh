// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_MPI_LINALG_INTERNALS_PBLAS_H
#define FML_MPI_LINALG_INTERNALS_PBLAS_H
#pragma once


#include "pblas_prototypes.h"


namespace fml
{
  namespace pblas
  {
    inline void gemm(const char transa, const char transb, const int m, const int n,
      const int k, const float alpha, const float *a, const int *desca,
      const float *b, const int *descb, const float beta, float *c,
      const int *descc)
    {
      int ij = 1;
      psgemm_(&transa, &transb, &m, &n, &k, &alpha, a, &ij, &ij, desca, b,
        &ij, &ij, descb, &beta, c, &ij, &ij, descc);
    }
    
    inline void gemm(const char transa, const char transb, const int m, const int n,
      const int k, const double alpha, const double *a, const int *desca,
      const double *b, const int *descb, const double beta, double *c,
      const int *descc)
    {
      int ij = 1;
      pdgemm_(&transa, &transb, &m, &n, &k, &alpha, a, &ij, &ij, desca, b,
        &ij, &ij, descb, &beta, c, &ij, &ij, descc);
    }
    
    
    
    inline void syrk(const char uplo, const char trans, const int n, const int k, 
      const float alpha, const float *a, const int *desca,
      const float beta, float *c, const int *descc)
    {
      int ij = 1;
      pssyrk_(&uplo, &trans, &n, &k, &alpha, a, &ij, &ij, desca, &beta, c, &ij,
        &ij, descc);
    }
    
    inline void syrk(const char uplo, const char trans, const int n, const int k, 
      const double alpha, const double *a, const int *desca,
      const double beta, double *c, const int *descc)
    {
      int ij = 1;
      pdsyrk_(&uplo, &trans, &n, &k, &alpha, a, &ij, &ij, desca, &beta, c, &ij,
        &ij, descc);
    }
    
    
    
    inline void tran(const int m, const int n, const float alpha, const float *a,
      const int *desca, const float beta, float *c, const int *descc)
    {
      int ij = 1;
      pstran_(&m, &n, &alpha, a, &ij, &ij, desca, &beta, c, &ij, &ij, descc);
    }
    
    inline void tran(const int m, const int n, const double alpha, const double *a,
      const int *desca, const double beta, double *c, const int *descc)
    {
      int ij = 1;
      pdtran_(&m, &n, &alpha, a, &ij, &ij, desca, &beta, c, &ij, &ij, descc);
    }
    
    
    
    inline void geadd(const char trans, const int m, const int n,
      const float alpha, const float *a, const int *desca, const float beta,
      float *c, const int *descc)
    {
      int ij = 1;
      psgeadd_(&trans, &m, &n, &alpha, a, &ij, &ij, desca, &beta, c, &ij, &ij,
        descc);
    }
    
    inline void geadd(const char trans, const int m, const int n,
      const double alpha, const double *a, const int *desca, const double beta,
      double *c, const int *descc)
    {
      int ij = 1;
      pdgeadd_(&trans, &m, &n, &alpha, a, &ij, &ij, desca, &beta, c, &ij, &ij,
        descc);
    }
  }
}


#endif
