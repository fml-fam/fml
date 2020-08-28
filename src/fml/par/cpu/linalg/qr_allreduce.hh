// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_PAR_CPU_LINALG_QR_ALLREDUCE_H
#define FML_PAR_CPU_LINALG_QR_ALLREDUCE_H
#pragma once


#include "../../../_internals/arraytools/src/arraytools.hpp"
#include "../../../_internals/omp.hh"
#include "../../../_internals/restrict.hh"

#include "../../internals/mpi_utils.hh"

#include "../../../cpu/linalg/lapack.hh"
#include "../../../cpu/cpumat.hh"


namespace fml
{
namespace tsqr
{
  namespace internals
  {
    bool badinfo;
    int _m, _n, minmn, mtb;
    int lwork;
    
    template <typename REAL>
    REAL *tallboy;
    template <typename REAL>
    REAL *work;
    template <typename REAL>
    REAL *qraux;
    
    
    
    template <typename REAL>
    void qr_global_cleanup()
    {
      arraytools::free(tallboy<REAL>);
      tallboy<REAL> = NULL;
      
      arraytools::free(work<REAL>);
      work<REAL> = NULL;
      
      arraytools::free(qraux<REAL>);
      qraux<REAL> = NULL;
    }
    
    
    
    template <typename REAL>
    static inline int qrworksize(const int m, const int n)
    {
      REAL tmp;
      
      int info;
      fml::lapack::geqrf(m, n, NULL, m, NULL, &tmp, -1, &info);
      int lwork = (int) tmp;
      
      return std::max(lwork, 1);
    }
    
    
    
    template <typename REAL>
    void qr_global_init(int m, int n)
    {
      _m = m;
      _n = n;
      minmn = std::min(_m, _n);
      mtb = 2*_m;
      
      badinfo = false;
      
      arraytools::alloc(mtb*_n, &(tallboy<REAL>));
      lwork = qrworksize<REAL>(mtb, _n);
      arraytools::alloc(lwork, &(work<REAL>));
      arraytools::alloc(minmn, &(qraux<REAL>));
      
      arraytools::check_alloc(tallboy<REAL>, work<REAL>, qraux<REAL>);
    }
    
    
    
    template <typename REAL>
    void custom_op_qr(void *a_, void *b_, int *len, MPI_Datatype *dtype)
    {
      (void)len;
      (void)dtype;
      
      REAL *a = (REAL*)a_;
      REAL *b = (REAL*)b_;
      
      #pragma omp parallel for default(shared) if(_m*_n > omp::OMP_MIN_SIZE)
      for (int j=0; j<_n; j++)
      {
        #pragma omp simd
        for (int i=0; i<_m; i++)
          tallboy<REAL>[i + mtb*j] = a[i + _m*j];
        
        #pragma omp simd
        for (int i=0; i<_m; i++)
          tallboy<REAL>[_m+i + mtb*j] = b[i + _m*j];
      }
      
      int info = 0;
      fml::lapack::geqrf(mtb, _n, tallboy<REAL>, mtb, qraux<REAL>, work<REAL>, lwork, &info);
      if (info != 0)
        badinfo = true;
      
      for (int j=0; j<_n; j++)
      {
        #pragma omp for simd
        for (int i=0; i<=j; i++)
          b[i + _m*j] = tallboy<REAL>[i + mtb*j];
        
        #pragma omp for simd
        for (int i=j+1; i<_m; i++)
          b[i + _m*j] = (REAL) 0.f;
      }
    }
  }
  
  
  
  template <typename REAL>
  void qr_allreduce(const int root, const int m, const int n,
    const REAL *const restrict a, REAL *const restrict b, MPI_Comm comm)
  {
    int mpi_ret;
    
    internals::qr_global_init<REAL>(m, n);
    
    // custom data type
    MPI_Datatype mat_type;
    mpi::contig_type(m*n, a, &mat_type);
    
    // custom op + reduce
    MPI_Op op;
    const int commutative = 1;
    
    MPI_Op_create((MPI_User_function*) internals::custom_op_qr<REAL>, commutative, &op);
    if (root == mpi::REDUCE_TO_ALL)
      mpi_ret = MPI_Allreduce(a, b, 1, mat_type, op, comm);
    else
      mpi_ret = MPI_Reduce(a, b, 1, mat_type, op, root, comm);
    
    // cleanup and return
    MPI_Op_free(&op);
    MPI_Type_free(&mat_type);
    
    internals::qr_global_cleanup<REAL>();
    
    mpi::check_MPI_ret(mpi_ret);
    if (internals::badinfo)
      throw std::runtime_error("unrecoverable error with LAPACK function geqrf() occurred during reduction");
  }
}
}


#endif
