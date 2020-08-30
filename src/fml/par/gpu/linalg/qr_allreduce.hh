// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_PAR_GPU_LINALG_QR_ALLREDUCE_NOGPUDIRECT_H
#define FML_PAR_GPU_LINALG_QR_ALLREDUCE_NOGPUDIRECT_H
#pragma once


#include "../../../_internals/arraytools/src/arraytools.hpp"
#include "../../../_internals/restrict.hh"

#include "../../../gpu/card.hh"

#include "../../internals/mpi_utils.hh"


namespace fml
{
namespace tsqr
{
  namespace internals
  {
    // TODO mark inline when cuda gets C++17 support
    fml::card_sp_t c;
    cusolverStatus_t check;
    dim3 griddim, blockdim;
    int *info_dev;
    
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
    REAL *a;
    template <typename REAL>
    REAL *b;
    
    
    
    template <typename REAL>
    void qr_global_cleanup()
    {
      c->mem_free(tallboy<REAL>);
      tallboy<REAL> = NULL;
      
      c->mem_free(work<REAL>);
      work<REAL> = NULL;
      
      c->mem_free(qraux<REAL>);
      qraux<REAL> = NULL;
      
      c->mem_free(info_dev);
      
      c.reset();
    }
    
    
    
    template <typename REAL>
    static inline int qrworksize(const int m, const int n)
    {
      REAL tmp;
      
      int info;
      check = fml::gpulapack::geqrf_buflen(c->lapack_handle(), m, n, &tmp, m, &lwork);
      
      // TODO check
      (void)info;
      
      return std::max(lwork, 1);
    }
    
    
    
    template <typename REAL>
    void qr_global_init(fml::card_sp_t c_, int m, int n)
    {
      c = c_;
      blockdim = fml::kernel_launcher::dim_block2();
      griddim = fml::kernel_launcher::dim_grid(m, n);
      
      _m = m;
      _n = n;
      minmn = std::min(_m, _n);
      mtb = 2*_m;
      
      badinfo = false;
      
      tallboy<REAL> = (REAL*) c->mem_alloc((size_t)mtb*_n*sizeof(REAL));
      
      lwork = qrworksize<REAL>(mtb, _n);
      work<REAL> = (REAL*) c->mem_alloc((size_t)lwork*sizeof(REAL));
      
      qraux<REAL> = (REAL*) c->mem_alloc((size_t)minmn*sizeof(REAL));
      
      info_dev = (int*) c->mem_alloc(sizeof(int));
    }
    
    
    
    template <typename REAL>
    __global__ void kernel_stack(const len_t m, const len_t n, const len_t mtb,
      const REAL *a, const REAL *b, REAL *tallboy)
    {
      int i = blockDim.x*blockIdx.x + threadIdx.x;
      int j = blockDim.y*blockIdx.y + threadIdx.y;
      
      if (i < m && j < n)
      {
        tallboy[i + mtb*j] = a[i + m*j];
        tallboy[m+i + mtb*j] = b[i + m*j];
      }
    }
    
    
    
    template <typename REAL>
    void custom_op_qr(void *a_, void *b_, int *len, MPI_Datatype *dtype)
    {
      (void)len;
      (void)dtype;
      
      REAL *a_cpu = (REAL*)a_;
      REAL *b_cpu = (REAL*)b_;
      
      c->mem_cpu2gpu(a<REAL>, a_cpu, _m*_n*sizeof(REAL));
      c->mem_cpu2gpu(b<REAL>, b_cpu, _m*_n*sizeof(REAL));
      c->synch();
      
      kernel_stack<<<griddim, blockdim>>>(_m, _n, mtb, a<REAL>, b<REAL>, tallboy<REAL>);
      
      int info;
      c->mem_set(info_dev, 0, sizeof(int));
      check = fml::gpulapack::geqrf(c->lapack_handle(), mtb, _n, tallboy<REAL>, mtb, qraux<REAL>, work<REAL>, lwork, info_dev);
      c->mem_gpu2cpu(&info, info_dev, sizeof(int));
      // TODO check
      if (info != 0)
        badinfo = true;
      
      c->mem_set(b<REAL>, 0, (size_t)_m*_n*sizeof(REAL));
      fml::gpu_utils::lacpy('U', _m, _n, tallboy<REAL>, mtb, b<REAL>, _m);
      
      c->mem_gpu2cpu(b_cpu, b<REAL>, _m*_n*sizeof(REAL));
    }
  }
  
  
  
  template <typename REAL>
  void qr_allreduce(const int root, const int m, const int n,
    REAL *const restrict a_, REAL *const restrict b_, MPI_Comm comm,
    fml::card_sp_t c_)
  {
    int mpi_ret;
    
    internals::qr_global_init<REAL>(c_, m, n);
    
    internals::a<REAL> = a_;
    internals::b<REAL> = b_;
    
    REAL *a_cpu, *b_cpu;
    arraytools::alloc(m, n, &a_cpu);
    arraytools::alloc(m, n, &b_cpu);
    arraytools::check_alloc(a_cpu, b_cpu);
    
    c_->mem_gpu2cpu(a_cpu, a_, m*n*sizeof(REAL));
    c_->synch();
    
    // custom data type
    MPI_Datatype mat_type;
    mpi::contig_type(m*n, a_cpu, &mat_type);
    
    // custom op + reduce
    MPI_Op op;
    const int commutative = 1;
    
    MPI_Op_create((MPI_User_function*) internals::custom_op_qr<REAL>, commutative, &op);
    if (root == mpi::REDUCE_TO_ALL)
      mpi_ret = MPI_Allreduce(a_cpu, b_cpu, 1, mat_type, op, comm);
    else
      mpi_ret = MPI_Reduce(a_cpu, b_cpu, 1, mat_type, op, root, comm);
    
    // cleanup and return
    c_->mem_cpu2gpu(internals::b<REAL>, b_cpu, m*n*sizeof(REAL));
    
    MPI_Op_free(&op);
    MPI_Type_free(&mat_type);
    
    internals::qr_global_cleanup<REAL>();
    
    arraytools::free(a_cpu);
    arraytools::free(b_cpu);
    
    mpi::check_MPI_ret(mpi_ret);
    if (internals::badinfo)
      throw std::runtime_error("unrecoverable error with LAPACK function geqrf() occurred during reduction");
  }
}
}


#endif
