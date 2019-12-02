// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_PAR_CPU_H
#define FML_PAR_CPU_H
#pragma once


#include "../cpu/cpumat.hh"
#include "../cpu/cpuvec.hh"
#include "parmat.hh"


template <typename REAL>
class parmat_cpu : public parmat<cpumat<REAL>, cpuvec<REAL>, REAL>
{
  using parmat<cpumat<REAL>, cpuvec<REAL>, REAL>::parmat;
  
  public:
    parmat_cpu(comm &mpi_comm, len_global_t nrows, len_t ncols);
    void fill_linspace(const REAL start, const REAL stop);
    void fill_eye();
    void fill_diag(const cpuvec<REAL> &d);
};



template <typename REAL>
parmat_cpu<REAL>::parmat_cpu(comm &mpi_comm, len_global_t nrows, len_t ncols)
{
  this->r = mpi_comm;
  
  this->m_global = nrows;
  len_t nrows_local = this->get_local_dim();
  this->data = cpumat<REAL>(nrows_local, ncols);
  
  this->m_global = (len_global_t) nrows_local;
  this->r.allreduce(1, &(this->m_global));
  this->num_preceding_rows();
}



template <typename REAL>
void parmat_cpu<REAL>::fill_linspace(const REAL start, const REAL stop)
{
  if (start == stop)
    this->fill_val(start);
  else
  {
    const len_t m_local = this->data.nrows();
    const len_t n = this->data.ncols();
    
    const REAL v = (stop-start)/((REAL) this->m_global*n - 1);
    REAL *d_p = this->data.data_ptr();
    
    #pragma omp parallel for if(m_local*n > omputils::OMP_MIN_SIZE)
    for (len_t j=0; j<n; j++)
    {
      #pragma omp simd
      for (len_t i=0; i<m_local; i++)
      {
        d_p[i + m_local*j] = v*((REAL) i + this->nb4 + this->m_global*j) + start;
      }
    }
  }
}



template <typename REAL>
void parmat_cpu<REAL>::fill_eye()
{
  cpuvec<REAL> v(1);
  v(0) = (REAL) 1;
  this->fill_diag(v);
}



template <typename REAL>
void parmat_cpu<REAL>::fill_diag(const cpuvec<REAL> &d)
{
  const len_t m_local = this->data.nrows();
  const len_t n = this->data.ncols();
  REAL *x_p = this->data.data_ptr();
  const REAL *d_p = d.data_ptr();
  
  #pragma omp for simd
  for (len_t j=0; j<n; j++)
  {
    for (len_t i=0; i<m_local; i++)
    {
      const len_global_t gi = i + this->nb4;
      if (gi == j)
        x_p[i + m_local*j] = d_p[gi % d.size()];
      else
        x_p[i + m_local*j] = 0;
    }
  }
}


#endif
