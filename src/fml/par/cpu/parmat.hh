// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_PAR_CPU_PARMAT_H
#define FML_PAR_CPU_PARMAT_H
#pragma once


#include "../../_internals/omp.hh"

#include "../../cpu/cpumat.hh"
#include "../../cpu/cpuvec.hh"

#include "../internals/parmat.hh"


namespace fml
{
  template <typename REAL>
  class parmat_cpu : public parmat<cpumat<REAL>, cpuvec<REAL>, REAL>
  {
    using parmat<cpumat<REAL>, cpuvec<REAL>, REAL>::parmat;
    
    public:
      parmat_cpu(comm mpi_comm);
      parmat_cpu(comm mpi_comm, const len_global_t nrows, const len_t ncols);
      parmat_cpu(comm mpi_comm, const len_global_t nrows, const len_t ncols, const len_global_t nb4_);
      
      void print(uint8_t ndigits=4, bool add_final_blank=true);
      
      // void resize(len_global_t nrows, len_t ncols);
      // void inherit(cpumat<REAL> &data_);
      
      void fill_linspace(const REAL start, const REAL stop);
      void fill_eye();
      void fill_diag(const cpuvec<REAL> &d);
  };
}



template <typename REAL>
fml::parmat_cpu<REAL>::parmat_cpu(fml::comm mpi_comm)
{
  this->r = mpi_comm;
  this->m_global = 0;
  this->nb4 = 0;
}



template <typename REAL>
fml::parmat_cpu<REAL>::parmat_cpu(fml::comm mpi_comm, const len_global_t nrows, const len_t ncols)
{
  this->r = mpi_comm;
  
  this->m_global = nrows;
  len_t nrows_local = this->get_local_dim();
  this->data.resize(nrows_local, ncols);
  
  this->num_preceding_rows();
}



template <typename REAL>
fml::parmat_cpu<REAL>::parmat_cpu(fml::comm mpi_comm, const len_global_t nrows, const len_t ncols, const len_global_t nb4_)
{
  this->r = mpi_comm;
  
  this->m_global = nrows;
  len_t nrows_local = this->get_local_dim();
  this->data.resize(nrows_local, ncols);
  
  this->nb4 = nb4_;
}



// template <typename REAL>
// void fml::parmat_cpu<REAL>::resize(len_global_t nrows, len_t ncols)
// {
// 
// }
// 
// 
// 
// template <typename REAL>
// void fml::parmat_cpu<REAL>::inherit(cpumat<REAL> &data_, bool free_on_destruct)
// {
// 
// }



template <typename REAL>
void fml::parmat_cpu<REAL>::print(uint8_t ndigits, bool add_final_blank)
{
  len_t n = this->data.ncols();
  cpuvec<REAL> pv(n);
  
  int myrank = this->r.rank();
  if (myrank == 0)
    this->data.print(ndigits, false);
  
  for (int rank=1; rank<this->r.size(); rank++)
  {
    if (rank == myrank)
    {
      len_t m = this->data.nrows();
      this->r.send(1, &m, 0);
      
      for (int i=0; i<m; i++)
      {
        this->data.get_row(i, pv);
        this->r.send(n, pv.data_ptr(), 0);
      }
    }
    else if (myrank == 0)
    {
      len_t m;
      this->r.recv(1, &m, rank);
      
      for (int i=0; i<m; i++)
      {
        this->r.recv(n, pv.data_ptr(), rank);
        pv.print(ndigits, false);
      }
    }
  
    this->r.barrier();
  }
  
  if (add_final_blank)
  {
    this->r.printf(0, "\n");
    this->r.barrier();
  }
}



template <typename REAL>
void fml::parmat_cpu<REAL>::fill_linspace(const REAL start, const REAL stop)
{
  if (start == stop)
    this->fill_val(start);
  else
  {
    const len_t m_local = this->data.nrows();
    const len_t n = this->data.ncols();
    
    const REAL v = (stop-start)/((REAL) this->m_global*n - 1);
    REAL *d_p = this->data.data_ptr();
    
    #pragma omp parallel for if(m_local*n > fml::omp::OMP_MIN_SIZE)
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
void fml::parmat_cpu<REAL>::fill_eye()
{
  fml::cpuvec<REAL> v(1);
  v(0) = (REAL) 1;
  this->fill_diag(v);
}



template <typename REAL>
void fml::parmat_cpu<REAL>::fill_diag(const fml::cpuvec<REAL> &d)
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
