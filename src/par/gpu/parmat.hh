// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_PAR_GPU_PARMAT_H
#define FML_PAR_GPU_PARMAT_H
#pragma once


#include "../../gpu/card.hh"
#include "../../gpu/gpumat.hh"
#include "../../gpu/gpuvec.hh"

#include "../internals/parmat.hh"


namespace fml
{
  template <typename REAL>
  class parmat_gpu : public parmat<gpumat<REAL>, gpuvec<REAL>, REAL>
  {
    using parmat<gpumat<REAL>, gpuvec<REAL>, REAL>::parmat;
    
    public:
      parmat_gpu(comm &mpi_comm, const len_global_t nrows, const len_t ncols);
      
      void fill_linspace(const REAL start, const REAL stop);
      void fill_eye();
      void fill_diag(const gpuvec<REAL> &d);
  };
}



template <typename REAL>
fml::parmat_gpu<REAL>::parmat_gpu(fml::comm &mpi_comm, const len_global_t nrows, const len_t ncols)
{
  this->r = mpi_comm;
  
  this->m_global = nrows;
  len_t nrows_local = this->get_local_dim();
  this->data.resize(nrows_local, ncols);
  
  this->m_global = (len_global_t) nrows_local;
  this->r.allreduce(1, &(this->m_global));
  this->num_preceding_rows();
}



template <typename REAL>
void fml::parmat_gpu<REAL>::fill_linspace(const REAL start, const REAL stop)
{
  if (start == stop)
    this->fill_val(start);
  else
  {
    const len_t m_local = this->data.nrows();
    const len_t n = this->data.ncols();
    
    const REAL v = (stop-start)/((REAL) this->m_global*n - 1);
    
    // TODO
    // kernelfuns::kernel_fill_linspace<<<dim_grid, dim_block>>>(start, stop, this->m, this->n, this->data);
    
    this->c->check();
  }
}



template <typename REAL>
void fml::parmat_gpu<REAL>::fill_eye()
{
  fml::gpuvec<REAL> v(1);
  v.fill_val(1);
  this->fill_diag(v);
}



template <typename REAL>
void fml::parmat_gpu<REAL>::fill_diag(const fml::gpuvec<REAL> &d)
{
  
}


#endif
