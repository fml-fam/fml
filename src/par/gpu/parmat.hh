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


template <typename REAL>
class parmat_gpu : public parmat<gpumat<REAL>, gpuvec<REAL>, REAL>
{
  using parmat<gpumat<REAL>, gpuvec<REAL>, REAL>::parmat;
  
  public:
    void fill_linspace(const REAL start, const REAL stop);
    void fill_eye();
    void fill_diag(const gpuvec<REAL> &d);
};



template <typename REAL>
void parmat_gpu<REAL>::fill_linspace(const REAL start, const REAL stop)
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
void parmat_cpu<REAL>::fill_eye()
{
  gpuvec<REAL> v(1);
  v.fill_val(1);
  this->fill_diag(v);
}



template <typename REAL>
void parmat_cpu<REAL>::fill_diag(const cpuvec<REAL> &d)
{
  
}


#endif
