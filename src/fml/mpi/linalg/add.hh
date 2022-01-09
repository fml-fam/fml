// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_MPI_LINALG_ADD_H
#define FML_MPI_LINALG_ADD_H
#pragma once


#include "../../_internals/linalgutils.hh"

#include "../mpimat.hh"

#include "internals/err.hh"
#include "internals/pblas.hh"


namespace fml
{
namespace linalg
{
  /**
    @brief Returns alpha*op(x) + beta*op(y) where op(A) is A or A^T
    
    @param[in] transx Should x^T be used?
    @param[in] transy Should y^T be used?
    @param[in] alpha,beta Scalars.
    @param[in] x,y The inputs to the sum.
    @param[out] ret The sum.
    
    @except If x and y are inappropriately sized for the sum, the method will
    throw a 'runtime_error' exception.
    
    @comm The method will communicate across all processes in the BLACS grid.
    
    @impl Uses the PBLAS function `pXgeadd()`.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  void add(const bool transx, const bool transy, const REAL alpha, const REAL beta, const mpimat<REAL> &x, const mpimat<REAL> &y, mpimat<REAL> &ret)
  {
    err::check_grid(x, y, ret);
    
    len_t m, n;
    fml::linalgutils::matadd_params(transx, transy, x.nrows(), x.ncols(), y.nrows(), y.ncols(), &m, &n);
    
    if (ret.nrows() != m || ret.ncols() != n)
      ret.resize(m, n);
    
    char ctransx = transx ? 'T' : 'N';
    char ctransy = transy ? 'T' : 'N';
    
    fml::pblas::geadd(ctransy, m, n, beta, y.data_ptr(), y.desc_ptr(), (REAL) 0.0f, ret.data_ptr(), ret.desc_ptr());
    fml::pblas::geadd(ctransx, m, n, alpha, x.data_ptr(), x.desc_ptr(), (REAL) 1.0f, ret.data_ptr(), ret.desc_ptr());
  }
  
  
  
  /// \overload
  template <typename REAL>
  mpimat<REAL> add(const bool transx, const bool transy, const REAL alpha, const REAL beta, const mpimat<REAL> &x, const mpimat<REAL> &y)
  {
    err::check_grid(x, y);
    
    len_t m, n;
    fml::linalgutils::matadd_params(transx, transy, x.nrows(), x.ncols(), y.nrows(), y.ncols(), &m, &n);
    
    const grid g = x.get_grid();
    mpimat<REAL> ret(g, m, n, x.bf_rows(), x.bf_cols());
    add(transx, transy, alpha, beta, x, y, ret);
    return ret;
  }
}
}


#endif
