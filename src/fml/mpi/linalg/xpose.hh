// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_MPI_LINALG_XPOSE_H
#define FML_MPI_LINALG_XPOSE_H
#pragma once


#include <stdexcept>

#include "../../_internals/linalgutils.hh"

#include "../mpimat.hh"

#include "internals/err.hh"
#include "internals/pblas.hh"


namespace fml
{
namespace linalg
{
  /**
    @brief Computes the transpose out-of-place (i.e. in a copy).
    
    @param[in] x Input data matrix.
    @param[out] tx The transpose.
    
    @impl Uses the PBLAS function `pXtran()`.
    
    @allocs If the output dimension is inappropriately sized, it will
    automatically be re-allocated.
    
    @except If a reallocation is triggered and fails, a `bad_alloc` exception
    will be thrown.
    
    @comm The method will communicate across all processes in the BLACS grid.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  void xpose(const mpimat<REAL> &x, mpimat<REAL> &tx)
  {
    err::check_grid(x, tx);
    
    const len_t m = x.nrows();
    const len_t n = x.ncols();
    
    if (m != tx.ncols() || n != tx.nrows())
      tx.resize(n, m);
    
    fml::pblas::tran(n, m, 1.f, x.data_ptr(), x.desc_ptr(), 0.f, tx.data_ptr(), tx.desc_ptr());
  }
  
  
  
  /// \overload
  template <typename REAL>
  mpimat<REAL> xpose(const mpimat<REAL> &x)
  {
    const len_t m = x.nrows();
    const len_t n = x.ncols();
    const grid g = x.get_grid();
    
    mpimat<REAL> tx(g, n, m, x.bf_rows(), x.bf_cols());
    xpose(x, tx);
    return tx;
  }
}
}


#endif
