// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_MPI_LINALG_LU_H
#define FML_MPI_LINALG_LU_H
#pragma once


#include "../../_internals/linalgutils.hh"
#include "../../cpu/cpuvec.hh"

#include "../mpimat.hh"

#include "internals/scalapack.hh"


namespace fml
{
namespace linalg
{
  /**
    @brief Computes the PLU factorization with partial pivoting.
    
    @details The input is replaced by its LU factorization, with L
    unit-diagonal.
    
    @param[inout] x Input data matrix, replaced by its LU factorization.
    @param[out] p Vector of pivots, representing the diagonal matrix P in the
    PLU.
    @param[out] info The ScaLAPACK return number.
    
    @impl Uses the ScaLAPACK function `pXgetrf()`.
    
    @allocs If the pivot vector is inappropriately sized, it will automatically
    be re-allocated.
    
    @except If a reallocation is triggered and fails, a `bad_alloc` exception
    will be thrown.
    
    @comm The method will communicate across all processes in the BLACS grid.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  void lu(mpimat<REAL> &x, cpuvec<int> &p, int &info)
  {
    info = 0;
    const len_t m = x.nrows();
    const len_t lipiv = std::min(m, x.ncols());
    
    p.resize(lipiv);
    
    fml::scalapack::getrf(m, x.ncols(), x.data_ptr(), x.desc_ptr(), p.data_ptr(), &info);
  }
  
  /// \overload
  template <typename REAL>
  void lu(mpimat<REAL> &x)
  {
    cpuvec<int> p;
    int info;
    
    lu(x, p, info);
    
    fml::linalgutils::check_info(info, "getrf");
  }
}
}


#endif
