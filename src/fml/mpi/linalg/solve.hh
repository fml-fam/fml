// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_MPI_LINALG_SOLVE_H
#define FML_MPI_LINALG_SOLVE_H
#pragma once


#include <stdexcept>

#include "../../_internals/linalgutils.hh"
#include "../../cpu/cpuvec.hh"

#include "../internals/bcutils.hh"
#include "../internals/mpi_utils.hh"

#include "../copy.hh"
#include "../mpimat.hh"

#include "internals/err.hh"
#include "internals/scalapack.hh"
#include "lu.hh"


namespace fml
{
namespace linalg
{
  /**
    @brief Solve a system of equations.
    
    @details The input is replaced by its LU factorization.
    
    @param[inout] x Input LHS. Should be square. Overwritten by LU.
    @param[inout] y Input RHS. Overwritten by solution.
    
    @impl Uses the ScaLAPACK functions `pXgesv()`.
    
    @allocs LU pivot data is allocated internally.
    
    @except If the matrix is non-square or if the RHS is incompatible with the
    LHS, a `runtime_error` exception is thrown. If the inputs are distributed
    on different grids, a `runtime_exception` is thrown. If an allocation
    fails, a `bad_alloc` exception will be thrown.
    
    @comm The method will communicate across all processes in the BLACS grid.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  void solve(mpimat<REAL> &x, mpimat<REAL> &y)
  {
    err::check_grid(x, y);
    
    const len_t n = x.nrows();
    if (!x.is_square())
      throw std::runtime_error("'x' must be a square matrix");
    if (n != y.nrows())
      throw std::runtime_error("rhs 'y' must be compatible with data matrix 'x'");
    
    int info;
    cpuvec<int> p(n);
    fml::scalapack::gesv(n, y.ncols(), x.data_ptr(), x.desc_ptr(), p.data_ptr(), y.data_ptr(), y.desc_ptr(), &info);
    fml::linalgutils::check_info(info, "gesv");
  }
}
}


#endif
