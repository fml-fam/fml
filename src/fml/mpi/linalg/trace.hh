// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_MPI_LINALG_TRACE_H
#define FML_MPI_LINALG_TRACE_H
#pragma once


#include "../internals/bcutils.hh"

#include "../mpimat.hh"


namespace fml
{
namespace linalg
{
  /**
    @brief Computes the trace, i.e. the sum of the diagonal.
    
    @param[in] x Input data matrix.
    
    @comm The method will communicate across all processes in the BLACS grid.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  REAL trace(const mpimat<REAL> &x)
  {
    const REAL *x_d = x.data_ptr();
    const len_t minmn = std::min(x.nrows(), x.ncols());
    const len_t m_local = x.nrows_local();
    const int mb = x.bf_rows();
    const int nb = x.bf_cols();
    const grid g = x.get_grid();
    
    REAL tr = 0;
    for (len_t gi=0; gi<minmn; gi++)
    {
      const len_local_t i = fml::bcutils::g2l(gi, mb, g.nprow());
      const len_local_t j = fml::bcutils::g2l(gi, nb, g.npcol());
      
      const int pr = fml::bcutils::g2p(gi, mb, g.nprow());
      const int pc = fml::bcutils::g2p(gi, nb, g.npcol());
      
      if (pr == g.myrow() && pc == g.mycol())
        tr += x_d[i + m_local*j];
    }
    
    g.allreduce(1, 1, &tr, 'A');
    
    return tr;
  }
}
}


#endif
