// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_MPI_INTERNALS_MPI_UTILS_H
#define FML_MPI_INTERNALS_MPI_UTILS_H
#pragma once


#include "bcutils.hh"

class grid;


namespace fml
{
  namespace mpi_utils
  {
    // zero specified triangle
    template <typename REAL>
    void tri2zero(const char uplo, const bool diag, const grid &g,
      const len_t m, const len_t n, REAL *A, const int *desc)
    {
      const int mb = desc[fml::bcutils::DESC_MB];
      const int nb = desc[fml::bcutils::DESC_NB];
      
      const len_t m_local = fml::bcutils::numroc(desc[fml::bcutils::DESC_M], mb, g.myrow(), 0, g.nprow());
      const len_t n_local = fml::bcutils::numroc(desc[fml::bcutils::DESC_N], nb, g.mycol(), 0, g.npcol());
      
      for (len_t j=0; j<n_local; j++)
      {
        for (len_t i=0; i<m_local; i++)
        {
          const int gi = fml::bcutils::l2g(i, mb, g.nprow(), g.myrow());
          const int gj = fml::bcutils::l2g(j, nb, g.npcol(), g.mycol());
          
          if ((gi < m && gj < n) && ((diag && gi == gj) || (uplo == 'U' && gi < gj) || (uplo == 'L' && gi > gj)))
            A[i + m_local*j] = (REAL) 0.0;
        }
      }
    }
  }
}


#endif
