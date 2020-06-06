// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_MPI_LINALG_LINALG_MISC_H
#define FML_MPI_LINALG_LINALG_MISC_H
#pragma once


#include <stdexcept>

#include "../../cpu/cpuvec.hh"

#include "../mpimat.hh"

#include "linalg_factorizations.hh"


namespace fml
{
namespace linalg
{
  /**
    @brief Computes the determinant in logarithmic form.
    
    @details The input is replaced by its LU factorization.
    
    @param[inout] x Input data matrix, replaced by its LU factorization.
    @param[out] sign The sign of the determinant.
    @param[out] modulus Log of the modulus.
    
    @impl Uses `lu()`.
    
    @allocs Allocates temporary storage to compute the LU.
    
    @except If an allocation is triggered and fails, a `bad_alloc` exception
    will be thrown.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  void det(mpimat<REAL> &x, int &sign, REAL &modulus)
  {
    if (!x.is_square())
      throw std::runtime_error("'x' must be a square matrix");
    
    cpuvec<int> p;
    int info;
    lu(x, p, info);
    
    if (info != 0)
    {
      if (info > 0)
      {
        sign = 1;
        modulus = -INFINITY;
        return;
      }
      else
        return;
    }
    
    
    // get determinant
    REAL mod = 0.0;
    int sgn = 1;
    
    const len_t m_local = x.nrows_local();
    const len_t n_local = x.ncols_local();
    
    const int *ipiv = p.data_ptr();
    const REAL *a = x.data_ptr();
    const grid g = x.get_grid();
    
    for (len_t i=0; i<m_local; i++)
    {
      len_t gi = fml::bcutils::l2g(i, x.bf_rows(), g.nprow(), g.myrow());
      
      if (ipiv[i] != (gi + 1))
        sgn = -sgn;
    }
    
    for (len_t j=0; j<n_local; j++)
    {
      for (len_t i=0; i<m_local; i++)
      {
        len_t gi = fml::bcutils::l2g(i, x.bf_rows(), g.nprow(), g.myrow());
        len_t gj = fml::bcutils::l2g(j, x.bf_cols(), g.npcol(), g.mycol());
        
        if (gi == gj)
        {
          const REAL d = a[i + m_local*j];
          if (d < 0)
          {
            mod += log(-d);
            sgn *= -1;
          }
          else
            mod += log(d);
        }
      }
    }
    
    g.allreduce(1, 1, &mod);
    
    sgn = (sgn<0 ? 1 : 0);
    g.allreduce(1, 1, &sgn, 'C');
    sgn = (sgn%2==0 ? 1 : -1);
    
    modulus = mod;
    sign = sgn;
  }
  
  
  
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
