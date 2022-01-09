// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_MPI_LINALG_DET_H
#define FML_MPI_LINALG_DET_H
#pragma once


#include <stdexcept>

#include "../../cpu/cpuvec.hh"

#include "../mpimat.hh"

#include "lu.hh"


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
}
}


#endif
