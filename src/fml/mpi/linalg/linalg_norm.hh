// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_MPI_LINALG_LINALG_NORM_H
#define FML_MPI_LINALG_LINALG_NORM_H
#pragma once


#include "../../cpu/cpuvec.hh"

#include "../mpimat.hh"


namespace fml
{
namespace linalg
{
  /**
    @brief Computes the 1 matrix norm, the maximum absolute column sum.
    
    @param[in] x Input data matrix, replaced by its LU factorization.
    
    @return Returns the norm.
    
    @allocs Allocates temporary storage to store the col sums.
    
    @except If an allocation is triggered and fails, a `bad_alloc` exception
    will be thrown.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  REAL norm_1(const mpimat<REAL> &x)
  {
    const len_t n = x.nrows();
    const len_t m_local = x.nrows_local();
    const len_t n_local = x.ncols_local();
    const REAL *x_d = x.data_ptr();
    const grid g = x.get_grid();
    
    REAL norm = 0;
    
    cpuvec<REAL> macs(n);
    macs.fill_zero();
    REAL *macs_d = macs.data_ptr();
    
    for (len_t j=0; j<n_local; j++)
    {
      for (len_t i=0; i<m_local; i++)
        macs_d[j] += fabs(x_d[i + m_local*j]);
    }
    
    g.allreduce(n, 1, macs_d, 'C');
    
    for (len_t j=0; j<n; j++)
    {
      if (norm < macs_d[j])
        norm = macs_d[j];
    }
    
    g.allreduce(1, 1, &norm, 'R', BLACS_MAX);
    
    return norm;
  }
  
  
  
  /**
    @brief Computes the infinity matrix norm, the maximum absolute row sum.
    
    @param[in] x Input data matrix, replaced by its LU factorization.
    
    @return Returns the norm.
    
    @allocs Allocates temporary storage to store the row sums.
    
    @except If an allocation is triggered and fails, a `bad_alloc` exception
    will be thrown.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  REAL norm_I(const mpimat<REAL> &x)
  {
    const len_t m = x.nrows();
    const len_t m_local = x.nrows_local();
    const len_t n_local = x.ncols_local();
    const REAL *x_d = x.data_ptr();
    const grid g = x.get_grid();
    
    REAL norm = 0;
    
    cpuvec<REAL> mars(m);
    mars.fill_zero();
    REAL *mars_d = mars.data_ptr();
    
    for (len_t j=0; j<n_local; j++)
    {
      for (len_t i=0; i<m_local; i++)
        mars_d[i] += fabs(x_d[i + m_local*j]);
    }
    
    g.allreduce(m, 1, mars_d, 'R');
    
    for (len_t i=0; i<m; i++)
    {
      if (norm < mars_d[i])
        norm = mars_d[i];
    }
    
    g.allreduce(1, 1, &norm, 'C', BLACS_MAX);
    
    return norm;
  }
  
  
  
  /**
    @brief Computes the Frobenius/Euclidean matrix norm.
    
    @param[in] x Input data matrix, replaced by its LU factorization.
    
    @return Returns the norm.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  REAL norm_F(const mpimat<REAL> &x)
  {
    const len_t m_local = x.nrows_local();
    const len_t n_local = x.ncols_local();
    const REAL *x_d = x.data_ptr();
    
    REAL norm = 0;
    
    for (len_t j=0; j<n_local; j++)
    {
      for (len_t i=0; i<m_local; i++)
        norm += x_d[i + m_local*j] * x_d[i + m_local*j];
    }
    
    x.get_grid().allreduce(1, 1, &norm, 'A');
    
    return sqrt(norm);
  }
  
  
  
  /**
    @brief Computes the maximum modulus matrix norm.
    
    @param[in] x Input data matrix, replaced by its LU factorization.
    
    @return Returns the norm.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  REAL norm_M(const mpimat<REAL> &x)
  {
    const len_t m_local = x.nrows_local();
    const len_t n_local = x.ncols_local();
    const REAL *x_d = x.data_ptr();
    
    REAL norm = 0;
    
    for (len_t j=0; j<n_local; j++)
    {
      for (len_t i=0; i<m_local; i++)
      {
        REAL tmp = fabs(x_d[i + m_local*j]);
        if (tmp > norm)
          norm = tmp;
      }
    }
    
    x.get_grid().allreduce(1, 1, &norm, 'A', BLACS_MAX);
    
    return norm;
  }
}
}


#endif
