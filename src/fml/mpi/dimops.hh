// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_MPI_DIMOPS_H
#define FML_MPI_DIMOPS_H
#pragma once


#include <cmath>

#include "../cpu/internals/vecops.hh"
#include "internals/bcutils.hh"
#include "mpimat.hh"


namespace fml
{
/// @brief Row/column operations.
namespace dimops
{
  /**
    @brief Compute the row sums.
    
    @param[in] x Input data.
    @param[out] s Row sums.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  void rowsums(const mpimat<REAL> &x, cpuvec<REAL> &s)
  {
    const REAL *x_d = x.data_ptr();
    const len_t m = x.nrows();
    const len_t m_local = x.nrows_local();
    const len_t n_local = x.ncols_local();
    const len_t mb = x.bf_rows();
    
    const grid g = x.get_grid();
    
    s.resize(m);
    s.fill_zero();
    REAL *s_d = s.data_ptr();
    
    for (len_t j=0; j<n_local; j++)
    {
      #pragma omp for simd
      for (len_t i=0; i<m_local; i++)
      {
        const len_t gi = fml::bcutils::l2g(i, mb, g.nprow(), g.myrow());
        s_d[gi] += x_d[i + m_local*j];
      }
    }
    
    x.get_grid().allreduce(m, 1, s_d, 'A');
  }
  
  
  
  /**
    @brief Compute the row means.
    
    @param[in] x Input data.
    @param[out] s Row means.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  void rowmeans(const mpimat<REAL> &x, cpuvec<REAL> &s)
  {
    rowsums(x, s);
    fml::vecops::cpu::sweep_mul((REAL) 1.0/x.ncols(), x.nrows(), s.data_ptr());
  }
  
  
  
  /**
    @brief Compute the column sums.
    
    @param[in] x Input data.
    @param[out] s Column sums.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  void colsums(const mpimat<REAL> &x, cpuvec<REAL> &s)
  {
    const REAL *x_d = x.data_ptr();
    const len_t n = x.ncols();
    const len_t m_local = x.nrows_local();
    const len_t n_local = x.ncols_local();
    const len_t nb = x.bf_cols();
    
    const grid g = x.get_grid();
    
    s.resize(n);
    s.fill_zero();
    REAL *s_d = s.data_ptr();
    
    for (len_t j=0; j<n_local; j++)
    {
      const len_t gj = fml::bcutils::l2g(j, nb, g.npcol(), g.mycol());
      fml::vecops::cpu::sum(m_local, x_d + m_local*j, s_d[gj]);
    }
    
    x.get_grid().allreduce(n, 1, s_d, 'A');
  }
  
  
  
  /**
    @brief Compute the column means.
    
    @param[in] x Input data.
    @param[out] s Column means.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  void colmeans(const mpimat<REAL> &x, cpuvec<REAL> &s)
  {
    colsums(x, s);
    fml::vecops::cpu::sweep_mul((REAL) 1.0/x.nrows(), x.ncols(), s.data_ptr());
  }
  
  
  
  namespace internals
  {
    template <typename REAL>
    static inline void col_mean(const grid g, const len_t j, const len_t m, const len_t m_local, const REAL *x, REAL &mean)
    {
      mean = 0;
      fml::vecops::cpu::sum(m_local, x + m_local*j, mean);
      g.allreduce(1, 1, &mean, 'C');
      mean /= (REAL) m;
    }
    
    template <typename REAL>
    static inline void col_var(const grid g, const len_t j, const len_t m, const len_t m_local, const REAL *x, const REAL &mean, REAL *work, REAL &var)
    {
      work[0] = 0;
      work[1] = 0;
      
      for (len_t i = 0; i<m_local; i++)
      {
        REAL diff = x[i + m_local*j] - mean;
        work[0] += diff*diff;
        work[1] += diff;
      }
      
      g.allreduce(2, 1, work, 'C');
      
      var = (work[0] - work[1]*work[1]/m) / (m-1);
    }
    
    
    
    template <typename REAL>
    static inline void center(mpimat<REAL> &x)
    {
      REAL *x_d = x.data_ptr();
      const len_t m = x.nrows();
      const len_t m_local = x.nrows_local();
      const len_t n_local = x.ncols_local();
      
      grid g = x.get_grid();
      
      for (len_t j=0; j<n_local; j++)
      {
        REAL mean;
        col_mean(g, j, m, m_local, x_d, mean);
        fml::vecops::cpu::sweep_add(-mean, m_local, x_d + m_local*j);
      }
    }
    
    template <typename REAL>
    static inline void scale(mpimat<REAL> &x)
    {
      REAL *x_d = x.data_ptr();
      const len_t m = x.nrows();
      const len_t m_local = x.nrows_local();
      const len_t n_local = x.ncols_local();
      
      grid g = x.get_grid();
      
      REAL work[2];
      
      for (len_t j=0; j<n_local; j++)
      {
        REAL mean;
        col_mean(g, j, m, m_local, x_d, mean);
        
        REAL var;
        col_var(g, j, m, m_local, x_d, mean, work, var);
        var = (REAL)1.0/sqrt(var);
        fml::vecops::cpu::sweep_mul(var, m_local, x_d + m_local*j);
      }
    }
    
    template <typename REAL>
    static inline void center_and_scale(mpimat<REAL> &x)
    {
      REAL *x_d = x.data_ptr();
      const len_t m = x.nrows();
      const len_t m_local = x.nrows_local();
      const len_t n_local = x.ncols_local();
      
      grid g = x.get_grid();
      
      REAL work[2];
      
      for (len_t j=0; j<n_local; j++)
      {
        REAL mean;
        col_mean(g, j, m, m_local, x_d, mean);
        
        REAL var;
        col_var(g, j, m, m_local, x_d, mean, work, var);
        
        #pragma omp for simd
        for (len_t i=0; i<m_local; i++)
          x_d[i + m_local*j] = (x_d[i + m_local*j] - mean) / sqrt(var);
      }
    }
  }
  
  
  
  /**
    @brief Remove the mean and/or the sd from a matrix.
    
    @param[in] rm_mean Remove the column means?
    @param[in] rm_sd Remove the column sds?
    @param[inout] x Data to center/scale.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  void scale(const bool rm_mean, const bool rm_sd, mpimat<REAL> &x)
  {
    if (rm_mean && rm_sd)
      internals::center_and_scale(x);
    else if (rm_mean)
      internals::center(x);
    else if (rm_sd)
      internals::scale(x);
  }
}
}


#endif
