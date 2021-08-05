// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_CPU_DIMOPS_H
#define FML_CPU_DIMOPS_H
#pragma once


#include <cmath>

#include "../_internals/omp.hh"

#include "cpumat.hh"
#include "cpuvec.hh"


namespace fml
{
/// @brief Row/column operations.
namespace dimops
{
  namespace internals
  {
    template <typename REAL>
    static inline void col_sum(const len_t j, const len_t m, const REAL *x, REAL &sum)
    {
      sum = 0;
        
      #pragma omp simd reduction(+:sum)
      for (len_t i=0; i<m; i++)
        sum += x[i + m*j];
    }
    
    template <typename REAL>
    static inline void col_mean(const len_t j, const len_t m, const REAL *x, REAL &mean)
    {
      mean = 0;
      col_sum(j, m, x, mean);
      mean /= (REAL) m;
    }
    
    template <typename REAL>
    static inline void col_mean_and_var(const len_t j, const len_t m, const REAL *x, REAL &mean, REAL &var)
    {
      mean = 0;
      var = 0;
        
      for (len_t i=0; i<m; i++)
      {
        REAL dt = x[i + m*j] - mean;
        mean += dt/((REAL) i+1);
        var += dt * (x[i + m*j] - mean);
      }
      
      var = sqrt(var / ((REAL) m-1));
    }
  }
  
  
  
  /**
    @brief Compute the row sums.
    
    @param[in] x Input data.
    @param[out] s Row sums.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  void rowsums(const cpumat<REAL> &x, cpuvec<REAL> &s)
  {
    const REAL *x_d = x.data_ptr();
    const len_t m = x.nrows();
    const len_t n = x.ncols();
    
    s.resize(m);
    s.fill_zero();
    REAL *s_d = s.data_ptr();
    
    for (len_t j=0; j<n; j++)
    {
      #pragma omp for simd
      for (len_t i=0; i<m; i++)
        s_d[i] += x_d[i + m*j];
    }
  }
  
  
  
  /**
    @brief Compute the row means.
    
    @param[in] x Input data.
    @param[out] s Row means.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  void rowmeans(const cpumat<REAL> &x, cpuvec<REAL> &s)
  {
    rowsums(x, s);
    
    const len_t m = x.nrows();
    const len_t n = x.ncols();
    REAL *s_d = s.data_ptr();
    
    #pragma omp for simd
    for (len_t i=0; i<m; i++)
      s_d[i] /= (REAL) n;
  }
  
  
  
  /**
    @brief Compute the column sums.
    
    @param[in] x Input data.
    @param[out] s Column sums.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  void colsums(const cpumat<REAL> &x, cpuvec<REAL> &s)
  {
    const REAL *x_d = x.data_ptr();
    const len_t m = x.nrows();
    const len_t n = x.ncols();
    
    s.resize(n);
    s.fill_zero();
    REAL *s_d = s.data_ptr();
    
    #pragma omp parallel for if(m*n > fml::omp::OMP_MIN_SIZE)
    for (len_t j=0; j<n; j++)
      internals::col_sum(j, m, x_d, s_d[j]);
  }
  
  
  
  /**
    @brief Compute the column means.
    
    @param[in] x Input data.
    @param[out] s Column means.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  void colmeans(const cpumat<REAL> &x, cpuvec<REAL> &s)
  {
    const REAL *x_d = x.data_ptr();
    const len_t m = x.nrows();
    const len_t n = x.ncols();
    
    s.resize(n);
    s.fill_zero();
    REAL *s_d = s.data_ptr();
    
    #pragma omp parallel for if(m*n > fml::omp::OMP_MIN_SIZE)
    for (len_t j=0; j<n; j++)
      internals::col_mean(j, m, x_d, s_d[j]);
  }
  
  
  
  enum sweep_op
  {
    /**
      TODO
    */
    SWEEP_ADD, SWEEP_SUB, SWEEP_MUL, SWEEP_DIV
  };
  
  template <typename REAL>
  static inline void rowsweep(cpumat<REAL> &x, const cpuvec<REAL> &s, const sweep_op op)
  {
    const len_t m = x.nrows();
    const len_t n = x.ncols();
    
    if (s.size() != m)
      throw std::runtime_error("non-conformal arguments");
    
    REAL *x_d = x.data_ptr();
    const REAL *s_d = s.data_ptr();
    
    if (op == SWEEP_ADD)
    {
      #pragma omp parallel for if(m*n > fml::omp::OMP_MIN_SIZE)
      for (len_t j=0; j<n; j++)
      {
        #pragma omp simd
        for (len_t i=0; i<m; i++)
          x_d[i + m*j] += s_d[i];
      }
    }
    else if (op == SWEEP_SUB)
    {
      #pragma omp parallel for if(m*n > fml::omp::OMP_MIN_SIZE)
      for (len_t j=0; j<n; j++)
      {
        #pragma omp simd
        for (len_t i=0; i<m; i++)
          x_d[i + m*j] -= s_d[i];
      }
    }
    else if (op == SWEEP_MUL)
    {
      #pragma omp parallel for if(m*n > fml::omp::OMP_MIN_SIZE)
      for (len_t j=0; j<n; j++)
      {
        #pragma omp simd
        for (len_t i=0; i<m; i++)
          x_d[i + m*j] *= s_d[i];
      }
    }
    else if (op == SWEEP_DIV)
    {
      #pragma omp parallel for if(m*n > fml::omp::OMP_MIN_SIZE)
      for (len_t j=0; j<n; j++)
      {
        #pragma omp simd
        for (len_t i=0; i<m; i++)
          x_d[i + m*j] /= s_d[i];
      }
    }
  }
  
  
  
  template <typename REAL>
  static inline void colsweep(cpumat<REAL> &x, const cpuvec<REAL> &s, const sweep_op op)
  {
    const len_t m = x.nrows();
    const len_t n = x.ncols();
    
    if (s.size() != n)
      throw std::runtime_error("non-conformal arguments");
    
    REAL *x_d = x.data_ptr();
    const REAL *s_d = s.data_ptr();
    
    if (op == SWEEP_ADD)
    {
      #pragma omp parallel for if(m*n > fml::omp::OMP_MIN_SIZE)
      for (len_t j=0; j<n; j++)
      {
        #pragma omp simd
        for (len_t i=0; i<m; i++)
          x_d[i + m*j] += s_d[j];
      }
    }
    else if (op == SWEEP_SUB)
    {
      #pragma omp parallel for if(m*n > fml::omp::OMP_MIN_SIZE)
      for (len_t j=0; j<n; j++)
      {
        #pragma omp simd
        for (len_t i=0; i<m; i++)
          x_d[i + m*j] -= s_d[j];
      }
    }
    else if (op == SWEEP_MUL)
    {
      #pragma omp parallel for if(m*n > fml::omp::OMP_MIN_SIZE)
      for (len_t j=0; j<n; j++)
      {
        #pragma omp simd
        for (len_t i=0; i<m; i++)
          x_d[i + m*j] *= s_d[j];
      }
    }
    else if (op == SWEEP_DIV)
    {
      #pragma omp parallel for if(m*n > fml::omp::OMP_MIN_SIZE)
      for (len_t j=0; j<n; j++)
      {
        #pragma omp simd
        for (len_t i=0; i<m; i++)
          x_d[i + m*j] /= s_d[j];
      }
    }
  }
  
  
  
  namespace internals
  {
    template <typename REAL>
    static inline void center(cpumat<REAL> &x)
    {
      REAL *x_d = x.data_ptr();
      const len_t m = x.nrows();
      const len_t n = x.ncols();
      
      #pragma omp parallel for if(m*n > fml::omp::OMP_MIN_SIZE)
      for (len_t j=0; j<n; j++)
      {
        REAL mean = 0;
        internals::col_mean(j, m, x_d, mean);
        
        #pragma omp simd
        for (len_t i=0; i<m; i++)
          x_d[i + m*j] -= mean;
      }
    }
    
    template <typename REAL>
    static inline void scale(cpumat<REAL> &x)
    {
      REAL *x_d = x.data_ptr();
      const len_t m = x.nrows();
      const len_t n = x.ncols();
      
      #pragma omp parallel for if(m*n > fml::omp::OMP_MIN_SIZE)
      for (len_t j=0; j<n; j++)
      {
        REAL mean = 0;
        REAL var = 0;
        internals::col_mean_and_var(j, m, x_d, mean, var);
        
        #pragma omp simd
        for (len_t i=0; i<m; i++)
          x_d[i + m*j] /= var;
      }
    }
    
    template <typename REAL>
    static inline void center_and_scale(cpumat<REAL> &x)
    {
      REAL *x_d = x.data_ptr();
      const len_t m = x.nrows();
      const len_t n = x.ncols();
      
      #pragma omp parallel for if(m*n > fml::omp::OMP_MIN_SIZE)
      for (len_t j=0; j<n; j++)
      {
        REAL mean = 0;
        REAL var = 0;
        internals::col_mean_and_var(j, m, x_d, mean, var);
        
        #pragma omp simd
        for (len_t i=0; i<m; i++)
          x_d[i + m*j] = (x_d[i + m*j] - mean) / var;
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
  void scale(const bool rm_mean, const bool rm_sd, cpumat<REAL> &x)
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
