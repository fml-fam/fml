// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_CPU_LINALG_LINALG_NORM_H
#define FML_CPU_LINALG_LINALG_NORM_H
#pragma once


#include <cmath>
#include <stdexcept>

#include "../../_internals/linalgutils.hh"
#include "../../_internals/omp.hh"

#include "../cpumat.hh"
#include "../cpuvec.hh"

#include "lapack.hh"
#include "linalg_factorizations.hh"


namespace fml
{
/// @brief Linear algebra functions.
namespace linalg
{
  /**
    @brief Computes the 1 matrix norm, the maximum absolute column sum.
    
    @param[in] x Input data matrix.
    
    @return Returns the norm.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  REAL norm_1(const cpumat<REAL> &x)
  {
    const len_t m = x.nrows();
    const len_t n = x.ncols();
    const REAL *x_d = x.data_ptr();
    
    REAL norm = 0;
    
    #pragma omp parallel for if(m*n > fml::omp::OMP_MIN_SIZE) reduction(max:norm)
    for (len_t j=0; j<n; j++)
    {
      REAL tmp = 0;
      
      #pragma omp simd reduction(+:tmp)
      for (len_t i=0; i<m; i++)
        tmp += fabs(x_d[i + m*j]);
      
      if (tmp > norm)
        norm = tmp;
    }
    
    return norm;
  }
  
  
  
  /**
    @brief Computes the infinity matrix norm, the maximum absolute row sum.
    
    @param[in] x Input data matrix.
    
    @return Returns the norm.
    
    @allocs Allocates temporary storage to store the row sums.
    
    @except If an allocation is triggered and fails, a `bad_alloc` exception
    will be thrown.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  REAL norm_I(const cpumat<REAL> &x)
  {
    const len_t m = x.nrows();
    const len_t n = x.ncols();
    const REAL *x_d = x.data_ptr();
    
    REAL norm = 0;
    
    cpuvec<REAL> mars(m);
    mars.fill_zero();
    REAL *mars_d = mars.data_ptr();
    
    for (len_t j=0; j<n; j++)
    {
      for (len_t i=0; i<m; i++)
        mars_d[i] += fabs(x_d[i + m*j]);
    }
    
    for (len_t i=0; i<m; i++)
    {
      if (mars_d[i] > norm)
        norm = mars_d[i];
    }
    
    return norm;
  }
  
  
  
  /**
    @brief Computes the Frobenius/Euclidean matrix norm.
    
    @param[in] x Input data matrix.
    
    @return Returns the norm.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  REAL norm_F(const cpumat<REAL> &x)
  {
    const len_t m = x.nrows();
    const len_t n = x.ncols();
    const REAL *x_d = x.data_ptr();
    
    REAL scale = 0;
    REAL sumsq = 1;
    
    for (len_t j=0; j<n; j++)
      lapack::lassq(m, x_d + m*j, 1, &scale, &sumsq);
    
    return scale * sqrtf(sumsq);
  }
  
  
  
  /**
    @brief Computes the maximum modulus matrix norm.
    
    @param[in] x Input data matrix.
    
    @return Returns the norm.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  REAL norm_M(const cpumat<REAL> &x)
  {
    const len_t m = x.nrows();
    const len_t n = x.ncols();
    const REAL *x_d = x.data_ptr();
    
    REAL norm = 0;
    
    #pragma omp parallel for if(m*n > fml::omp::OMP_MIN_SIZE) reduction(max:norm)
    for (len_t j=0; j<n; j++)
    {
      for (len_t i=0; i<m; i++)
      {
        REAL tmp = fabs(x_d[i + m*j]);
        if (tmp > norm)
          norm = tmp;
      }
    }
    
    return norm;
  }
  
  
  
  /**
    @brief Computes the 2/spectral matrix norm.
    
    @details Returns the largest singular value.
    
    @param[inout] x Input data matrix. Values are overwritten.
    
    @return Returns the norm.
    
    @impl Uses `linalg::svd()`.
    
    @allocs Allocates temporary storage to compute the singular values.
    
    @except If an allocation is triggered and fails, a `bad_alloc` exception
    will be thrown.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  REAL norm_2(cpumat<REAL> &x)
  {
    REAL ret;
    cpuvec<REAL> s;
    svd(x, s);
    ret = s.get(0);
    
    return ret;
  }
  
  
  
  namespace
  {
    template <typename REAL>
    REAL cond_square_internals(const char norm, cpumat<REAL> &x)
    {
      const len_t n = x.nrows();
      
      REAL ret;
      int info;
      
      lu(x);
      
      cpuvec<REAL> work(4*n);
      cpuvec<REAL> work2(n);
      REAL xnorm = norm_1(x);
      lapack::gecon(norm, n, x.data_ptr(), n, xnorm, &ret, work.data_ptr(),
        work2.data_ptr(), &info);
      
      fml::linalgutils::check_info(info, "gecon");
      
      return ((REAL)1)/ret;
    }
    
    template <typename REAL>
    REAL cond_nonsquare_internals(const char norm, cpumat<REAL> &x)
    {
      const len_t m = x.nrows();
      const len_t n = x.ncols();
      
      REAL ret;
      int info;
      
      cpuvec<REAL> aux;
      
      if (m > n)
      {
        cpumat<REAL> R;
        qr(false, x, aux);
        qr_R(x, R);
        
        aux.resize(R.nrows());
        lapack::trcon(norm, 'U', 'N', n, R.data_ptr(), n, &ret,
          x.data_ptr(), aux.data_ptr(), &info);
      }
      else
      {
        cpumat<REAL> L;
        lq(x, aux);
        lq_L(x, L);
        
        aux.resize(L.nrows());
        lapack::trcon(norm, 'L', 'N', m, L.data_ptr(), m, &ret,
          x.data_ptr(), aux.data_ptr(), &info);
      }
      
      fml::linalgutils::check_info(info, "trcon");
      
      return ((REAL)1)/ret;
    }
  }
  
  /**
    @brief Estimates the condition number under the 1-norm.
    
    @param[in] x Input data matrix.
    
    @param[inout] x Input data matrix. The data is overwritten.
    
    @impl Computes L or R (whichever is smaller) and the LAPACK function
    `Xtrcon()` if the input is not square, and `Xgecon()` on the LU of the input
    otherwise.
    
    @allocs Allocates temporary storage to compute the QR/LQ/LU, as well as
    workspace arrays for the LAPACK condition number function.
    
    @except If an allocation is triggered and fails, a `bad_alloc` exception
    will be thrown.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  REAL cond_1(cpumat<REAL> &x)
  {
    if (x.is_square())
      return cond_square_internals('1', x);
    else
      return cond_nonsquare_internals('1', x);
  }
  
  
  
  /**
    @brief Estimates the condition number under the infinity norm.
    
    @param[in] x Input data matrix.
    
    @param[inout] x Input data matrix. The data is overwritten.
    
    @impl Computes L or R (whichever is smaller) and the LAPACK function
    `Xtrcon()` if the input is not square, and `Xgecon()` on the LU of the input
    otherwise.
    
    @allocs Allocates temporary storage to compute the QR/LQ/LU, as well as
    workspace arrays for the LAPACK condition number function.
    
    @except If an allocation is triggered and fails, a `bad_alloc` exception
    will be thrown.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  REAL cond_I(cpumat<REAL> &x)
  {
    if (x.is_square())
      return cond_square_internals('I', x);
    else
      return cond_nonsquare_internals('I', x);
  }
  
  
  
  /**
    @brief Estimates the condition number under the 2 norm.
    
    @param[in] x Input data matrix.
    
    @param[inout] x Input data matrix. The data is overwritten.
    
    @impl Uses `linalg::svd()`.
    
    @allocs Allocates temporary storage to compute the QR/LQ/LU, as well as
    workspace arrays for the LAPACK condition number function.
    
    @except If an allocation is triggered and fails, a `bad_alloc` exception
    will be thrown.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  REAL cond_2(cpumat<REAL> &x)
  {
    cpuvec<REAL> s;
    svd(x, s);
    
    REAL *s_d = s.data_ptr();
    
    REAL max = s_d[0];
    REAL min = s_d[0];
    
    #pragma omp parallel for if(s.size() > fml::omp::OMP_MIN_SIZE) reduction(max:max) reduction(min:min)
    for (len_t i=1; i<s.size(); i++)
    {
      if (s_d[i] > max)
        max = s_d[i];
      if (s_d[i] > 0 && s_d[i] < min)
        min = s_d[i];
    }
    
    if (max == 0)
      return 0;
    else
      return max/min;
  }
}
}


#endif
