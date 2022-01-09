// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_MPI_LINALG_NORM_H
#define FML_MPI_LINALG_NORM_H
#pragma once


#include "../../cpu/cpuvec.hh"

#include "../mpimat.hh"

#include "lu.hh"
#include "qr.hh"
#include "svd.hh"


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
    
    #pragma omp parallel for if(m_local*n_local > fml::omp::OMP_MIN_SIZE)
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
  
  
  
  /**
    @brief Computes the 2/spectral matrix norm.
    
    @details Returns the largest singular value.
    
    @param[inout] x Input data matrix. Values are overwritten.
    
    @return Returns the norm.
    
    @allocs Allocates temporary storage to compute the singular values.
    
    @except If an allocation is triggered and fails, a `bad_alloc` exception
    will be thrown.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  REAL norm_2(mpimat<REAL> &x)
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
    REAL cond_square_internals(const char norm, mpimat<REAL> &x)
    {
      const len_t n = x.nrows();
      
      REAL ret;
      int info;
      
      REAL xnorm = norm_1(x);
      
      lu(x);
      
      REAL tmp;
      int liwork;
      scalapack::gecon(norm, n, x.data_ptr(), x.desc_ptr(), xnorm,
        &ret, &tmp, -1, &liwork, -1, &info);
      
      int lwork = (int) tmp;
      cpuvec<REAL> work(lwork);
      cpuvec<int> iwork(liwork);
      
      scalapack::gecon(norm, n, x.data_ptr(), x.desc_ptr(), xnorm,
        &ret, work.data_ptr(), lwork, iwork.data_ptr(), liwork, &info);
      
      fml::linalgutils::check_info(info, "gecon");
      
      return ((REAL)1)/ret;
    }
    
    template <typename REAL>
    REAL cond_nonsquare_internals(const char norm, mpimat<REAL> &x)
    {
      const len_t m = x.nrows();
      const len_t n = x.ncols();
      
      REAL ret;
      int info;
      
      cpuvec<REAL> aux;
      
      if (m > n)
      {
        mpimat<REAL> R(x.get_grid(), x.bf_rows(), x.bf_cols());
        qr(false, x, aux);
        qr_R(x, R);
        
        REAL tmp;
        int liwork;
        scalapack::trcon(norm, 'U', 'N', n, R.data_ptr(), R.desc_ptr(), &ret,
          &tmp, -1, &liwork, -1, &info);
        
        int lwork = (int) tmp;
        cpuvec<REAL> work(lwork);
        cpuvec<int> iwork(liwork);
        
        scalapack::trcon(norm, 'U', 'N', n, R.data_ptr(), R.desc_ptr(), &ret,
          x.data_ptr(), lwork, iwork.data_ptr(), iwork.size(), &info);
      }
      else
      {
        mpimat<REAL> L(x.get_grid(), x.bf_rows(), x.bf_cols());
        lq(x, aux);
        lq_L(x, L);
        
        REAL tmp;
        int liwork;
        scalapack::trcon(norm, 'L', 'N', m, L.data_ptr(), L.desc_ptr(), &ret,
          &tmp, -1, &liwork, -1, &info);
        
        int lwork = (int) tmp;
        cpuvec<REAL> work(lwork);
        cpuvec<int> iwork(liwork);
        
        scalapack::trcon(norm, 'L', 'N', m, L.data_ptr(), L.desc_ptr(), &ret,
          x.data_ptr(), x.nrows()*x.ncols(), iwork.data_ptr(), iwork.size(), &info);
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
  REAL cond_1(mpimat<REAL> &x)
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
  REAL cond_I(mpimat<REAL> &x)
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
  REAL cond_2(mpimat<REAL> &x)
  {
    cpuvec<REAL> s;
    svd(x, s);
    
    REAL *s_d = s.data_ptr();
    
    REAL max = s_d[0];
    REAL min = s_d[0];
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
