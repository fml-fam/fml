// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_GPU_DIMOPS_H
#define FML_GPU_DIMOPS_H
#pragma once


#include "internals/kernelfuns.hh"

#include "gpumat.hh"
#include "gpuvec.hh"
#include "linalg.hh"


/// @brief Row/column operations.
namespace dimops
{
  namespace internals
  {
    template <typename REAL>
    __global__ void kernel_sweep_add(const len_t m, const len_t n,
      REAL *x, const REAL *s, const REAL sgn)
    {
      int i = blockDim.x*blockIdx.x + threadIdx.x;
      int j = blockDim.y*blockIdx.y + threadIdx.y;
      
      if (i < m && j < n)
        x[i + m*j] += sgn * s[j];
    }
    
    template <typename REAL>
    static inline void sweep_add(gpumat<REAL> &x, const gpuvec<REAL> &s)
    {
      kernel_sweep_add<<<x.get_griddim(), x.get_blockdim()>>>(x.nrows(),
        x.ncols(), x.data_ptr(), s.data_ptr(), (REAL) 1.0);
    }
    
    template <typename REAL>
    static inline void sweep_sub(gpumat<REAL> &x, const gpuvec<REAL> &s)
    {
      kernel_sweep_add<<<x.get_griddim(), x.get_blockdim()>>>(x.nrows(),
        x.ncols(), x.data_ptr(), s.data_ptr(), (REAL) -1.0);
    }
    
    template <typename REAL>
    __global__ void kernel_sweep_mul(const len_t m, const len_t n,
      REAL *x, const REAL *s)
    {
      int i = blockDim.x*blockIdx.x + threadIdx.x;
      int j = blockDim.y*blockIdx.y + threadIdx.y;
      
      if (i < m && j < n)
        x[i + m*j] *= s[j];
    }
    
    template <typename REAL>
    static inline void sweep_mul(gpumat<REAL> &x, const gpuvec<REAL> &s)
    {
      kernel_sweep_mul<<<x.get_griddim(), x.get_blockdim()>>>(x.nrows(),
        x.ncols(), x.data_ptr(), s.data_ptr());
    }
    
    template <typename REAL>
    __global__ void kernel_sweep_div(const len_t m, const len_t n,
      REAL *x, const REAL *s)
    {
      int i = blockDim.x*blockIdx.x + threadIdx.x;
      int j = blockDim.y*blockIdx.y + threadIdx.y;
      
      if (i < m && j < n)
        x[i + m*j] /= s[j];
    }
    
    template <typename REAL>
    static inline void sweep_div(gpumat<REAL> &x, const gpuvec<REAL> &s)
    {
      kernel_sweep_div<<<x.get_griddim(), x.get_blockdim()>>>(x.nrows(),
        x.ncols(), x.data_ptr(), s.data_ptr());
    }
    
    
    
    template <typename REAL>
    __global__ void kernel_colsdevs(const len_t m, const len_t n,
      const REAL *x, const REAL *means, REAL *sdevs)
    {
      int i = blockDim.x*blockIdx.x + threadIdx.x;
      int j = blockDim.y*blockIdx.y + threadIdx.y;
      
      if (i < m && j < n)
      {
        REAL tmp = (x[i + m*j] - means[j]) * (x[i + m*j] - means[j]) / (m-1);
        atomicAdd(sdevs+j, tmp);
      }
    }
    
    template <typename REAL>
    static inline void colsdevs(const gpumat<REAL> &x,
      const gpuvec<REAL> &means, gpuvec<REAL> &sdevs)
    {
      sdevs.fill_zero();
      kernel_colsdevs<<<x.get_griddim(), x.get_blockdim()>>>(x.nrows(),
        x.ncols(), x.data_ptr(), means.data_ptr(), sdevs.data_ptr());
      fml::kernelfuns::kernel_root_abs<<<sdevs.get_griddim(),
        sdevs.get_blockdim()>>>(sdevs.size(), sdevs.data_ptr());
    }
  }
  
  
  
  /**
    @brief Compute the row sums.
    
    @param[in] x Input data.
    @param[out] s Row sums.
    
    @impl Uses `linalg::matmult()` on a vector of ones.
    
    @allocs If the output is inappropriately sized, it will automatically be
    re-allocated. Additionally, some temporary work storage is needed.
    
    @except If a (re-)allocation is triggered and fails, a `bad_alloc`
    exception will be thrown.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  void rowsums(const gpumat<REAL> &x, gpuvec<REAL> &s)
  {
    linalg::err::check_card(x, s);
    const len_t m = x.nrows();
    const len_t n = x.ncols();
    
    s.resize(m);
    s.fill_zero();
    gpumat<REAL> s_mat(s.get_card(), s.data_ptr(), m, 1, false);
    
    gpumat<REAL> ones(s.get_card(), n, 1);
    ones.fill_val(1);
    
    linalg::matmult(false, false, (REAL) 1.0, x, ones, s_mat);
  }
  
  
  
  /**
    @brief Compute the row means.
    
    @param[in] x Input data.
    @param[out] s Row means.
    
    @impl Uses `linalg::matmult()` on a vector of ones.
    
    @allocs If the output is inappropriately sized, it will automatically be
    re-allocated. Additionally, some temporary work storage is needed.
    
    @except If a (re-)allocation is triggered and fails, a `bad_alloc`
    exception will be thrown.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  void rowmeans(const gpumat<REAL> &x, gpuvec<REAL> &s)
  {
    rowsums(x, s);
    s.scale((REAL) 1.0/x.ncols());
  }
  
  
  
  /**
    @brief Compute the column sums.
    
    @param[in] x Input data.
    @param[out] s Column sums.
    
    @impl Uses `linalg::matmult()` on a vector of ones.
    
    @allocs If the output is inappropriately sized, it will automatically be
    re-allocated. Additionally, some temporary work storage is needed.
    
    @except If a (re-)allocation is triggered and fails, a `bad_alloc`
    exception will be thrown.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  void colsums(const gpumat<REAL> &x, gpuvec<REAL> &s)
  {
    linalg::err::check_card(x, s);
    const len_t m = x.nrows();
    const len_t n = x.ncols();
    
    s.resize(n);
    s.fill_zero();
    gpumat<REAL> s_mat(s.get_card(), s.data_ptr(), 1, n, false);
    
    gpumat<REAL> ones(s.get_card(), 1, m);
    ones.fill_val(1);
    
    linalg::matmult(false, false, (REAL) 1.0, ones, x, s_mat);
  }
  
  
  
  /**
    @brief Compute the column means.
    
    @param[in] x Input data.
    @param[out] s Column means.
    
    @impl Uses `linalg::matmult()` on a vector of ones.
    
    @allocs If the output is inappropriately sized, it will automatically be
    re-allocated. Additionally, some temporary work storage is needed.
    
    @except If a (re-)allocation is triggered and fails, a `bad_alloc`
    exception will be thrown.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  void colmeans(const gpumat<REAL> &x, gpuvec<REAL> &s)
  {
    colsums(x, s);
    s.scale((REAL) 1.0/x.nrows());
  }
  
  
  
  /**
    @brief Remove the mean and/or the sd from a matrix.
    
    @param[in] rm_mean Remove the column means?
    @param[in] rm_sd Remove the column sds?
    @param[inout] x Data to center/scale.
    
    @allocs Some temporary work storage is needed to store vectors of means
    and/or standard deviations, depending on what is requested.
    
    @except If an allocation fails, a `bad_alloc` exception will be thrown.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  void scale(const bool rm_mean, const bool rm_sd, gpumat<REAL> &x)
  {
    const len_t n = x.ncols();
    
    if (rm_mean && rm_sd)
    {
      gpuvec<REAL> means(x.get_card(), n);
      gpuvec<REAL> sdevs(x.get_card(), n);
      colmeans(x, means);
      internals::colsdevs(x, means, sdevs);
      
      means.print();
      sdevs.print();
      
      internals::sweep_sub(x, means);
      internals::sweep_div(x, sdevs);
    }
    else if (rm_mean)
    {
      gpuvec<REAL> means(x.get_card());
      colmeans(x, means);
      
      internals::sweep_sub(x, means);
    }
    else if (rm_sd)
    {
      gpuvec<REAL> means(x.get_card());
      gpuvec<REAL> sdevs(x.get_card(), n);
      colmeans(x, means);
      internals::colsdevs(x, means, sdevs);
      
      internals::sweep_div(x, sdevs);
    }
  }
}


#endif
