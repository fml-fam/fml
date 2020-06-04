// This file is part of arraytools which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef ARRAYTOOLS_HPP
#define ARRAYTOOLS_HPP
#pragma once


#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#ifdef __CUDACC__
#include <cublas.h>
#endif


/// @brief Tools for working with arrays.
namespace arraytools
{
  /**
    Allocate an array. Wrapper around malloc().
    
    @param[in] len Number of elements (not the number of bytes!).
    @param[out] x Array to be allocated.
   */
  template <typename T>
  static inline void alloc(const size_t len, T **x)
  {
    *x = (T*) std::malloc(len*sizeof(T));
  }
  
  /// \overload
  template <typename T>
  static inline void alloc(const size_t nrows, const size_t ncols, T **x)
  {
    const size_t len = nrows * ncols;
    alloc(len, x);
  }
  
  
  
  /**
    Zero-allocate an array. Wrapper around calloc.
    
    @param[in] len Number of elements (not the number of bytes!).
    @param[out] x Array to be allocated.
   */
  template <typename T>
  static inline void zero_alloc(const size_t len, T **x)
  {
    *x = (T*) std::calloc(len, sizeof(T));
  }
  
  template <typename T>
  static inline void zero_alloc(const size_t nrows, const size_t ncols, T **x)
  {
    const size_t len = nrows * ncols;
    zero_alloc(len, x);
  }
  
  
  
  /**
    Re-allocate an array. Wrapper around realloc(). If the realloc fails, the
    array will be set to NULL.
    
    @param[in] len Number of elements (not the number of bytes!).
    @param[out] x Array to be re-allocated.
   */
  template <typename T>
  static inline void realloc(const size_t len, T **x)
  {
    void *realloc_ptr = std::realloc(*x, len*sizeof(T));
    if (realloc_ptr == NULL)
      std::free(*x);
    
    *x = (T*) realloc_ptr;
  }
  
  /// \overload
  template <typename T>
  static inline void realloc(const size_t nrows, const size_t ncols, T **x)
  {
    const size_t len = nrows * ncols;
    realloc(len, x);
  }
  
  
  
  /**
    Free an array if supplied pointer is not NULL. Wrapper around free().
    
    @param[in] x Array to be allocated.
   */
  template <typename T>
  static inline void free(T *x)
  {
    if (x)
      std::free(x);
  }
  
  
  
  /** 
    Copy one array onto another. Array types can differ. If they are the same, it
    reduces to a memcpy() call.
    
    @param[in] len Number of elements (not the number of bytes!).
    @param[in] src Source array.
    @param[out] dst Destination array.
   */
  template <typename T>
  static inline void copy(const size_t len, const T *src, T *dst)
  {
    std::memcpy(dst, src, len*sizeof(*src));
  }
  
  /// \overload
  template <typename T>
  static inline void copy(const size_t nrows, const size_t ncols, const T *src, T *dst)
  {
    const size_t len = nrows * ncols;
    copy(len, src, dst);
  }
  
  /// \overload
  template <typename SRC, typename DST>
  static inline void copy(const size_t len, const SRC *src, DST *dst)
  {
    #pragma omp for simd
    for (size_t i=0; i<len; i++)
      dst[i] = (DST) src[i];
  }

  /// \overload
  template <typename SRC, typename DST>
  static inline void copy(const size_t nrows, const size_t ncols, const SRC *src, DST *dst)
  {
    const size_t len = nrows * ncols;
    copy(len, src, dst);
  }
  
  
  
  /**
    Set an array's values to 0. Wrapper around memset().
    
    @param[in] len Number of elements (not the number of bytes!).
    @param[inout] x Array to be zeroed.
   */
  template <typename T>
  static inline void zero(const size_t len, T *x)
  {
    std::memset(x, 0, len*sizeof(*x));
  }
  
  /// \overload
  template <typename T>
  static inline void zero(const size_t nrows, const size_t ncols, T *x)
  {
    const size_t len = nrows * ncols;
    zero(len, x);
  }
  
  
  
  namespace
  {
    static const int ALLOC_FAILED = -1;
    static const int ALLOC_OK = 0;
    
    
    
    static inline int check_alloc()
    {
      return ALLOC_OK;
    }
    
    template <typename T>
    static inline int check_alloc(T *x)
    {
      if (x == NULL)
        return ALLOC_FAILED;
      else
        return ALLOC_OK;
    }
    
    template <typename T, typename... VAT>
    static inline int check_alloc(T *x, VAT... vax)
    {
      return check_alloc(x) + check_alloc(vax ...);
    }
    
    
    
    static inline void vafree(){}
    
    template <typename T, typename... VAT>
    static inline void vafree(T *x, VAT... vax)
    {
      arraytools::free(x);
      vafree(vax ...);
    }
  }
  
  
  
  /**
    Check variable number of arrays. If one is NULL, then all others will be
    automatically freed and std::bad_alloc() will be thrown.
    
    @param[in] x Array.
    @param[in] vax Optional more arrays.
   */
  template <typename T, typename... VAT>
  static inline void check_allocs(T *x, VAT... vax)
  {
    int check = check_alloc(x) + check_alloc(vax ...);
    
    if (check != ALLOC_OK)
    {
      arraytools::free(x);
      vafree(vax ...);
      
      throw std::bad_alloc();
    }
  }
  
  
  
  namespace fltcmp
  {
    namespace
    {
      // eps ~ sqrt(precision)
    #ifdef __CUDACC__
      const __half eps_hlf = 1e-2; // machine epsilon 0x1p-11 ~ 2.2e-2
    #endif
      const float eps_flt = 1e-4;
      const double eps_dbl = 1e-8;
      
      // smallest representible number with 1 as the leading bit of the significand
    #ifdef __CUDACC__
      const __half min_normal_hlf = 0x1p-14; // (2^(5-1)-1)-1 = 14
    #endif
      const float min_normal_flt = 0x1p-126;
      const double min_normal_dbl = 0x1p-1022;
      
    #ifdef __CUDACC__
      const __half max_hlf = 1>>15;
    #endif
      const float max_flt = FLT_MAX;
      const double max_dbl = DBL_MAX;
      
      
      
    #ifdef __CUDACC__
      static inline __half fabsh(const __half x)
      {
        return __float2half(fabsf(__half2float(x)));
      }
    #endif
      
      
      
      static inline bool subnormal(const float x)
      {
        return (x < min_normal_flt);
      }
      
      static inline bool subnormal(const double x)
      {
        return (x < min_normal_dbl);
      }
      
      
      
      template <typename T>
      static inline bool abseq(const T x, const T y)
      {
        float fx = (float) x;
        float fy = (float) y;
        return fabsf(fx - fy) < (eps_flt * min_normal_flt);
      }
      
      static inline bool abseq(const float x, const float y)
      {
        return fabsf(x - y) < (eps_flt * min_normal_flt);
      }
      
      static inline bool abseq(const double x, const double y)
      {
        return fabs(x - y) < (eps_dbl * min_normal_dbl);
      }
      
      
      
      template <typename T>
      static inline bool releq(const T x, const T y)
      {
        float fx = (float) x;
        float fy = (float) y;
        return fabsf(fx - fy) / std::min(fabsf(fx)+fabsf(fy), max_flt) < eps_flt;
      }
      
      static inline bool releq(const float x, const float y)
      {
        return fabsf(x - y) / std::min(fabsf(x)+fabsf(y), max_flt) < eps_flt;
      }
      
      static inline bool releq(const double x, const double y)
      {
        return fabs(x - y) / std::min(fabs(x)+fabs(y), max_dbl) < eps_dbl;
      }
    }
    
    
    
    // modified from https://floating-point-gui.de/errors/comparison/
    template <typename REAL>
    static inline bool eq(const REAL x, const REAL y)
    {
      if (x == y)
        return true;
      else if (x == 0.f || y == 0.f || subnormal(fabs(x)+fabs(y)))
        return abseq(x, y);
      else
        return releq(x, y);
    }
    
    
    
    static inline bool eq(const int x, const int y)
    {
      return (x == y);
    }
    
    
    
    template <typename REAL>
    static inline bool eq(const REAL x, const int y)
    {
      return eq(x, (REAL) y);
    }
    
    template <typename REAL>
    static inline bool eq(const int x, const REAL y)
    {
      return eq((REAL) x, y);
    }
    
    
    
    static inline bool eq(const float x, double y)
    {
      return eq(x, (float) y);
    }
    
    static inline bool eq(const double x, const float y)
    {
      return eq((float) x, y);
    }
  }
  
  
  
  /** 
    Do the two arrays contain the same elements (up to a cast)? Not suitable
    for floats.
    
    @param[in] len Number of elements (not the number of bytes!).
    @param[in] a,b The arrays.
    @return The index of the first mismatch, or len if there is no mismatch.
   */
  template <typename TA, typename TB>
  static inline size_t cmp_firstmiss(const size_t len, const TA *a, const TB *b)
  {
    size_t ret = len;
    
    #pragma omp simd reduction(min:ret)
    for (size_t i=0; i<len; i++)
    {
      if (!fltcmp::eq(a[i], b[i]))
        ret = i;
    }
    
    return ret;
  }
  
  
  
  /** 
    Do the two numbers or arrays of numbers contain the same values (up to a
    cast)?
    
    @param[in] len Number of elements (not the number of bytes!).
    @param[in] a,b The arrays.
    @return 'true' if the elements are the same and 'false' otherwise.
   */
  template <typename TA, typename TB>
  static inline bool cmp(const size_t len, const TA *a, const TB *b)
  {
    size_t fm = cmp_firstmiss(len, a, b);
    return (fm == len ? true : false);
  }
  
  /// \overload
  template <typename TA, typename TB>
  static inline bool cmp(const TA a, const TB b)
  {
    return fltcmp::eq(a, b);
  }
}


#endif
