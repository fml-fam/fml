// This file is part of arraytools which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef ARRAYTOOLS_CUH
#define ARRAYTOOLS_CUH
#pragma once


#include <cfloat>
#include <cmath>
#include <cstdlib>

#ifdef __CUDACC__
#include <cublas.h>
#endif


namespace arraytools
{
  namespace fltcmp_gpu
  {
    #define ARRAYTOOLS_FLTCMP_GPU_MIN(a,b) ((a)<(b)?(a):(b))
    
    #define FML_FLTCMP_GPU_EPS_HLF ((__half)1e-2) // machine epsilon 0x1p-11 ~ 2.2e-2
    #define FML_FLTCMP_GPU_EPS_FLT ((float)1e-4)
    #define FML_FLTCMP_GPU_EPS_DBL ((double)1e-8)
      
    // smallest representible number with 1 as the leading bit of the significand
    #define FML_FLTCMP_GPU_MIN_NORMAL_HLF ((__half)0x1p-14) // (2^(5-1)-1)-1 = 14
    #define FML_FLTCMP_GPU_MIN_NORMAL_FLT ((float)0x1p-126)
    #define FML_FLTCMP_GPU_MIN_NORMAL_DBL ((double)0x1p-1022)
      
    #define FML_FLTCMP_GPU_MAX_HLF ((__half)1>>15)
    #define FML_FLTCMP_GPU_MAX_FLT ((float)FLT_MAX)
    #define FML_FLTCMP_GPU_MAX_DBL ((double)DBL_MAX)
    
    
    static __device__ void subnormal(const float *x, bool *ret)
    {
      (*ret) = (*x < FML_FLTCMP_GPU_MIN_NORMAL_FLT);
    }
    
    static __device__ void subnormal(const double *x, bool *ret)
    {
      (*ret) = (*x < FML_FLTCMP_GPU_MIN_NORMAL_DBL);
    }
    
    
    
    template <typename T>
    static __device__ void abseq(const T *x, const T *y, bool *ret)
    {
      float fx = (float) *x;
      float fy = (float) *y;
      (*ret) = (fabsf(fx - fy) < (FML_FLTCMP_GPU_EPS_FLT * FML_FLTCMP_GPU_MIN_NORMAL_FLT));
    }
    
    static __device__ void abseq(const float *x, const float *y, bool *ret)
    {
      (*ret) = (fabsf(*x - *y) < (FML_FLTCMP_GPU_EPS_FLT * FML_FLTCMP_GPU_MIN_NORMAL_FLT));
    }
    
    static __device__ void abseq(const double *x, const double *y, bool *ret)
    {
      (*ret) = (fabs(*x - *y) < (FML_FLTCMP_GPU_EPS_DBL * FML_FLTCMP_GPU_MIN_NORMAL_DBL));
    }
    
    
    
    template <typename T>
    static __device__ void releq(const T *x, const T *y, bool *ret)
    {
      float fx = (float) *x;
      float fy = (float) *y;
      (*ret) = fabsf(fx - fy) / ARRAYTOOLS_FLTCMP_GPU_MIN(fabsf(fx)+fabsf(fy), FML_FLTCMP_GPU_MAX_FLT) < FML_FLTCMP_GPU_EPS_FLT;
    }
    
    static __device__ void releq(const float *x, const float *y, bool *ret)
    {
      (*ret) = fabsf(*x - *y) / ARRAYTOOLS_FLTCMP_GPU_MIN(fabsf(*x)+fabsf(*y), FML_FLTCMP_GPU_MAX_FLT) < FML_FLTCMP_GPU_EPS_FLT;
    }
    
    static __device__ void releq(const double *x, const double *y, bool *ret)
    {
      (*ret) = fabs(*x - *y) / ARRAYTOOLS_FLTCMP_GPU_MIN(fabs(*x)+fabs(*y), FML_FLTCMP_GPU_MAX_DBL) < FML_FLTCMP_GPU_EPS_DBL;
    }
    
    
    
    template <typename REAL>
    static inline __device__ void eq(const REAL *x, const REAL *y, bool *ret)
    {
      if (*x == *y)
        (*ret) = true;
      else
      {
        const float test = fabs(*x) + fabs(*y);
        bool is_subnormal;
        subnormal(&test, &is_subnormal);
        if (*x == 0.f || *y == 0.f || is_subnormal)
          abseq(x, y, ret);
        else
          releq(x, y, ret);
      }
    }
    
    
    
    static inline __device__ void eq(const int *x, const int *y, bool *ret)
    {
      (*ret) = (*x == *y);
    }
    
    
    
    template <typename REAL>
    static inline __device__ void eq(const REAL *x, const int *y, bool *ret)
    {
      const REAL y_REAL = (REAL) *y;
      eq(x, &y_REAL, ret);
    }
    
    template <typename REAL>
    static inline __device__ void eq(const int *x, const REAL *y, bool *ret)
    {
      eq(y, x, ret);
    }
    
    
    
    static inline __device__ void eq(const float *x, const double *y, bool *ret)
    {
      const float y_float = (float) *y;
      eq(x, &y_float, ret);
    }
    
    static inline __device__ void eq(const double *x, const float *y, bool *ret)
    {
      eq(y, x, ret);
    }
    
    
    
    #undef FML_FLTCMP_GPU_EPS_HLF
    #undef FML_FLTCMP_GPU_EPS_FLT
    #undef FML_FLTCMP_GPU_EPS_DBL
    
    #undef FML_FLTCMP_GPU_MIN_NORMAL_HLF
    #undef FML_FLTCMP_GPU_MIN_NORMAL_FLT
    #undef FML_FLTCMP_GPU_MIN_NORMAL_DBL
    
    #undef FML_FLTCMP_GPU_MAX_HLF
    #undef FML_FLTCMP_GPU_MAX_FLT
    #undef FML_FLTCMP_GPU_MAX_DBL
  }
}


#endif
