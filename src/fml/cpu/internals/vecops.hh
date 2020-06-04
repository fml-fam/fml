// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_CPU_INTERNALS_VECOPS_H
#define FML_CPU_INTERNALS_VECOPS_H
#pragma once


#include <cmath>

#include "../../_internals/types.hh"


namespace fml
{
  namespace vecops
  {
    namespace cpu
    {
      template <typename REAL>
      static inline void sum(const len_t len, const REAL *x, REAL &s)
      {
        s = 0;
          
        #pragma omp simd reduction(+:s)
        for (len_t i=0; i<len; i++)
          s += x[i];
      }
      
      
      
      template <typename REAL>
      static inline void sweep_add(const REAL c, const len_t len, REAL *x)
      {
        #pragma omp for simd
        for (len_t i=0; i<len; i++)
          x[i] += c;
      }
      
      
      
      template <typename REAL>
      static inline void sweep_mul(const REAL c, const len_t len, REAL *x)
      {
        #pragma omp for simd
        for (len_t i=0; i<len; i++)
          x[i] *= c;
      }
      
      
      
      template <typename REAL>
      static inline void var(const len_t n, const REAL *x, REAL &mean, REAL &var)
      {
        mean = 0;
        var = 0;
        
        for (len_t i=0; i<n; i++)
        {
          REAL dt = x[i] - mean;
          mean += dt/((REAL) i+1);
          var += dt * (x[i] - mean);
        }
        
        var = sqrt(var / ((REAL) n-1));
      }
    }
  }
}


#endif
