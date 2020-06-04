// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_CPU_INTERNALS_CPU_UTILS_H
#define FML_CPU_INTERNALS_CPU_UTILS_H
#pragma once


namespace fml
{
  namespace cpu_utils
  {
    // zero specified triangle
    template <typename REAL>
    void tri2zero(const char uplo, const bool diag, const len_t m, const len_t n,
      REAL *A, const len_t lda)
    {
      if (uplo == 'U')
      {
        const len_t offset = diag ? 1 : 0;
        
        for (len_t j=0; j<n; j++)
        {
          const len_t top = std::min(j+offset, m);
          
          #pragma omp for simd
          for (len_t i=0; i<top; i++)
            A[i + lda*j] = (REAL)0;
        }
      }
      else // if (uplo == 'L')
      {
        const len_t offset = diag ? 0 : 1;
        
        for (len_t j=0; j<n; j++)
        {
          #pragma omp for simd
          for (len_t i=j+offset; i<m; i++)
            A[i + lda*j] = (REAL)0;
        }
      }
    }
  }
}


#endif
