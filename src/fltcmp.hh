#ifndef FML_FLTCMP_H
#define FML_FLTCMP_H


#include <algorithm>
#include <cfloat>
#include <cmath>

#ifdef __CUDACC__
#include <cublas.h>
#endif


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
    inline __half fabsh(const __half x)
    {
      return __float2half(fabsf(__half2float(x)));
    }
  #endif
    
    
    
    inline bool subnormal(const float x)
    {
      return (x < min_normal_flt);
    }
    
    inline bool subnormal(const double x)
    {
      return (x < min_normal_dbl);
    }
    
    
    
    inline bool abseq(const float x, const float y)
    {
      return fabsf(x - y) < (eps_flt * min_normal_flt);
    }
    
    inline bool abseq(const double x, const double y)
    {
      return fabs(x - y) < (eps_dbl * min_normal_dbl);
    }
    
    
    
    inline bool releq(const float x, const float y)
    {
      return fabsf(x - y) / std::min(fabsf(x)+fabsf(y), max_flt) < eps_flt;
    }
    
    inline bool releq(const double x, const double y)
    {
      return fabs(x - y) / std::min(fabs(x)+fabs(y), max_dbl) < eps_dbl;
    }
  }
  
  
  
  // modified from https://floating-point-gui.de/errors/comparison/
  template <typename REAL>
  bool eq(const REAL x, const REAL y)
  {
    if (x == y)
      return true;
    else if (x == 0.f || y == 0.f || subnormal(fabs(x)+fabs(y)))
      return abseq(x, y);
    else
      return releq(x, y);
  }
  
  
  
  inline bool eq(const int x, const int y)
  {
    return (x == y);
  }
  
  
  
  template <typename REAL>
  inline bool eq(const REAL x, const int y)
  {
    return eq(x, (REAL) y);
  }
  
  template <typename REAL>
  inline bool eq(const int x, const REAL y)
  {
    return eq((REAL) x, y);
  }
  
  
  
  inline bool eq(const float x, double y)
  {
    return eq(x, (float) y);
  }
  
  inline bool eq(const double x, const float y)
  {
    return eq((float) x, y);
  }
}


#endif
