#ifndef FML_FLTCMP_H
#define FML_FLTCMP_H


namespace fltcmp
{
  const float eps_flt = 1e-4;
  const double eps_dbl = 1e-8;
  
  
  inline bool eq(const float x, const float y)
  {
    return (std::abs(x-y) < eps_flt);
  }
  
  
  
  inline bool eq(const double x, const double y)
  {
    return (std::abs(x-y) < eps_dbl);
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
