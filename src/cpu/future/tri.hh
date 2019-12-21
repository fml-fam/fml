// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_CPU_FUTURE_TRI_H
#define FML_CPU_FUTURE_TRI_H
#pragma once


#include <cmath>
#include <stdexcept>

#include "../cpumat.hh"


namespace tri
{
  template <typename REAL>
  bool is_tri(const cpumat<REAL> &x)
  {
    
    return true;
  }

  template <typename REAL>
  bool is_lowertri(const cpumat<REAL> &x)
  {
    
    return true;
  }

  template <typename REAL>
  bool is_uppertri(const cpumat<REAL> &x)
  {
    
    return true;
  }
  

  
  template <typename REAL>
  void make_uppertri(cpumat<REAL> &x)
  {

  }

  template <typename REAL>
  void make_lowertri(cpumat<REAL> &x)
  {
     
  }
}


#endif
