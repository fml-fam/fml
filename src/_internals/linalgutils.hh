// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML__INTERNALS_LINALGUTILS_H
#define FML__INTERNALS_LINALGUTILS_H
#pragma once


#include <stdexcept>

#include "types.hh"


namespace linalgutils
{
  inline void matadd_params(const bool transx, const bool transy, const len_t mx, const len_t nx, const len_t my, const len_t ny, len_t *m, len_t *n)
  {
    if (!transx && !transy)
    {
      if (mx != my || nx != ny)
        throw std::runtime_error("non-conformable arguments");
      
      *m = mx;
      *n = nx;
    }
    else if (transx && transy)
    {
      if (mx != my || nx != ny)
        throw std::runtime_error("non-conformable arguments");
      
      *m = nx;
      *n = mx;
    }
    else if (transx && !transy)
    {
      if (mx != ny || nx != my)
        throw std::runtime_error("non-conformable arguments");
      
      *m = nx;
      *n = mx;
    }
    else if (!transx && transy)
    {
      if (mx != ny || nx != my)
        throw std::runtime_error("non-conformable arguments");
      
      *m = mx;
      *n = nx;
    }
    else
    {
      throw std::runtime_error("this should be impossible");
    }
  }
  
  
  
  inline void matmult_params(const bool transx, const bool transy, const len_t mx, const len_t nx, const len_t my, const len_t ny, len_t *m, len_t *n, len_t *k)
  {
    if (!transx && !transy)
    {
      if (nx != my)
        throw std::runtime_error("non-conformable arguments");
    }
    else if (transx && transy)
    {
      if (mx != ny)
        throw std::runtime_error("non-conformable arguments");
    }
    else if (transx && !transy)
    {
      if (mx != my)
        throw std::runtime_error("non-conformable arguments");
    }
    else if (!transx && transy)
    {
      if (nx != ny)
        throw std::runtime_error("non-conformable arguments");
    }
    
    // m = # rows of op(x)
    // n = # cols of op(y)
    // k = # cols of op(x)
    
    if (transx)
    {
      *m = nx;
      *k = mx;
    }
    else
    {
      *m = mx;
      *k = nx;
    }
    
    *n = transy ? my : ny;
  }
}


#endif
