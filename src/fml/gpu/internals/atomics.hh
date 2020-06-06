// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_GPU_INTERNALS_ATOMICS_H
#define FML_GPU_INTERNALS_ATOMICS_H
#pragma once


#include <cuda_runtime.h>

#include "../../_internals/types.hh"

#include "../../_internals/arraytools/src/arraytools.cuh"


namespace fml
{
  namespace atomics
  {
    // Based on https://forums.developer.nvidia.com/t/atomicmax-with-floats/8791/10
    static __device__ float atomicMaxf(float *address, float val)
    {
      int *address_int = (int*) address;
      int old = *address_int, assumed;
      
      while (val > __int_as_float(old))
      {
        assumed = old;
        old = atomicCAS(address_int, assumed, __float_as_int(val));
      }
      
      return __int_as_float(old);
    }
    
    static __device__ double atomicMaxf(double *address, double val)
    {
      unsigned long long *address_ull = (unsigned long long*) address;
      unsigned long long old = *address_ull, assumed;
      
      while (val > __longlong_as_double(old))
      {
        assumed = old;
        old = atomicCAS(address_ull, assumed, __double_as_longlong(val));
      }
      
      return __longlong_as_double(old);
    }
    
    
    
    static __device__ float atomicMinf(float *address, float val)
    {
      int *address_int = (int*) address;
      int old = *address_int, assumed;
      
      while (val < __int_as_float(old))
      {
        assumed = old;
        old = atomicCAS(address_int, assumed, __float_as_int(val));
      }
      
      return __int_as_float(old);
    }
    
    static __device__ double atomicMinf(double *address, double val)
    {
      unsigned long long *address_ull = (unsigned long long*) address;
      unsigned long long old = *address_ull, assumed;
      
      while (val < __longlong_as_double(old))
      {
        assumed = old;
        old = atomicCAS(address_ull, assumed, __double_as_longlong(val));
      }
      
      return __longlong_as_double(old);
    }
  }
}


#endif
