// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_GPU_INTERNALS_LAUNCHER_H
#define FML_GPU_INTERNALS_LAUNCHER_H
#pragma once


#include "../../_internals/types.hh"


namespace kernel_launcher
{
  namespace
  {
    static const int BLOCK_SIZE = 16;
    
    static inline int grid_len(len_t len)
    {
      return (len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    }
  }
  
  
  
  static inline dim3 dim_block1()
  {
    dim3 block(BLOCK_SIZE);
    return block;
  }
  
  static inline dim3 dim_block2()
  {
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    return block;
  }
  
  
  
  static inline dim3 dim_grid(len_t len)
  {
    dim3 grid(grid_len(len));
    return grid;
  }
  
  static inline dim3 dim_grid(len_t m, len_t n)
  {
    dim3 grid(grid_len(m), grid_len(n));
    return grid;
  }
}


#endif
