// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_PAR_GPU_H
#define FML_PAR_GPU_H
#pragma once


#include "internals/parmat.hh"
#include "gpu/parmat.hh"
#include "cpu/copy.hh"


namespace fml
{
  /**
    @brief Returns the device number of the GPU to use with the calling MPI
    process.
    
    @details 
    
    @param[in] c A communicator object.
    @return The ordinal device number for the calling MPI process to use in,
    e.g. `fml::new_card()`.
    
    @except If there are more MPI ranks within any node than there are GPUS, the
    function will throw a 'runtime_error' exception.
   */
  inline int get_device_num(const comm &c)
  {
    int ngpus = get_device_count();
    // if (c.localsize() > ngpus)
    //   throw std::runtime_error("parmat_gpu can not be used with more MPI ranks than GPUs per node");
    
    return c.rank() % ngpus;
  }
}


#endif
