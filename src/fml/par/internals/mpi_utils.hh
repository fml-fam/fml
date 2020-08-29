// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_PAR_INTERNALS_MPI_UTILS_H
#define FML_PAR_INTERNALS_MPI_UTILS_H
#pragma once


#include "../comm.hh"


namespace fml
{
  namespace mpi
  {
    static const int REDUCE_TO_ALL = -1;
    
    
    
    static inline void check_MPI_ret(int ret)
    {
      if (ret != MPI_SUCCESS)
      {
        int slen;
        char s[MPI_MAX_ERROR_STRING];
        
        MPI_Error_string(ret, s, &slen);
        throw std::runtime_error(s);
      }
    }
    
    
    
    void contig_type(const int count, const float *x, MPI_Datatype *newtype)
    {
      (void)x;
      // int ret = 
      MPI_Type_contiguous(count, MPI_FLOAT, newtype);
      MPI_Type_commit(newtype);
    }
    
    void contig_type(const int count, const double *x, MPI_Datatype *newtype)
    {
      (void)x;
      MPI_Type_contiguous(count, MPI_DOUBLE, newtype);
      MPI_Type_commit(newtype);
    }
  }
}


#endif
