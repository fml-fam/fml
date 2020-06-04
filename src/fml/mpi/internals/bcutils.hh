// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_MPI_INTERNALS_BCUTILS_H
#define FML_MPI_INTERNALS_BCUTILS_H
#pragma once


#include <cmath>
#include <stdlib.h>


namespace fml
{
  namespace bcutils
  {
    enum desc_pos
    {
      DESC_DTYPE,
      DESC_CTXT,
      DESC_M,
      DESC_N,
      DESC_MB,
      DESC_NB,
      DESC_RSRC,
      DESC_CSRC,
      DESC_LLD
    };
    
    
    
    enum blacs_pos
    {
      GRID_NPROCS,
      GRID_NPROW,
      GRID_NPCOL,
      GRID_MYPROW,
      GRID_MYPCOL
    };
    
    
    
    inline void descinit(int *desc, const int ictxt, const int m, const int n, const int mb, const int nb, const int lld)
    {
      desc[DESC_DTYPE] = 1;
      desc[DESC_CTXT] = ictxt;
      desc[DESC_M] = m;
      desc[DESC_N] = n;
      desc[DESC_MB] = mb;
      desc[DESC_NB] = nb;
      desc[DESC_RSRC] = 0;
      desc[DESC_CSRC] = 0;
      desc[DESC_LLD] = (lld<1?1:lld);
    }
    
    
    
    inline int numroc(int n, int nb, int iproc, int isrcproc, int nprocs)
    {
      int mydist = (nprocs+iproc-isrcproc) % nprocs;
      int nblocks = n / nb;
      
      int ret = (nblocks/nprocs) * nb;
      
      int extrablks = nblocks % nprocs;
      if (mydist < extrablks)
        ret += nb;
      else if (mydist == extrablks)
        ret += (n%nb);
      
      return ret;
    }
    
    
    
    inline int l2g(const int i, const int nb, const int nprocs, const int myproc)
    {
      return nprocs*nb*(i/nb) + (i%nb) + ((nprocs+myproc)%nprocs)*nb;
    }
    
    
    
    inline int g2l(const int gi, const int nb, const int nprocs)
    {
      return nb * (gi/(nb*nprocs)) + (gi%nb);
    }
    
    
    
    inline int g2p(const int gi, const int nb, const int nprocs)
    {
      return (gi/nb) % nprocs;
    }
  }
}


#endif
