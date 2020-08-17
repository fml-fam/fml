// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_MPI_LINALG_LINALG_ERR_H
#define FML_MPI_LINALG_LINALG_ERR_H
#pragma once


#include <stdexcept>

#include "../mpimat.hh"


namespace fml
{
namespace linalg
{
  namespace err
  {
    template <class T>
    void check_grid(const T &a){}
    
    template <class T, class S>
    void check_grid(const T &a, const S &b)
    {
      if (a.get_grid().ictxt() != b.get_grid().ictxt())
        throw std::runtime_error("mpimat objects must be distributed on the same process grid");
    }
    
    template <class T, class S, typename... VAT>
    void check_grid(const T &a, const S &b, VAT&&... vax)
    {
      check_grid(a, b);
      check_grid(a, vax ...);
    }
  }
}
}


#endif
