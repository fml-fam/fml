// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_GPU_LINALG_LINALG_ERR_H
#define FML_GPU_LINALG_LINALG_ERR_H
#pragma once


#include <stdexcept>

#include "../arch/arch.hh"

#include "../gpumat.hh"
#include "../gpuvec.hh"


namespace fml
{
namespace linalg
{
  namespace err
  {
    template <class T, class S>
    void check_card(const T &a, const S &b)
    {
      if (a.get_card()->get_id() != b.get_card()->get_id())
        throw std::runtime_error("gpumat/gpuvex objects must be allocated on the same gpu");
    }
  }
}
}


#endif
