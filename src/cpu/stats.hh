// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_CPU_STATS_H
#define FML_CPU_STATS_H
#pragma once


#include "dimops.hh"
#include "linalg.hh"


namespace stats
{
  template <typename REAL>
  void pca(const bool mean, const bool sd, cpumat<REAL> &x, cpuvec<REAL> &sdev, cpumat<REAL> &rot)
  {
    dimops::scale(mean, sd, x);
    
    cpumat<REAL> u;
    cpumat<REAL> trot;
    linalg::svd(x, sdev, u, trot);
    linalg::xpose(trot, rot);
  }
}


#endif
