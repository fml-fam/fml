// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_CPU_STATS_H
#define FML_CPU_STATS_H
#pragma once


#include "dimops.hh"
#include "linalg.hh"


/// @brief Statistics kernels.
namespace stats
{
  /**
    @brief Principal components analysis.
    
    @param[in] rm_mean,rm_sd Should the column means/sds be removed?
    @param[inout] x Input data. Values are overwritten.
    @param[out] sdev Standard deviations of the principal components.
    @param[out] rot The variable loadings.
    
    @impl Uses `linalg::svd()`.
    
    @allocs If the dimensions of the outputs are inappropriately sized, they
    will automatically be re-allocated.
    
    @except If a reallocation is triggered and fails, a `bad_alloc` exception
    will be thrown.
    
    @tparam REAL should be 'float' or 'double'.
   */
  template <typename REAL>
  void pca(const bool rm_mean, const bool rm_sd, cpumat<REAL> &x,
    cpuvec<REAL> &sdev, cpumat<REAL> &rot)
  {
    dimops::scale(rm_mean, rm_sd, x);
    
    cpumat<REAL> u;
    cpumat<REAL> trot;
    linalg::svd(x, sdev, u, trot);
    
    const REAL d = 1.0 / sqrt((REAL)std::max(x.nrows()-1, 1));
    sdev.scale(d);
    
    linalg::xpose(trot, rot);
  }
  
  /// \overload
  template <typename REAL>
  void pca(const bool rm_mean, const bool rm_sd, cpumat<REAL> &x,
    cpuvec<REAL> &sdev)
  {
    dimops::scale(rm_mean, rm_sd, x);
    
    linalg::svd(x, sdev);
    
    const REAL d = 1.0 / sqrt((REAL)std::max(x.nrows()-1, 1));
    sdev.scale(d);
  }
}


#endif
