// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_GPU_ARCH_CUDA_GPURAND_H
#define FML_GPU_ARCH_CUDA_GPURAND_H
#pragma once


#include <curand.h>


namespace gpurand
{
  inline void gen_runif(const uint32_t seed, const size_t len, float *x)
  {
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MT19937);
    
    curandSetPseudoRandomGeneratorSeed(gen, seed);
    curandGenerateUniform(gen, x, len);
    
    curandDestroyGenerator(gen);
  }
  
  inline void gen_runif(const uint32_t seed, const size_t len, double *x)
  {
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MT19937);
    
    curandSetPseudoRandomGeneratorSeed(gen, seed);
    curandGenerateUniformDouble(gen, x, len);
    
    curandDestroyGenerator(gen);
  }
  
  
  
  inline void gen_rnorm(const uint32_t seed, const float mean, const float sd, const size_t len, float *x)
  {
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MT19937);
    
    curandSetPseudoRandomGeneratorSeed(gen, seed);
    curandGenerateNormal(gen, x, len, mean, sd);
    
    curandDestroyGenerator(gen);
  }
  
  inline void gen_rnorm(const uint32_t seed, const float mean, const float sd, const size_t len, double *x)
  {
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MT19937);
    
    curandSetPseudoRandomGeneratorSeed(gen, seed);
    curandGenerateNormalDouble(gen, x, len, mean, sd);
    
    curandDestroyGenerator(gen);
  }
}


#endif
