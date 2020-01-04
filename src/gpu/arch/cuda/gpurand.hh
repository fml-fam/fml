// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_GPU_ARCH_CUDA_GPURAND_H
#define FML_GPU_ARCH_CUDA_GPURAND_H
#pragma once


#include <curand.h>


namespace gpurand
{
  namespace err
  {
    inline void check_init(curandStatus_t st)
    {
      if (st != CURAND_STATUS_SUCCESS)
        throw std::runtime_error("unable to initialize GPU generator");
    }
    
    inline void check_seed_set(curandStatus_t st)
    {
      if (st != CURAND_STATUS_SUCCESS)
        throw std::runtime_error("unable to set GPU generator seed");
    }
    
    inline void check_generation(curandStatus_t st)
    {
      if (st != CURAND_STATUS_SUCCESS)
        throw std::runtime_error("unable to utilize GPU generator");
    }
  }
  
  
  
  inline void gen_runif(const uint32_t seed, const size_t len, float *x)
  {
    curandStatus_t st;
    curandGenerator_t gen;
    
    st = curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MT19937);
    err::check_init(st);
    
    st = curandSetPseudoRandomGeneratorSeed(gen, seed);
    err::check_seed_set(st);
    st = curandGenerateUniform(gen, x, len);
    err::check_generation(st);
    
    curandDestroyGenerator(gen);
  }
  
  inline void gen_runif(const uint32_t seed, const size_t len, double *x)
  {
    curandStatus_t st;
    curandGenerator_t gen;
    
    st = curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MT19937);
    err::check_init(st);
    
    st = curandSetPseudoRandomGeneratorSeed(gen, seed);
    err::check_seed_set(st);
    st = curandGenerateUniformDouble(gen, x, len);
    err::check_generation(st);
    
    curandDestroyGenerator(gen);
  }
  
  
  
  inline void gen_rnorm(const uint32_t seed, const float mean, const float sd, const size_t len, float *x)
  {
    curandStatus_t st;
    curandGenerator_t gen;
    
    st = curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MT19937);
    err::check_init(st);
    
    st = curandSetPseudoRandomGeneratorSeed(gen, seed);
    err::check_seed_set(st);
    st = curandGenerateNormal(gen, x, len, mean, sd);
    err::check_generation(st);
    
    curandDestroyGenerator(gen);
  }
  
  inline void gen_rnorm(const uint32_t seed, const float mean, const float sd, const size_t len, double *x)
  {
    curandStatus_t st;
    curandGenerator_t gen;
    
    st = curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MT19937);
    err::check_init(st);
    
    st = curandSetPseudoRandomGeneratorSeed(gen, seed);
    err::check_seed_set(st);
    st = curandGenerateNormalDouble(gen, x, len, mean, sd);
    err::check_generation(st);
    
    curandDestroyGenerator(gen);
  }
}


#endif
