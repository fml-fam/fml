// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_GPU_ARCH_CUDA_GPURAND_H
#define FML_GPU_ARCH_CUDA_GPURAND_H
#pragma once


#include <curand.h>


namespace gpurand
{
  namespace defs
  {
    static const curandRngType_t gen_type = CURAND_RNG_PSEUDO_MTGP32;
  }
  
  
  
  namespace err
  {
    static inline void check_init(curandStatus_t st)
    {
      if (st != CURAND_STATUS_SUCCESS)
        throw std::runtime_error("unable to initialize GPU generator");
    }
    
    static inline void check_seed_set(curandStatus_t st)
    {
      if (st != CURAND_STATUS_SUCCESS)
        throw std::runtime_error("unable to set GPU generator seed");
    }
    
    static inline void check_generation(curandStatus_t st)
    {
      if (st != CURAND_STATUS_SUCCESS)
        throw std::runtime_error("unable to utilize GPU generator");
    }
  }
  
  
  
  namespace generics
  {
    static inline curandStatus_t gpu_rand_unif(curandGenerator_t generator, float *outputPtr, size_t num)
    {
      return curandGenerateUniform(generator, outputPtr, num);
    }
    
    static inline curandStatus_t gpu_rand_unif(curandGenerator_t generator, double *outputPtr, size_t num)
    {
      return curandGenerateUniformDouble(generator, outputPtr, num);
    }
    
    
    
    static inline curandStatus_t gpu_rand_norm(curandGenerator_t generator, float *outputPtr, size_t num, float mean, float stddev)
    {
      return curandGenerateNormal(generator, outputPtr, num, mean, stddev);
    }
    
    static inline curandStatus_t gpu_rand_norm(curandGenerator_t generator, double *outputPtr, size_t num, double mean, double stddev)
    {
      return curandGenerateNormalDouble(generator, outputPtr, num, mean, stddev);
    }
  }
  
  
  
  template <typename REAL>
  inline void gen_runif(const uint32_t seed, const size_t len, REAL *x)
  {
    curandStatus_t st;
    curandGenerator_t gen;
    
    st = curandCreateGenerator(&gen, defs::gen_type);
    err::check_init(st);
    
    st = curandSetPseudoRandomGeneratorSeed(gen, seed);
    err::check_seed_set(st);
    st = generics::gpu_rand_unif(gen, x, len);
    err::check_generation(st);
    
    curandDestroyGenerator(gen);
  }
  
  
  
  template <typename REAL>
  inline void gen_rnorm(const uint32_t seed, const REAL mean, const REAL sd, const size_t len, REAL *x)
  {
    curandStatus_t st;
    curandGenerator_t gen;
    
    st = curandCreateGenerator(&gen, defs::gen_type);
    err::check_init(st);
    
    st = curandSetPseudoRandomGeneratorSeed(gen, seed);
    err::check_seed_set(st);
    st = generics::gpu_rand_norm(gen, x, len, mean, sd);
    err::check_generation(st);
    
    curandDestroyGenerator(gen);
  }
}


#endif
