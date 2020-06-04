// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML_GPU_ARCH_HIP_GPURAND_H
#define FML_GPU_ARCH_HIP_GPURAND_H
#pragma once


#include <hiprand.hpp>


namespace gpurand
{
  namespace defs
  {
    static const hiprandRngType_t gen_type = HIPRAND_RNG_PSEUDO_MTGP32;
  }
  
  
  
  namespace err
  {
    static inline void check_init(hiprandStatus_t st)
    {
      if (st != HIPRAND_STATUS_SUCCESS)
        throw std::runtime_error("unable to initialize GPU generator");
    }
    
    static inline void check_seed_set(hiprandStatus_t st)
    {
      if (st != HIPRAND_STATUS_SUCCESS)
        throw std::runtime_error("unable to set GPU generator seed");
    }
    
    static inline void check_generation(hiprandStatus_t st)
    {
      if (st != HIPRAND_STATUS_SUCCESS)
        throw std::runtime_error("unable to utilize GPU generator");
    }
  }
  
  
  
  namespace generics
  {
    static inline hiprandStatus_t gpu_rand_unif(hiprandGenerator_t generator, float *outputPtr, size_t num)
    {
      return hiprandGenerateUniform(generator, outputPtr, num);
    }
    
    static inline hiprandStatus_t gpu_rand_unif(hiprandGenerator_t generator, double *outputPtr, size_t num)
    {
      return hiprandGenerateUniformDouble(generator, outputPtr, num);
    }
    
    
    
    static inline hiprandStatus_t gpu_rand_norm(hiprandGenerator_t generator, float *outputPtr, size_t num, float mean, float stddev)
    {
      return hiprandGenerateNormal(generator, outputPtr, num, mean, stddev);
    }
    
    static inline hiprandStatus_t gpu_rand_norm(hiprandGenerator_t generator, double *outputPtr, size_t num, double mean, double stddev)
    {
      return hiprandGenerateNormalDouble(generator, outputPtr, num, mean, stddev);
    }
  }
  
  
  
  template <typename REAL>
  inline void gen_runif(const uint32_t seed, const size_t len, REAL *x)
  {
    hiprandStatus_t st;
    hiprandGenerator_t gen;
    
    st = hiprandCreateGenerator(&gen, defs::gen_type);
    err::check_init(st);
    
    st = hiprandSetPseudoRandomGeneratorSeed(gen, seed);
    err::check_seed_set(st);
    st = generics::gpu_rand_unif(gen, x, len);
    err::check_generation(st);
    
    hiprandDestroyGenerator(gen);
  }
  
  
  
  template <typename REAL>
  inline void gen_rnorm(const uint32_t seed, const REAL mean, const REAL sd, const size_t len, REAL *x)
  {
    hiprandStatus_t st;
    hiprandGenerator_t gen;
    
    st = hiprandCreateGenerator(&gen, defs::gen_type);
    err::check_init(st);
    
    st = hiprandSetPseudoRandomGeneratorSeed(gen, seed);
    err::check_seed_set(st);
    st = generics::gpu_rand_norm(gen, x, len, mean, sd);
    err::check_generation(st);
    
    hiprandDestroyGenerator(gen);
  }
}


#endif
