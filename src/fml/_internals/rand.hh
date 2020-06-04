// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML__INTERNALS_RAND_H
#define FML__INTERNALS_RAND_H
#pragma once


#include <ctime>

#include "platform.h"

#if OS_WINDOWS
#include <process.h>
#elif OS_NIX
#include <sys/types.h>
#include <unistd.h>
#endif


namespace fml
{
  namespace rand
  {
    namespace
    {
      // Robert Jenkins' 96-bit mix function
      inline uint32_t mix_96(uint32_t a, uint32_t b, uint32_t c)
      {
        a=a-b;  a=a-c;  a=a^(c >> 13);
        b=b-c;  b=b-a;  b=b^(a << 8);
        c=c-a;  c=c-b;  c=c^(b >> 13);
        a=a-b;  a=a-c;  a=a^(c >> 12);
        b=b-c;  b=b-a;  b=b^(a << 16);
        c=c-a;  c=c-b;  c=c^(b >> 5);
        a=a-b;  a=a-c;  a=a^(c >> 3);
        b=b-c;  b=b-a;  b=b^(a << 10);
        c=c-a;  c=c-b;  c=c^(b >> 15);
        
        return c;
      }
    }
    
    
    
    inline uint32_t get_seed()
    {
      uint32_t pid;
      uint32_t ret;
      
    #if OS_WINDOWS
      pid = _getpid();
    #elif OS_NIX
      pid = (uint32_t) getpid();
    #else
      #error "Unable to get PID"
    #endif
      
      ret = mix_96((uint32_t) time(NULL), (uint32_t) clock(), pid);
      
      return ret;
    }
  }
}


#endif
