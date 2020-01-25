// This file is part of fml which is released under the Boost Software
// License, Version 1.0. See accompanying file LICENSE or copy at
// https://www.boost.org/LICENSE_1_0.txt

#ifndef FML__INTERNALS_PRINT_H
#define FML__INTERNALS_PRINT_H
#pragma once


#if (!defined(FML_PRINT_STD) && !defined(FML_PRINT_R))
  #define FML_PRINT_STD
#endif



#if (defined(FML_PRINT_STD))
  #include <cstdarg>
  #include <cstdio>
#elif defined(FML_PRINT_R)
  #include <R_ext/Print.h>
  #include <cstdarg>
#endif



namespace fml
{
  namespace print
  {
    inline void putchar(const char c)
    {
      #if (defined(FML_PRINT_STD))
        std::putchar(c);
      #elif (defined(FML_PRINT_R))
        Rprintf("%c", c);
      #endif
    }
    
    
    
    inline void vprintf(const char *fmt, va_list args)
    {
      #if (defined(FML_PRINT_STD))
        std::vprintf(fmt, args);
      #elif (defined(FML_PRINT_R))
        Rvprintf(fmt, args);
      #endif
    }
    
    
    
    inline void printf(const char *fmt, ...)
    {
      va_list args;
      va_start(args, fmt);
      fml::print::vprintf(fmt, args);
      va_end(args);
    }
  }
}


#endif
