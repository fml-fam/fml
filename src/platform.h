#ifndef FML_PLATFORM_H
#define FML_PLATFORM_H
#pragma once


// "portability"
#if (defined(__gnu_linux__) || defined(__linux__) || defined(__linux) || defined(linux))
#define OS_LINUX 1
#else
#define OS_LINUX 0
#endif

#if (defined(_WIN32) || defined(__WIN32__) || defined(_WIN64) || defined(__TOS_WIN__) || defined(__WINDOWS__))
#define OS_WINDOWS 1
#else
#define OS_WINDOWS 0
#endif

#if ((defined(__APPLE__) && defined(__MACH__)) || macintosh || Macintosh)
#define OS_MAC 1
#else
#define OS_MAC 0
#endif

#if defined(__FreeBSD__)
#define OS_FREEBSD 1
#else
#define OS_FREEBSD 0
#endif

#if (defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__) || defined(__bsdi__) || defined(__DragonFly__))
#define OS_BSD 1
#else
#define OS_BSD 0
#endif

#if (defined(__sun) || defined(sun))
#define OS_SOLARIS 1
#else
#define OS_SOLARIS 0
#endif

// why the hell not
#if (defined(__GNU__) || defined(__gnu_hurd__))
#define OS_HURD 1
#else
#define OS_HURD 0
#endif

#if (OS_BSD || OS_HURD || OS_LINUX || OS_MAC || OS_SOLARIS)
#define OS_NIX 1
#else
#define OS_NIX 0
#endif


#endif
