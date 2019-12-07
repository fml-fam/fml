#ifndef FML_BENCH_H
#define FML_BENCH_H


#include <chrono>
#include <climits>
#include <cstdarg>
#include <cstdio>
#include <string>
#include <vector>


class bench
{
  public:
    bench();
    bench(unsigned int maxiter);
    
    void start();
    void stop();
    void reset();
    
    void print_header(const char *fmt, ...);
    void report(std::string msg="", bool reset_timer=true);
    
    bool proceed();
  
  private:
    std::chrono::high_resolution_clock::time_point query_clock() const;
    std::chrono::high_resolution_clock::time_point _start;
    double _elapsed;
    unsigned int _iters;
    unsigned int _maxiter;
    
    static const int print_width = 50;
    void print_hline();
};



inline bench::bench()
{
  _maxiter = UINT_MAX;
  reset();
}



inline bench::bench(unsigned int maxiter)
{
  _maxiter = maxiter;
  reset();
}



inline void bench::start()
{
  _start = query_clock();
}


inline void bench::stop()
{
  std::chrono::duration<double> elapsed = query_clock() - _start;
  _elapsed += elapsed.count();
  _iters++;
}



inline void bench::reset()
{
  _elapsed = 0.0;
  _iters = 0;
}



inline void bench::print_header(const char *fmt, ...)
{
  print_hline();
  
  printf("| ");
  char s[print_width];
  va_list args;
  va_start(args, fmt);
  int nchars = vsnprintf(s, print_width, fmt, args);
  printf("%s", s);
  va_end(args);
  
  for (int i=0; i<std::max(print_width-nchars, 1); i++)
    putchar(' ');
  
  printf(" |\n");
  print_hline();
}


inline void bench::report(std::string msg, bool reset_timer)
{
  printf("| %-15s", msg.c_str());
  printf("%10f / %-5d = %10f avg", _elapsed, _iters, _elapsed/_iters);
  printf(" |\n");
  
  if (reset_timer)
    reset();
}



inline bool bench::proceed()
{
  return (_iters < _maxiter);
}



inline std::chrono::high_resolution_clock::time_point bench::query_clock() const
{
  return std::chrono::high_resolution_clock::now();
}


inline void bench::print_hline()
{
  printf("| ");
  for (int i=0; i<print_width; i++)
    putchar('-');
  
  printf(" |\n");
}


#endif
