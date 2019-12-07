#include <bench.hh>

#include <cpu/cpumat.hh>
#include <cpu/linalg.hh>
#include <gpu/gpumat.hh>
#include <gpu/linalg.hh>

len_t m;
len_t n;
len_t nrhs;



template <typename REAL>
static void gemm_cpu(bench &b)
{
  cpumat<REAL> x(m, n);
  cpumat<REAL> y(n, nrhs);
  cpumat<REAL> z(m, nrhs);
  x.fill_runif();
  y.fill_runif();
  z.fill_runif();
  
  while (b.proceed())
  {
    b.start();
    linalg::matmult(false, false, (REAL) 1.f, x, y, z);
    b.stop();
  }
  
  b.report("cpu");
}



template <typename REAL>
static void gemm_gpu(bench &b)
{
  b.start();
  auto c = std::make_shared<card>(0);
  b.stop();
  b.report("gpu init");
  
  gpumat<REAL> x(c, m, n);
  gpumat<REAL> y(c, n, nrhs);
  gpumat<REAL> z(c, m, nrhs);
  
  REAL start = (REAL) 1.f;
  REAL stop = (REAL) 20.f;
  
  x.fill_linspace(start, stop);
  y.fill_linspace(start, stop);
  z.fill_linspace(start, stop);
  
  while (b.proceed())
  {
    b.start();
    linalg::matmult(false, false, (REAL) 1.f, x, y, z);
    b.stop();
  }
  
  b.report("gpu");
}



int main(int argc, char **argv)
{
  m = 1000000;
  n = 250;
  nrhs = 1;
  bench b(2);
  
  double problem_size = sizeof(float) * m * n / 1024 / 1024;
  b.print_header("GEMM: %dx%d * %dx%d (%.3f MiB)", m, n, n, nrhs, problem_size);
  
  gemm_cpu<float>(b);
  gemm_gpu<__half>(b);
  
  return 0;
}
