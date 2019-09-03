#include <gpu/card.hh>
#include <gpu/gpumat.hh>
#include <gpu/linalg.hh>

#include <gpu/gpuhelpers.hh>


int main()
{
  auto c = gpuhelpers::new_card(0);
  c->info();
  
  len_t n = 5;
  gpumat<float> x(c, n, n);
  x.fill_linspace(1.f, (float) n*n);
  
  gpumat<float> y(c, n, n);
  y.fill_linspace(1.f, (float) n*n);
  
  gpumat<__half> xh, yh;
  gpuhelpers::copy(x, xh);
  gpuhelpers::copy(y, yh);
  
  gpumat<__half> zh = linalg::matmult(false, false, (__half)1.f, xh, yh);
  gpumat<float> z;
  
  gpuhelpers::copy(zh, z);
  z.info();
  z.print(0);
  
  return 0;
}
