#include <fml/gpu.hh>


int main()
{
  auto c = fml::gpuhelpers::new_card(0);
  c->info();
  
  len_t n = 5;
  fml::gpumat<float> x(c, n, n);
  x.fill_linspace(1.f, (float) n*n);
  
  fml::gpumat<float> y(c, n, n);
  y.fill_linspace(1.f, (float) n*n);
  
  fml::gpumat<__half> xh(c), yh(c);
  fml::gpuhelpers::gpu2gpu(x, xh);
  fml::gpuhelpers::gpu2gpu(y, yh);
  
  fml::gpumat<__half> zh = fml::linalg::matmult(false, false, (__half)1.f, xh, yh);
  zh.info();
  zh.print(0);
  
  fml::gpumat<float> z(c);
  fml::gpuhelpers::gpu2gpu(zh, z);
  z.info();
  z.print(0);
  
  return 0;
}
