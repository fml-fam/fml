#include "../catch.hpp"

#include <_internals/arraytools/src/arraytools.hpp>
#include <gpu/card.hh>

using namespace arraytools;

extern std::shared_ptr<fml::card> c;


TEMPLATE_TEST_CASE("memory operations", "[card]", float, double)
{
  size_t len = sizeof(TestType);
  TestType d_cpu;
  TestType *d_gpu = (TestType*) c->mem_alloc(len);
  
  d_cpu = (TestType) 1.f;
  c->mem_cpu2gpu(d_gpu, &d_cpu, len);
  c->mem_set(d_gpu, 0, len);
  c->mem_gpu2cpu(&d_cpu, d_gpu, len);
  c->mem_free(d_gpu);
  
  REQUIRE( fltcmp::eq(d_cpu, 0) );
}
