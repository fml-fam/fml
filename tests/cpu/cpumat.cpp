#include "../catch.hpp"

#include <cpu/cpumat.hh>


TEST_CASE("inheriting memory", "[cpumat]")
{
  len_t m = 2;
  len_t n = 3;
  
  float *data = (float*) malloc(m*n*sizeof(*data));
  cpumat<float> x(data, m, n);
  x.fill_eye();
  x.~cpumat();
  REQUIRE( data[0] == 1.0f );
  
  cpumat<float> y;
  y.set(data, m, n);
  y.fill_zero();
  y.~cpumat();
  REQUIRE( data[0] == 0.0f );
  
  free(data);
}
 


TEST_CASE("resize", "[cpumat]")
{
  len_t m = 3;
  len_t n = 2;
  
  cpumat<float> x;
  x.resize(m, n);
  x.fill_eye();
  
  REQUIRE( x.nrows() == m );
  REQUIRE( x.ncols() == n );
  
  REQUIRE( (x.data_ptr())[0] == 1.0f );
  REQUIRE( (x.data_ptr())[1] == 0.0f );
}



TEST_CASE("scale", "[cpumat]")
{
  len_t m = 3;
  len_t n = 2;
  
  cpumat<float> x(m, n);
  x.fill_one();
  
  x.scale(3.0f);
  REQUIRE( (x.data_ptr())[0] == 3.0f );
  REQUIRE( (x.data_ptr())[1] == 3.0f );
}
