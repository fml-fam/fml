#define CATCH_CONFIG_RUNNER
#include "../catch.hpp"

#include <par/comm.hh>
#include <cstdio>

comm r;


int main(int argc, char *argv[])
{
  int num_failed_tests;
  r = comm();
  
  if (!r.rank0())
    fclose(stdout);
  
  num_failed_tests = Catch::Session().run(argc, argv);
  
  r.finalize();
  
  return num_failed_tests;
}
