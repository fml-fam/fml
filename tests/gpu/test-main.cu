#define CATCH_CONFIG_RUNNER
#include "../catch.hpp"

#include <gpu/card.hh>
#include <gpu/gpuhelpers.hh>

std::shared_ptr<card> c;


int main(int argc, char *argv[])
{
  int num_failed_tests;
  // c = std::make_shared<card>(0);
  c = gpuhelpers::new_card(0);
  
  num_failed_tests = Catch::Session().run(argc, argv);
  
  c.reset();
  
  return num_failed_tests;
}
