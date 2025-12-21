#include "dmt/core/math/vec4.h"

#include <iostream>

#ifdef _WIN32
int wmain() {
#else
int main() {
#endif
  std::cout << "Hello Beautiful World" << std::endl;
  dmt::Vec4f const theVec{4, 0, 0, 0};
  dmt::Vec4f const theVec1{1, 0, 2, 0};
  float const theDot = dmt::dot(theVec, theVec1);
  std::cout << theDot << std::endl;
}