import testdmt;

#include <iostream>

int main()
{
    std::cout << "Hello, the result is " << dmt::TestMath::add(4, 3) << std::endl;
    dmt::TestMath::printStuff();
}