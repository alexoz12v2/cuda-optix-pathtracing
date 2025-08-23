#include <bit>
#include <algorithm>
#include <iostream>

#if (DMT_OS_WINDOWS)
int wmain()
{
#else
int main()
{
#endif
    std::cout << "Hello Cruel World" << std::endl;
}