module;

#include <iostream>

export module testdmt;

namespace dmt
{
export class TestMath
{
public:
    static int  add(int a, int b);
    static void printStuff()
    {
        std::cout << "Hello darkness my old Friend\n";
    }
};
} // namespace dmt