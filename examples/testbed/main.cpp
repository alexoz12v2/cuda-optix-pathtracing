module;

#include <iostream>
#include <numbers>
#include <string_view>

module platform;

int main()
{
    using namespace std::string_view_literals;
    dmt::CircularOStringStream oss;
    char const*                formatStr = "this is a \\{} {} string. Pi: {}, 4 pi: {}, 1000 == {}, thuthy: {}\n";
    float                      pi        = std::numbers::pi_v<float>;
    bool                       b         = true;
    int                        thou      = 1000;
    std::string_view           arg{"format"};
    oss.logInitList(formatStr, {arg, pi, dmt::StrBuf(pi, "%.5f"), thou, b});
    std::cout << oss.str() << std::endl;

    dmt::ConsoleLogger logger;
    logger.log("Hello World from logger");
    logger.warn("Hello Warn from logger");
    logger.error("Hello error from logger");
    logger.log("Hello World from logger");
    logger.log("Hello {} from logger", {"world"sv});
    logger.trace("I shall not be seen");
}