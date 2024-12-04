#pragma once

#include <concepts>
#include <fff/fff.h>
#include <initializer_list>

namespace dmt
{

int testWork();

void resetMockHistory();

} // namespace dmt

// Converts argument to a string
#define STRINGIZE(arg)  STRINGIZE1(arg)
#define STRINGIZE1(arg) STRINGIZE2(arg)
#define STRINGIZE2(arg) #arg

// Concatenates two tokens
#define CONCATENATE(arg1, arg2)  CONCATENATE1(arg1, arg2)
#define CONCATENATE1(arg1, arg2) CONCATENATE2(arg1, arg2)
#define CONCATENATE2(arg1, arg2) arg1##arg2

// Count the number of arguments
#define PP_NARG(...)  PP_NARG_(__VA_ARGS__, PP_RSEQ_N())
#define PP_NARG_(...) PP_ARG_N(__VA_ARGS__)
#define PP_ARG_N(_1,  \
                 _2,  \
                 _3,  \
                 _4,  \
                 _5,  \
                 _6,  \
                 _7,  \
                 _8,  \
                 _9,  \
                 _10, \
                 _11, \
                 _12, \
                 _13, \
                 _14, \
                 _15, \
                 _16, \
                 _17, \
                 _18, \
                 _19, \
                 _20, \
                 _21, \
                 _22, \
                 _23, \
                 _24, \
                 _25, \
                 _26, \
                 _27, \
                 _28, \
                 _29, \
                 _30, \
                 _31, \
                 _32, \
                 _33, \
                 _34, \
                 _35, \
                 _36, \
                 _37, \
                 _38, \
                 _39, \
                 _40, \
                 _41, \
                 _42, \
                 _43, \
                 _44, \
                 _45, \
                 _46, \
                 _47, \
                 _48, \
                 _49, \
                 _50, \
                 _51, \
                 _52, \
                 _53, \
                 _54, \
                 _55, \
                 _56, \
                 _57, \
                 _58, \
                 _59, \
                 _60, \
                 _61, \
                 _62, \
                 _63, \
                 N,   \
                 ...) \
    N
#define PP_RSEQ_N()                                                                                                   \
    63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36,   \
        35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, \
        7, 6, 5, 4, 3, 2, 1, 0

// Define CALL_RESETn macros
#define CALL_RESET1(a)             RESET_FAKE(a)
#define CALL_RESET2(a, b)          RESET_FAKE(a) RESET_FAKE(b)
#define CALL_RESET3(a, b, c)       RESET_FAKE(a) RESET_FAKE(b) RESET_FAKE(c)
#define CALL_RESET4(a, b, c, d)    RESET_FAKE(a) RESET_FAKE(b) RESET_FAKE(c) RESET_FAKE(d)
#define CALL_RESET5(a, b, c, d, e) RESET_FAKE(a) RESET_FAKE(b) RESET_FAKE(c) RESET_FAKE(d) RESET_FAKE(e)
#define CALL_RESET6(a, b, c, d, e, f) \
    RESET_FAKE(a) RESET_FAKE(b) RESET_FAKE(c) RESET_FAKE(d) RESET_FAKE(e) RESET_FAKE(f)
#define CALL_RESET7(a, b, c, d, e, f, g) \
    RESET_FAKE(a) RESET_FAKE(b) RESET_FAKE(c) RESET_FAKE(d) RESET_FAKE(e) RESET_FAKE(f) RESET_FAKE(g)
#define CALL_RESET8(a, b, c, d, e, f, g, h) \
    RESET_FAKE(a) RESET_FAKE(b) RESET_FAKE(c) RESET_FAKE(d) RESET_FAKE(e) RESET_FAKE(f) RESET_FAKE(g) RESET_FAKE(h)

// General CALL_RESET macro
#define CALL_RESET(...) CONCATENATE(CALL_RESET, PP_NARG(__VA_ARGS__))(__VA_ARGS__)
