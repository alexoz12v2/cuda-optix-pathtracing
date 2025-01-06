#pragma once

#if !defined(DMT_INTERFACE_AS_HEADER)
#define DMT_MODULE_EXPORT export namespace
#else
#define DMT_MODULE_EXPORT namespace
#endif

#if defined(DMT_COMPILER_MSVC)
#define DMT_INTERFACE __declspec(novtable)
#else
#define DMT_INTERFACE
#endif

// TODO reg, interface dll