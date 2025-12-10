#ifndef DMT_TRAINING_COOPERATIVE_GROUPS_THE_MACROS_H
#define DMT_TRAINING_COOPERATIVE_GROUPS_THE_MACROS_H

#define WANSI_RED L"\033[31m"
#define WANSI_YLW L"\033[93m"
#define WANSI_RST L"\033[0m"
#define ANSI_RED "\033[31m"
#define ANSI_YLW "\033[93m"
#define ANSI_RST "\033[0m"

#define CUDA_SUCC(cudStatus)                                                                                  \
    [err = (cudStatus)]() {                                                                                   \
        if (err != ::cudaSuccess)                                                                             \
        {                                                                                                     \
            std::cerr << ANSI_RED "[CUDA Error] " << cudaGetErrorName(err) << ": " << cudaGetErrorString(err) \
                      << ANSI_RST << std::endl;                                                               \
            return false;                                                                                     \
        }                                                                                                     \
        return true;                                                                                          \
    }()
#define CU_SUCC(cuStatus)                                                                              \
    [err = (cuStatus)]() {                                                                             \
        if (err != ::CUDA_SUCCESS)                                                                     \
        {                                                                                              \
            char const* name    = "the name";                                                          \
            char const* message = "the message";                                                       \
            cuGetErrorName(err, &name);                                                                \
            cuGetErrorString(err, &message);                                                           \
            std::cerr << ANSI_RED "[CUDA Error] " << name << ": " << message << ANSI_RST << std::endl; \
            return false;                                                                              \
        }                                                                                              \
        return true;                                                                                   \
    }()

#endif // DMT_TRAINING_COOPERATIVE_GROUPS_THE_MACROS_H
