// Include local CUDA header files.
#include <cudaTest.h>

// Include C++ header files.
#include <iostream>
#include <random>
#include <source_location>
#include <string>

import platform;

inline constexpr uint32_t N = 10000;

// Single precision A X plus Y
// saxpy: z_i = a \times x_i + y_i
void cpu_saxpy_vect(float const* x, float const* y, float a, float* z, uint32_t n)
{
    for (int i = 0; i < n; i++)
    {
        z[i] = a * x[i] + y[i];
    }
}

int main()
{
    static constexpr uint32_t             threshold = 10e-4;
    std::random_device                    seed;
    std::mt19937                          gen{seed()};
    std::uniform_real_distribution<float> random;
    float                                 A[N];
    float                                 B[N];
    float                                 C[N], C_cpu[N];
    float const                           scalar = random(gen);
    dmt::ConsoleLogger                    logger = dmt::ConsoleLogger::create();

    for (int i = 0; i < N; i++)
    {
        A[i]     = random(gen);
        B[i]     = random(gen);
        C_cpu[i] = 0;
        C[i]     = 0;
    }

    logger.log("Starting saxpy computation on the CPU...");
    cpu_saxpy_vect(A, B, scalar, C_cpu, N);
    logger.log("Done!");

    logger.log("Starting saxpy computation on the GPU...");
    dmt::kernel(A, B, scalar, C, N);
    logger.log("Done! Showing first 4 elements of each result:");
    logger.log("CPU[0:3] = ");

    std::string str;
    for (uint32_t i = 0; i != 4; ++i)
    {
        str += std::to_string(C_cpu[i]) + ' ';
    }

    logger.log("  {}", {dmt::StrBuf(str)});
    logger.log("GPU[0:3] = ");
    str.clear();

    for (uint32_t i = 0; i != 4; ++i)
    {
        str += std::to_string(C[i]) + ' ';
    }
    logger.log("  {}\n", {dmt::StrBuf(str)});

    bool  error = false;
    float diff  = 0.0;
    for (int i = 0; i < N; i++)
    {
        diff = abs(C[i] - C_cpu[i]);
        if (diff > threshold)
        {
            error = true;
            logger.log("{} {} {} {}", {i, diff, C[i], C_cpu[i]});
        }
    }

    if (error)
        logger.log("The Results are Different!");
    else
        logger.log("The Results match!");

    logger.log("Programm Finished!");
}