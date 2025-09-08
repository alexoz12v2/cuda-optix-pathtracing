#pragma once

float const PI = 3.14159265f;

// Tausworth parameters
unsigned const TAUSWORTHE_NUM_THREADS             = 256;
unsigned const TAUSWORTHE_NUM_BLOCKS              = 16;
unsigned const TAUSWORTHE_TOTAL_NUM_THREADS       = TAUSWORTHE_NUM_THREADS * TAUSWORTHE_NUM_BLOCKS;
unsigned const TAUSWORTHE_NUM_SEEDS_PER_GENERATOR = 4;
unsigned const TAUSWORTHE_NUM_SEEDS = TAUSWORTHE_NUM_BLOCKS * TAUSWORTHE_NUM_THREADS * TAUSWORTHE_NUM_SEEDS_PER_GENERATOR;
unsigned const TAUSWORTHE_NUM_OUTPUTS_PER_RUN           = 2048;
unsigned const TAUSWORTHE_SEED_SIZE                     = TAUSWORTHE_NUM_THREADS * 4;
unsigned const TAUSWORTHE_TOTAL_SEED_SIZE               = TAUSWORTHE_SEED_SIZE * TAUSWORTHE_NUM_BLOCKS;
unsigned const TAUSWORTHE_NUM_RANDOM_NUMBERS_PER_THREAD = TAUSWORTHE_NUM_OUTPUTS_PER_RUN * 8;
unsigned const TAUSWORTHE_NUM_RANDOM_NUMBERS_PER_BLOCK = TAUSWORTHE_NUM_RANDOM_NUMBERS_PER_THREAD * TAUSWORTHE_NUM_THREADS;
unsigned const TAUSWORTHE_OUTPUT_SIZE = TAUSWORTHE_NUM_RANDOM_NUMBERS_PER_BLOCK * TAUSWORTHE_NUM_BLOCKS;

#if defined(DMT_CUDAUTILS_IMPLEMENTATION)
    #if defined(__CUDA_ARCH__)
__device__ unsigned TausStep(unsigned& z, int S1, int S2, int S3, unsigned M)
{
    unsigned b = (((z << S1) ^ z) >> S2);
    return z   = (((z & M) << S3) ^ b);
}

__device__ unsigned LCGStep(unsigned& z) { return z = (1664525 * z + 1013904223); }

// Uniform, need to do box muller on this
__device__ float getRandomValueTauswortheUniform(unsigned& z1, unsigned& z2, unsigned& z3, unsigned& z4)
{
    unsigned taus = TausStep(z1, 13, 19, 12, 4294967294UL) ^ TausStep(z2, 2, 25, 4, 4294967288UL) ^
                    TausStep(z3, 3, 11, 17, 4294967280UL);
    unsigned lcg = LCGStep(z4);

    return 2.3283064365387e-10f * (taus ^ lcg); // taus+
}

__device__ void boxMuller(float u1, float u2, float& uo1, float& uo2)
{
    float z1 = sqrtf(-2.0f * logf(u1));
    float s1 = sinf(2.0f * PI * u2);
    float s2 = cosf(2.0f * PI * u2);
    uo1      = z1 * s1;
    uo2      = z1 * s2;
}
__device__ float getRandomValueTausworthe(unsigned& z1, unsigned& z2, unsigned& z3, unsigned& z4, float& temporary, unsigned phase)
{
    if (phase & 1)
    {
        // Return second value of pair
        return temporary;
    }
    else
    {
        float t1, t2, t3;
        // Phase is even, generate pair, return first of values, store second
        t1 = getRandomValueTauswortheUniform(z1, z2, z3, z4);
        t2 = getRandomValueTauswortheUniform(z1, z2, z3, z4);
        boxMuller(t1, t2, t3, temporary);
        return t3;
    }
}

__device__ void generateRandomNumbers_tausworthe(unsigned* globalPool, float* output)
{
    unsigned z1, z2, z3, z4;
    float    temporary;

    // Initialise tausworth with seeds
    unsigned address = __mul24(blockIdx.x, TAUSWORTHE_NUM_THREADS) + threadIdx.x;
    z1               = globalPool[address];
    z2               = globalPool[TAUSWORTHE_TOTAL_NUM_THREADS + address];
    z3               = globalPool[2 * TAUSWORTHE_TOTAL_NUM_THREADS + address];
    z4               = globalPool[3 * TAUSWORTHE_TOTAL_NUM_THREADS + address];

    // Loop generating outputs repeatedly
    for (int loop = 0; loop < TAUSWORTHE_NUM_OUTPUTS_PER_RUN; loop++)
    {
        unsigned intermediate_address = __mul24(loop, 8 * TAUSWORTHE_TOTAL_NUM_THREADS) +
                                        __mul24(8 * TAUSWORTHE_NUM_THREADS, blockIdx.x) + threadIdx.x;

        for (int i = 0; i < 8; i++)
        {
            output[intermediate_address + i * TAUSWORTHE_NUM_THREADS] = getRandomValueTausworthe(z1, z2, z3, z4, temporary, i);
        }
    }

    // Store seeds for later use
    globalPool[threadIdx.x]                                = z1;
    globalPool[TAUSWORTHE_TOTAL_NUM_THREADS + address]     = z2;
    globalPool[2 * TAUSWORTHE_TOTAL_NUM_THREADS + address] = z3;
    globalPool[3 * TAUSWORTHE_TOTAL_NUM_THREADS + address] = z4;
}
    #endif
#endif