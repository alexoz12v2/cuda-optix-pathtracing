#include "dmtmacros.h"
#include <platform/platform.h>
#include <platform/platform-cuda-utils.h>
#include <platform/platform-cuda-utils.cuh>
#include <cudautils/cudautils.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <bit>
#include <limits>
#include <unordered_map>
#include <memory_resource>
#include <string>
#include <string_view>
#include <charconv>
#include <system_error>

#include <cstdio>
#include <cstdint>

using dmt::sid_t;
using dmt::operator""_side;

static void printSpectrum(dmt::DenselySampledSpectrum const& spectrum)
{
    dmt::AppContextJanitor j;
    std::string            str = "\t";
    char                   buf[32];
    str.reserve(128);
    j.actx.log("{");
    for (float lambda : dmt::Rangef(dmt::lambdaMin(), dmt::lambdaMax() + 0.1f, 1.f))
    {
        auto charRes = std::to_chars(buf, buf + 10, lambda, std::chars_format::fixed, 3);
        assert(charRes.ec == std::errc());
        str += std::string_view{buf, static_cast<size_t>(charRes.ptr - buf)};
        str += ": ";
        charRes = std::to_chars(buf, buf + 10, spectrum(lambda), std::chars_format::scientific, 3);
        assert(charRes.ec == std::errc());
        str += std::string_view{buf, static_cast<size_t>(charRes.ptr - buf)};
        if (str.length() >= 128)
        {
            j.actx.log("{}", {str});
            str.clear();
            str += "\t";
        }
        else
        {
            str += " | ";
        }
    }
    j.actx.log("{}", {str});
    j.actx.log("}");
}

static std::string printSpectrumSamples(dmt::SampledSpectrum const& spectrum)
{
    static char buf[64];
    std::string str = "{ ";
    str.reserve(64);
    for (float value : spectrum.values)
    {
        auto charRes = std::to_chars(buf, buf + 64, value, std::chars_format::scientific, 3);
        assert(charRes.ec == std::errc());
        str += std::string_view(buf, static_cast<size_t>(charRes.ptr - buf));
        str += ", ";
    }
    str[str.length() - 2] = ' ';
    str[str.length() - 1] = '}';
    return str;
}

__global__ void kSpectrumAt(dmt::DenselySampledSpectrum* spectrum, float value, float* result)
{
    int32_t gid = dmt::globalThreadIndex();
    if (gid == 0)
        *result = (*spectrum)(value);
}

__global__ void kSpectrumSample(dmt::DenselySampledSpectrum* spectrum,
                                dmt::SampledWavelengths      wavelengths,
                                dmt::SampledSpectrum*        result)
{
    int32_t gid = dmt::globalThreadIndex();
    if (gid == 0)
        *result = spectrum->sample(wavelengths);
}

static void testSpectrum(dmt::BaseMemoryResource* unified, dmt::BaseMemoryResource* mem)
{
    dmt::AppContextJanitor j;
    dmt::ConstantSpectrum  constantSpectrum(3.f);
    cudaError_t            cudaStatus = ::cudaSuccess;
    dmt::FloatFunction2    blackbody673K{673.15f, 0.f, [](float _t, float _param0, float _param1) {
        return dmt::blackbody(_t, _param0);
    }};

    j.actx.log("------------------- Testing __host__ Spectrum Creation from another spectrum -----------------------");
    dmt::DenselySampledSpectrum fromConstant{constantSpectrum, mem};
    printSpectrum(fromConstant);
    j.actx.log("------------------- Testing __host__ Spectrum Creation from a Function -----------------------------");
    dmt::DenselySampledSpectrum fromFunctionHost{blackbody673K, mem};
    printSpectrum(fromFunctionHost);
    j.actx.log("------------------- Testing __host__ Spectrum Computation of Max -----------------------------------");
    float const expected = 1980.50895344f;
    float const actual   = fromFunctionHost.maxValue();
    j.actx.log("Max value: expected {}, actual {}", {expected, actual});
    j.actx.log("------------------- Testing __host__ Spectrum Computation of Samples -------------------------------");
    dmt::SampledWavelengths samples = dmt::sampleUniforms(0.5f);
    float expectedSamples[dmt::numSpectrumSamples()]{0.400375138067f, 60.7999945299f, 1980.50895344f, 0.000174271418803f};
    dmt::SampledSpectrum actualSamples = fromFunctionHost.sample(samples);
    j.actx.log("Actual Samples:   {}", {printSpectrumSamples(actualSamples)});
    j.actx.log("Expected Samples: {{ {}, {}, {}, {} }}",
               {expectedSamples[0], expectedSamples[1], expectedSamples[2], expectedSamples[3]});
    j.actx.log("------------------- Testing __device__ Spectrum operator() -----------------------------------------");
    j.actx.log(
        "  (also testing DynaArray's copy constructor, as we are copying the DenselySampled Spectrum to __managed__ "
        "memory)");
    dmt::DenselySampledSpectrum* managedSpectrum = nullptr;
    cudaStatus = cudaMallocManaged(&managedSpectrum, sizeof(dmt::DenselySampledSpectrum));
    assert(cudaStatus == ::cudaSuccess);

    static constexpr size_t numResults = 4;
    float*                  results    = nullptr;
    cudaStatus = cudaMallocManaged(&results, numResults * sizeof(float) + sizeof(dmt::SampledSpectrum));
    static_assert(alignof(dmt::SampledSpectrum) == 4);
    assert(cudaStatus == ::cudaSuccess);

    j.actx.log("  calling copy constructor of DenselySampledSpectrum");
    std::construct_at(managedSpectrum, fromFunctionHost);
    kSpectrumAt<<<1, 32>>>(managedSpectrum, 712.5f, &results[0]);
    cudaStatus = cudaGetLastError();
    assert(cudaStatus == ::cudaSuccess);
    cudaStatus = cudaDeviceSynchronize();
    assert(cudaStatus == ::cudaSuccess);
    j.actx.log("  operator() result: Expected {}, Actual {}, __host__ {}",
               {expectedSamples[1], results[0], actualSamples.values[1]});

    j.actx.log("------------------- Testing __device__ Spectrum Sample function ------------------------------------");
    dmt::SampledSpectrum* managedSamples = std::construct_at(std::bit_cast<dmt::SampledSpectrum*>(results + 1));
    kSpectrumSample<<<1, 32>>>(managedSpectrum, samples, managedSamples);
    cudaStatus = cudaGetLastError();
    assert(cudaStatus == ::cudaSuccess);
    cudaStatus = cudaDeviceSynchronize();
    assert(cudaStatus == ::cudaSuccess);
    printSpectrumSamples(*managedSamples);

    j.actx.log("------------------- Testing __device__ Spectrum Max computation ------------------------------------");
    j.actx.log("------------------- Testing __device__ Spectrum Construction from function -------------------------");

    // cleanup
    cudaStatus = cudaFree(results);
    assert(cudaStatus == ::cudaSuccess);

    std::destroy_at(managedSpectrum);
    cudaStatus = cudaFree(managedSpectrum);
    assert(cudaStatus == ::cudaSuccess);
}

int32_t main()
{
    printf("Adding elements in the map from the host\n");
    dmt::AppContext actx;
    dmt::ctx::init(actx);

    auto helloInfo = dmt::cudaHello(&actx.mctx());

    auto* resource = dmt::UnifiedMemoryResource::create();
    { // resources on the unified resource must not outlive them
        dmt::BuddyResourceSpec spec{
            .pmctx        = &actx.mctx(),
            .pHostMemRes  = std::pmr::get_default_resource(),
            .maxPoolSize  = 4ULL << 20, // 1ULL << 30,
            .minBlockSize = 256,
            .minBlocks    = (2ULL << 20) / 256,
            .deviceId     = helloInfo.device,
        };
        dmt::AllocBundle buddyBundle(resource, dmt::EMemoryResourceType::eHost, dmt::EMemoryResourceType::eHostToDevMemMap, &spec);

        testSpectrum(resource, buddyBundle.pMemRes);
    }

    dmt::UnifiedMemoryResource::destroy(resource);
    dmt::ctx::unregister();
    getc(stdin);
}