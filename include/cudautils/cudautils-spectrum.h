#pragma once

#include "cudautils/cudautils-macro.h"
#include <platform/platform-utils.h>
// #include <platform/platform-cuda-utils.h>

#include <concepts>

namespace dmt {
    /** https://www.desmos.com/calculator/e0arbln6ip
     * @note This has a very low accuracy and I don't know why.
     */
    DMT_CORE_API DMT_CPU_GPU float blackbody(float lambdaNanometers, float TKelvin);

    DMT_CPU_GPU inline constexpr float    lambdaMin() { return 360.f; }
    DMT_CPU_GPU inline constexpr float    lambdaMax() { return 830.f; }
    DMT_CPU_GPU inline constexpr uint32_t numSpectrumSamples() { return 4u; }
    static_assert(numSpectrumSamples() > 0 && nextPOT(numSpectrumSamples()) == numSpectrumSamples());

    template <typename T>
    struct SOA;

    // TODO move elsewhere
    /** type which encapsulates a real values function with up to two float parameters  */
    class DMT_CORE_API FloatFunction2
    {
    public:
        using Func = float (*)(float _t, float _param0, float _param1);
        DMT_CPU_GPU FloatFunction2(float _param0, float _param1, Func _f) : func(_f), param0(_param0), param1(_param1)
        {
        }

        inline DMT_CPU_GPU float operator()(float _t) const { return func(_t, param0, param1); }

    private:
        Func  func;
        float param0;
        float param1;
    };

    // Sampled Spectrum and Sampled Wavelengths: Class Definitions ----------------------------------------------------
    struct DMT_CORE_API SampledSpectrum
    {
        friend struct SOA<SampledSpectrum>;

        SampledSpectrum() = default;
        DMT_CPU_GPU SampledSpectrum(ArrayView<float> values);

        float values[numSpectrumSamples()]{};
    };

    struct DMT_CORE_API SampledWavelengths
    {
        friend struct SOA<SampledWavelengths>;

        SampledWavelengths() = default;
        /**
         * After phenomena like light dispersion or any scattering event which causes two wavelengths to travel
         * different paths, we terminate all sampled wavelengths (ie setting their PDF value to 0) escept for
         * the one at index 0
         */
        DMT_CPU_GPU void terminateSecondary();
        DMT_CPU_GPU bool secondaryTerminated() const;

        float lambda[numSpectrumSamples()]{};
        float pdf[numSpectrumSamples()]{};
    };

    struct DMT_CORE_API ValueIndex
    {
        float   value;
        int32_t index;
    };

    // Sampled Spectrum and Sampled Wavelengths: Boilerplate ----------------------------------------------------------
    DMT_CORE_API DMT_CPU_GPU SampledSpectrum  operator+(SampledSpectrum const& spec0, SampledSpectrum const& spec1);
    DMT_CORE_API DMT_CPU_GPU SampledSpectrum  operator-(SampledSpectrum const& spec0, SampledSpectrum const& spec1);
    DMT_CORE_API DMT_CPU_GPU SampledSpectrum  operator*(SampledSpectrum const& spec0, SampledSpectrum const& spec1);
    DMT_CORE_API DMT_CPU_GPU SampledSpectrum  operator/(SampledSpectrum const& spec0, SampledSpectrum const& spec1);
    DMT_CORE_API DMT_CPU_GPU SampledSpectrum& operator+=(SampledSpectrum& spec0, SampledSpectrum const& spec1);
    DMT_CORE_API DMT_CPU_GPU SampledSpectrum& operator-=(SampledSpectrum& spec0, SampledSpectrum const& spec1);
    DMT_CORE_API DMT_CPU_GPU SampledSpectrum& operator*=(SampledSpectrum& spec0, SampledSpectrum const& spec1);
    DMT_CORE_API DMT_CPU_GPU SampledSpectrum& operator/=(SampledSpectrum& spec0, SampledSpectrum const& spec1);

    DMT_CORE_API DMT_CPU_GPU bool       hasNaN(SampledSpectrum const& spec);
    DMT_CORE_API DMT_CPU_GPU ValueIndex max(SampledSpectrum const& spec);
    DMT_CORE_API DMT_CPU_GPU ValueIndex min(SampledSpectrum const& spec);
    DMT_CORE_API DMT_CPU_GPU float      average(SampledSpectrum const& spec);

    // Sampled Spectrum and Sampled Wavelengths: Functions ------------------------------------------------------------
    DMT_CORE_API DMT_CPU_GPU SampledWavelengths
        sampleUniforms(float u, float _lambdaMin = lambdaMin(), float _lambdaMax = lambdaMax());
    DMT_CORE_API DMT_CPU_GPU SampledWavelengths sampleVisible(float u);
    DMT_CORE_API DMT_CPU_GPU SampledSpectrum    PDF(SampledWavelengths& wavelengths);

    // Spectrum Concept -----------------------------------------------------------------------------------------------
    // clang-format off
    template <typename T>
    concept SpectrumType = requires (T t) {
        { t(3.f) } -> std::same_as<float>;
        { t.maxValue() } -> std::same_as<float>;
        { t.sample(std::declval<SampledWavelengths>()) } -> std::same_as<SampledSpectrum>;
    };
    // clang-format on

    // Spectrum Classes: Constant Spectrum ----------------------------------------------------------------------------
    /**
     * represents a constant spectrum. not strictly necessary as we could store directly the float value, but who cares
     */
    struct DMT_CORE_API ConstantSpectrum
    {
    public:
        DMT_CPU_GPU                 ConstantSpectrum(float value);
        DMT_CPU_GPU float           operator()(float lambdanm) const;
        DMT_CPU_GPU float           maxValue() const;
        DMT_CPU_GPU SampledSpectrum sample(SampledWavelengths const& wavelengths) const;

    private:
        float m_value;
    };
    static_assert(SpectrumType<ConstantSpectrum>);

    /**
     * Class holding a buffer whose floating point values represent samples of a spectral distribution over the
     * specified range, taken at 1nm intervals. The typical size for this array is 830 - 360 = 470, meaning
     * `~1.84 kB` with float32
     */
    // TODO remake
    //struct DenselySampledSpectrum
    //{
    //public:
    //    DMT_CPU_GPU DenselySampledSpectrum(FloatFunction2      func,
    //                                       BaseMemoryResource* alloc,
    //                                       CudaStreamHandle    stream     = noStream,
    //                                       int32_t             _lambdaMin = lambdaMin(),
    //                                       int32_t             _lambdaMax = lambdaMax());

    //    /**
    //     * construct a denssely sampled spectrum from an existing spectrum. Useful if the starting spectrum is costly to evaluate
    //     * @note if you are executing dmt with a CUDA capable device, this should be created with `UnifiedMemoryResource` or passed by
    //     * value to the device (64 bytes anyways)
    //     * @note cannot allow construction from the device cause we don't know, due to divergence, how many active threads enter the
    //     * constructor. If constrution from the device is desired, use the construction which doesn't initialze the values
    //     */
    //    template <SpectrumType S>
    //    DMT_CPU DenselySampledSpectrum(S const&            spectrum,
    //                                   BaseMemoryResource* alloc,
    //                                   CudaStreamHandle    stream     = noStream,
    //                                   int32_t             _lambdaMin = lambdaMin(),
    //                                   int32_t             _lambdaMax = lambdaMax()) :
    //    m_values(sizeof(float), alloc, stream),
    //    m_lambdaMin(_lambdaMin),
    //    m_lambdaMax(_lambdaMax)
    //    {
    //        assert(m_lambdaMax > m_lambdaMin + 3); // at least 4 values
    //        int32_t const numel = m_lambdaMax - m_lambdaMin + 1;

    //        m_values.resize(numel);
    //        if (m_values.resource()->hostHasAccess())
    //        {
    //            m_values.lockForRead();
    //            for (int32_t index = 0; index < numel; ++index)
    //                m_values.operator[]<float>(index) = spectrum(index + m_lambdaMin);
    //            m_values.unlockForRead();
    //        }
    //        else
    //        {
    //            std::unique_ptr<float[]> data = std::make_unique<float[]>(numel);
    //            for (int32_t index = 0; index < numel; ++index)
    //                data[index] = spectrum(index + m_lambdaMin);
    //            m_values.copyFromHostAsync(data.get(), numel * sizeof(float));
    //            m_values.syncWithStream();
    //        }
    //    }

    //    // `SpectrumType` Interface -----------------------------------------------------------------------------------
    //    /** @warning If called from `__host__` and the `DynaArray` is not accessible by the host, a
    //     * `cudaMemcpyAsync` will be issued to fetch a single elemnent. Should be used only for testing purposes
    //     */
    //    DMT_CPU_GPU float           operator()(float lambdanm) const;
    //    DMT_CPU_GPU float           maxValue() const;
    //    DMT_CPU_GPU SampledSpectrum sample(SampledWavelengths const& wavelengths) const;

    //    // Other public Stuff -----------------------------------------------------------------------------------------
    //    DMT_CPU_GPU void scale(float factor);

    //private:
    //    DMT_CPU_GPU float*       getBuffer();
    //    DMT_CPU_GPU float const* getBuffer() const;

    //private:
    //    DynaArray m_values;
    //    int32_t   m_lambdaMin, m_lambdaMax;
    //};
    //static_assert(sizeof(DenselySampledSpectrum) == 64 && SpectrumType<DenselySampledSpectrum>);
} // namespace dmt

#if defined(DMT_CUDAUTILS_IMPL) || defined(DMT_CUDAUTILS_SPECTRUM_IMPL)
#include "cudautils-spectrum.cu"
#endif
