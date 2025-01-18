#pragma once

#include "dmtmacros.h"
#include <platform/platform-utils.h>

#include <concepts>

namespace dmt {
    DMT_CPU_GPU float blackbody(float lambdaNanometers, float TKelvin);

    DMT_CPU_GPU inline constexpr float    lambdaMin() { return 360.f; }
    DMT_CPU_GPU inline constexpr float    lambdaMax() { return 830.f; }
    DMT_CPU_GPU inline constexpr uint32_t numSpectrumSamples() { return 4u; }
    static_assert(numSpectrumSamples() > 0 && nextPOT(numSpectrumSamples()) == numSpectrumSamples());

    template <typename T>
    struct SOA;

    // Sampled Spectrum and Sampled Wavelengths: Class Definitions ----------------------------------------------------
    struct SampledSpectrum
    {
        friend struct SOA<SampledSpectrum>;

        SampledSpectrum() = default;
        DMT_CPU_GPU SampledSpectrum(ArrayView<float> values);

        float values[numSpectrumSamples()]{};
    };

    struct SampledWavelengths
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

    struct ValueIndex
    {
        float   value;
        int32_t index;
    };

    // Sampled Spectrum and Sampled Wavelengths: Boilerplate ----------------------------------------------------------
    DMT_CPU_GPU SampledSpectrum  operator+(SampledSpectrum const& spec0, SampledSpectrum const& spec1);
    DMT_CPU_GPU SampledSpectrum  operator-(SampledSpectrum const& spec0, SampledSpectrum const& spec1);
    DMT_CPU_GPU SampledSpectrum  operator*(SampledSpectrum const& spec0, SampledSpectrum const& spec1);
    DMT_CPU_GPU SampledSpectrum  operator/(SampledSpectrum const& spec0, SampledSpectrum const& spec1);
    DMT_CPU_GPU SampledSpectrum& operator+=(SampledSpectrum& spec0, SampledSpectrum const& spec1);
    DMT_CPU_GPU SampledSpectrum& operator-=(SampledSpectrum& spec0, SampledSpectrum const& spec1);
    DMT_CPU_GPU SampledSpectrum& operator*=(SampledSpectrum& spec0, SampledSpectrum const& spec1);
    DMT_CPU_GPU SampledSpectrum& operator/=(SampledSpectrum& spec0, SampledSpectrum const& spec1);

    DMT_CPU_GPU bool       hasNaN(SampledSpectrum const& spec);
    DMT_CPU_GPU ValueIndex max(SampledSpectrum const& spec);
    DMT_CPU_GPU ValueIndex min(SampledSpectrum const& spec);
    DMT_CPU_GPU float      average(SampledSpectrum const& spec);

    // Sampled Spectrum and Sampled Wavelengths: Functions ------------------------------------------------------------
    DMT_CPU_GPU SampledWavelengths sampleUniforms(float u, float _lambdaMin = lambdaMin(), float _lambdaMax = lambdaMax());
    DMT_CPU_GPU SampledWavelengths sampleVisible(float u);
    DMT_CPU_GPU SampledSpectrum    PDF(SampledWavelengths& wavelengths);

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
    struct ConstantSpectrum
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
} // namespace dmt
