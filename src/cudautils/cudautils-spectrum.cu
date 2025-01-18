#include "cudautils-spectrum.h"

#include <cudautils-float.h>
#include "cudautils-vecconv.cuh"
#include <cudautils-numbers.h>

namespace dmt {
    // Constants used for sampled Spectrum and Wavelengths vectorization
    static constexpr uint32_t residual = numSpectrumSamples() % 4;
    static constexpr uint32_t numVect  = numSpectrumSamples() / 4;

    __host__ __device__ float blackbody(float lambdaNanometers, float TKelvin)
    {
        float                  Le          = 0.f;
        static constexpr float _2hc        = 2.f * fl::c() * fl::c() * fl::planck();
        static constexpr float _hc_over_kb = fl::planck() * fl::c() / fl::kBoltz();
        float const            tmp         = lambdaNanometers * 1e-9f;
        assert(glm::epsilonNotEqual(tmp, 0.f, std::numeric_limits<float>::epsilon()));
        float lambdaMeters5 = lambdaNanometers * 1e-9f;
        lambdaMeters5 *= lambdaMeters5;
        lambdaMeters5 *= lambdaMeters5;
        lambdaMeters5 *= tmp;
        float const lambdaT = tmp * TKelvin;
        float const expArg  = _hc_over_kb * fl::rcp(lambdaT);
        assert(expArg <= 700.f); // exp(700) is near float limit
        Le = _2hc / (lambdaMeters5 * Eigen::numext::expm1(expArg));
        return Le;
    }

    // Sampled Spectrum and Sampled Wavelengths: Class Definitions ----------------------------------------------------

    __host__ __device__ SampledSpectrum::SampledSpectrum(ArrayView<float> _values)
    {
        static constexpr size_t numBytes = numSpectrumSamples() * sizeof(float);
        assert(_values.length == numSpectrumSamples());
        ::memcpy(values, _values.data, numBytes);
    }

    __host__ __device__ void SampledWavelengths::terminateSecondary()
    {
        static constexpr size_t numBytes = (numSpectrumSamples() - 1) * sizeof(float);
        if (secondaryTerminated())
            return;
        ::memset(&pdf[1], 0, numBytes);
        pdf[0] /= numSpectrumSamples();
    }

    __host__ __device__ bool SampledWavelengths::secondaryTerminated() const
    {
        glm::vec4 const zero{0.f};
        for (uint32_t i = 1; i < 4; ++i)
        {
            if (pdf[i] != 0)
                return false;
        }
        if constexpr (numVect > 1)
        {
            for (uint32_t i = 1; i < numVect; ++i)
            {
                bool isNotZero = std::bit_cast<glm::vec4 const*>(&lambda[0])[i] != zero;
                if (isNotZero)
                    return false;
            }
            if constexpr (residual != 0)
            {
                for (uint32_t i = numVect << 2; i < numSpectrumSamples(); ++i)
                {
                    if (pdf[i] != 0)
                        return false;
                }
            }
        }
        return true;
    }

    // Sampled Spectrum and Sampled Wavelengths: Boilerplate ----------------------------------------------------------
    __host__ __device__ SampledSpectrum operator+(SampledSpectrum const& spec0, SampledSpectrum const& spec1)
    {
        SampledSpectrum ret = spec0;
        operator+=(ret, spec1);
        return ret;
    }
    __host__ __device__ SampledSpectrum operator-(SampledSpectrum const& spec0, SampledSpectrum const& spec1)
    {
        SampledSpectrum ret = spec0;
        operator-=(ret, spec1);
        return ret;
    }
    __host__ __device__ SampledSpectrum operator*(SampledSpectrum const& spec0, SampledSpectrum const& spec1)
    {
        SampledSpectrum ret = spec0;
        operator*=(ret, spec1);
        return ret;
    }
    __host__ __device__ SampledSpectrum operator/(SampledSpectrum const& spec0, SampledSpectrum const& spec1)
    {
        SampledSpectrum ret = spec0;
        operator/=(ret, spec1);
        return ret;
    }
    __host__ __device__ SampledSpectrum& operator+=(SampledSpectrum& spec0, SampledSpectrum const& spec1)
    {
        for (uint32_t i = 0; i < numVect; ++i)
            std::bit_cast<glm::vec4*>(&spec0.values[0])[i] += std::bit_cast<glm::vec4 const*>(&spec1.values[0])[i];
        if constexpr (residual != 0)
        {
            for (uint32_t i = numVect << 2; i < numSpectrumSamples(); ++i)
                spec0.values[i] += spec1.values[i];
        }
        return spec0;
    }
    __host__ __device__ SampledSpectrum& operator-=(SampledSpectrum& spec0, SampledSpectrum const& spec1)
    {
        for (uint32_t i = 0; i < numVect; ++i)
            std::bit_cast<glm::vec4*>(&spec0.values[0])[i] -= std::bit_cast<glm::vec4 const*>(&spec1.values[0])[i];
        if constexpr (residual != 0)
        {
            for (uint32_t i = numVect << 2; i < numSpectrumSamples(); ++i)
                spec0.values[i] -= spec1.values[i];
        }
        return spec0;
    }
    __host__ __device__ SampledSpectrum& operator*=(SampledSpectrum& spec0, SampledSpectrum const& spec1)
    {
        for (uint32_t i = 0; i < numVect; ++i)
            std::bit_cast<glm::vec4*>(&spec0.values[0])[i] *= std::bit_cast<glm::vec4 const*>(&spec1.values[0])[0];
        if constexpr (residual != 0)
        {
            for (uint32_t i = numVect << 2; i < numSpectrumSamples(); ++i)
                spec0.values[i] *= spec1.values[i];
        }
        return spec0;
    }
    __host__ __device__ SampledSpectrum& operator/=(SampledSpectrum& spec0, SampledSpectrum const& spec1)
    {
        for (uint32_t i = 0; i < numVect; ++i)
            std::bit_cast<glm::vec4*>(&spec0.values[0])[i] /= std::bit_cast<glm::vec4 const*>(&spec1.values[0])[i];
        if constexpr (residual != 0)
        {
            for (uint32_t i = numVect << 2; i < numSpectrumSamples(); ++i)
                spec0.values[i] /= spec1.values[i];
        }
        return spec0;
    }

    __host__ __device__ bool hasNaN(SampledSpectrum const& spec)
    {
        for (uint32_t i = 0; i < numSpectrumSamples(); ++i)
        {
            if (fl::isNaN(spec.values[i]))
                return true;
        }
        return false;
    }
    __host__ __device__ ValueIndex max(SampledSpectrum const& spec)
    {
        ValueIndex ret{.value = -std::numeric_limits<float>::infinity(), .index = -1};
        for (uint32_t i = 0; i < numSpectrumSamples(); ++i)
        {
            if (ret.value < spec.values[i])
            {
                ret.value = spec.values[i];
                ret.index = i;
            }
        }
        return ret;
    }
    __host__ __device__ ValueIndex min(SampledSpectrum const& spec)
    {
        ValueIndex ret{.value = std::numeric_limits<float>::infinity(), .index = -1};
        for (uint32_t i = 0; i < numSpectrumSamples(); ++i)
        {
            if (ret.value > spec.values[i])
            {
                ret.value = spec.values[i];
                ret.index = i;
            }
        }
        return ret;
    }
    __host__ __device__ float average(SampledSpectrum const& spec)
    {
        float sum = spec.values[0];
        for (uint32_t i = 1; i < numSpectrumSamples(); ++i)
            sum += spec.values[i];
        sum /= numSpectrumSamples();
        return sum;
    }

    // Sampled Spectrum and Sampled Wavelengths: Functions ------------------------------------------------------------
    __host__ __device__ SampledWavelengths sampleUniforms(float u, float _lambdaMin, float _lambdaMax)
    {
        SampledWavelengths ret;
        ret.lambda[0]        = glm::lerp(_lambdaMax, _lambdaMin, u);
        float const delta    = (_lambdaMax - _lambdaMin) / numSpectrumSamples();
        float const pdfValue = fl::rcp(_lambdaMax - _lambdaMin);
        for (uint32_t i = 1; i < numSpectrumSamples(); ++i)
        {
            ret.lambda[i] = ret.lambda[i - 1] + delta;
            if (ret.lambda[i] > _lambdaMax) // wrap around
                ret.lambda[i] = _lambdaMin + (ret.lambda[i] - _lambdaMax);
        }
        for (uint32_t i = 1; i < numSpectrumSamples(); ++i)
            ret.pdf[i] = pdfValue;

        return ret;
    }
    __host__ __device__ SampledWavelengths sampleVisible(float u)
    {
        SampledWavelengths ret;
        glm::vec4 ups{u, u + 1.f / numSpectrumSamples(), u + 2.f / numSpectrumSamples(), u + 3.f / numSpectrumSamples()};
        glm::vec4 const _one = glmOne();
        for (uint32_t i = 0; i < numVect; ++i)
        {
            ups -= static_cast<glm::vec4>(glm::greaterThan(ups, _one)) * _one;
            std::bit_cast<Vector4f*>(&ret.lambda[0])[i] = sampling::sampleVisibleWavelengths(fromGLM(ups));
            std::bit_cast<Vector4f*>(&ret.pdf[0])[i]    = sampling::visibleWavelengthsPDF(
                std::bit_cast<Vector4f*>(&ret.lambda[0])[i]);
        }
        if constexpr (residual != 0)
        {
            for (uint32_t i = numVect << 2; i < numSpectrumSamples(); ++i)
            {
                float up = u + static_cast<float>(i) / numSpectrumSamples();
                if (up > 1.f)
                    up -= 1.f;

                ret.lambda[i] = sampling::sampleVisibleWavelengths(up);
                ret.pdf[i]    = sampling::visibleWavelengthsPDF(ret.lambda[i]);
            }
        }
        return ret;
    }
    __host__ __device__ SampledSpectrum PDF(SampledWavelengths& wavelengths)
    {
        static constexpr size_t numBytes = numSpectrumSamples() * sizeof(float);
        SampledSpectrum         ret;
        ::memcpy(ret.values, wavelengths.pdf, numBytes);
        return ret;
    }


    // Constant Spectrum ----------------------------------------------------------------------------------------------

    __host__ __device__ ConstantSpectrum::ConstantSpectrum(float value) : m_value(value) {}

    __host__ __device__ float ConstantSpectrum::operator()(float lambdanm) const { return m_value; }

    __host__ __device__ float ConstantSpectrum::maxValue() const { return m_value; }

} // namespace dmt
