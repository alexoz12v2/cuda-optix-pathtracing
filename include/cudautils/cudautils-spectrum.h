#pragma once

#include "cudautils/cudautils-macro.h"
#include "cudautils/cudautils-vecmath.h"

#if !defined(__CUDA_ARCH__)
    #include <platform/platform-utils.h>

    #include <concepts>
#endif

namespace dmt {
    /** https://www.desmos.com/calculator/e0arbln6ip
     * @note This has a very low accuracy and I don't know why.
     */
    DMT_CORE_API DMT_CPU_GPU float blackbody(float lambdaNanometers, float TKelvin);

    DMT_CPU_GPU inline constexpr float    lambdaMin() { return 360.f; }
    DMT_CPU_GPU inline constexpr float    lambdaMax() { return 830.f; }
    DMT_CPU_GPU inline constexpr uint32_t numSpectrumSamples() { return 4u; }
#if !defined(__CUDA_ARCH__)
    static_assert(numSpectrumSamples() > 0 && nextPOT(numSpectrumSamples()) == numSpectrumSamples());
#endif

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
        SampledSpectrum() = default;
        DMT_CPU_GPU SampledSpectrum(ArrayView<float> values);
        DMT_CPU_GPU explicit SampledSpectrum(float c)
        {
            for (int i = 0; i < numSpectrumSamples(); i++)
                values[i] = c;
        };
        float values[numSpectrumSamples()]{};
    };

    struct DMT_CORE_API SampledWavelengths
    {
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

#if __cplusplus >= 202002L

    // clang-format off
    template <typename T>
    concept SpectrumType = requires (T t) {
        { t(3.f) } -> std::same_as<float>;
        { t.maxValue() } -> std::same_as<float>;
        { t.sample(std::declval<SampledWavelengths>()) } -> std::same_as<SampledSpectrum>;
    };
    // clang-format on

#endif

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
#if __cplusplus >= 202002L
    static_assert(SpectrumType<ConstantSpectrum>);
#endif

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

#if defined(DMT_CUDAUTILS_IMPLEMENTATION)

    #include "cudautils/cudautils-numbers.h"

namespace dmt {
    // Constants used for sampled Spectrum and Wavelengths vectorization
    static constexpr uint32_t residual = numSpectrumSamples() % 4;
    static constexpr uint32_t numVect  = numSpectrumSamples() / 4;

    #if defined(__CUDA_ARCH__)
    inline __device__ bool float_equal_strict(float a, float b)
    {
        // exact bitwise comparison for float
        return a == b;
    }

    inline __device__ float expm1f_kahan(float x)
    {
        float u = expf(x);

        if (float_equal_strict(u, 1.0f))
        {
            return x; // small x, exp(x)-1 ≈ x
        }

        float um1 = u - 1.0f;
        if (float_equal_strict(um1, -1.0f))
        {
            return -1.0f; // x very negative, exp(x)-1 ≈ -1
        }

        float logu = logf(u);
        return float_equal_strict(u, logu) ? u : (u - 1.0f) * x / logu;
    }
    #endif

    __host__ __device__ float blackbody(float lambdaNanometers, float TKelvin)
    {
        float                  Le          = 0.f;
        static constexpr float _2hc        = 1.1910428681e-16f; // 2.f * fl::c() * fl::c() * fl::planck();
        static constexpr float _hc_over_kb = 0.0143877695998f;  // fl::planck() * fl::c() / fl::kBoltz();
        float const            tmp         = lambdaNanometers * 1e-9f;
        assert(glm::epsilonNotEqual(tmp, 0.f, std::numeric_limits<float>::epsilon()));
        float lambdaMeters5 = lambdaNanometers * 1e-9f;
        lambdaMeters5 *= lambdaMeters5;
        lambdaMeters5 *= lambdaMeters5;
        lambdaMeters5 *= tmp;
        float const lambdaT = tmp * TKelvin;
        float const expArg  = _hc_over_kb * fl::rcp(lambdaT);
        assert(expArg <= 700.f); // exp(700) is near float limit
        Le = _2hc / (lambdaMeters5 *
    #if defined(__CUDA_ARCH__)
                     expm1f_kahan(expArg)
    #else
                     Eigen::numext::expm1(expArg)
    #endif
                    );
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
                bool isNotZero = reinterpret_cast<glm::vec4 const*>(&lambda[0])[i] != zero;
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
            reinterpret_cast<glm::vec4*>(&spec0.values[0])[i] += reinterpret_cast<glm::vec4 const*>(&spec1.values[0])[i];
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
            reinterpret_cast<glm::vec4*>(&spec0.values[0])[i] -= reinterpret_cast<glm::vec4 const*>(&spec1.values[0])[i];
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
            reinterpret_cast<glm::vec4*>(&spec0.values[0])[i] *= reinterpret_cast<glm::vec4 const*>(&spec1.values[0])[0];
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
            reinterpret_cast<glm::vec4*>(&spec0.values[0])[i] /= reinterpret_cast<glm::vec4 const*>(&spec1.values[0])[i];
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
        ValueIndex ret{-std::numeric_limits<float>::infinity(), -1};
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
        ValueIndex ret{std::numeric_limits<float>::infinity(), -1};
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
        for (uint32_t i = 0; i < numSpectrumSamples(); ++i)
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
            reinterpret_cast<Vector4f*>(&ret.lambda[0])[i] = sampling::sampleVisibleWavelengths(*fromGLM(&ups));
            reinterpret_cast<Vector4f*>(&ret.pdf[0])[i]    = sampling::visibleWavelengthsPDF(
                reinterpret_cast<Vector4f*>(&ret.lambda[0])[i]);
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

    // Densely Sampled Spectrum ---------------------------------------------------------------------------------------

    /* TODO remake
    __host__ __device__ DenselySampledSpectrum::DenselySampledSpectrum(
        FloatFunction2      func,
        BaseMemoryResource* alloc,
        CudaStreamHandle    stream,
        int32_t             _lambdaMin,
        int32_t             _lambdaMax) :
    m_values(sizeof(float), alloc, stream),
    m_lambdaMin(_lambdaMin),
    m_lambdaMax(_lambdaMax)
    {
        m_values.lockForWrite();
        // TODO Device path should use warp level explicit stuff?
        assert(m_lambdaMax > m_lambdaMin + 3); // at least 4 values
        int32_t const numel = m_lambdaMax - m_lambdaMin + 1;
        m_values.resize(numel, false);

        float* buffer = nullptr;
#if !defined(__CUDA_ARCH__)
        dmt::AppContextJanitor j;
        if (!m_values.resource()->hostHasAccess())
        {
            auto* ptr = std::bit_cast<float*>(
                j.actx.stackAllocate(numel * sizeof(float), alignof(float), EMemoryTag::eBuffer, 0));
            m_values.copyToHostSync(ptr, false);
            buffer = ptr;
        }
        else
            buffer = std::bit_cast<float*>(m_values.data());
#else
        buffer = std::bit_cast<float*>(m_values.data());
#endif

        for (int32_t lambda = m_lambdaMin; lambda <= m_lambdaMax; ++lambda)
        {
            buffer[lambda - m_lambdaMin] = func(lambda);
        }
#if !defined(__CUDA_ARCH__)
        if (!m_values.resource()->hostHasAccess())
        {
            if (m_values.copyFromHostAsync(buffer, numel * sizeof(float), false))
                m_values.syncWithStream(false);
        }
#endif

        m_values.unlockForWrite();
    }

    __host__ __device__ float DenselySampledSpectrum::operator()(float lambdanm) const
    {
        int32_t index = static_cast<int32_t>(glm::round(lambdanm)) - m_lambdaMin;
        if (index < 0 || index > (m_lambdaMax - m_lambdaMin))
            return 0.f;
        else
        {
#if defined(__CUDA_ARCH__)
            return m_values.operator[]<float>(index);
#else
            if (m_values.resource()->hostHasAccess())
                return m_values.operator[]<float>(index);
            else
            {
                // TODO decomment
                //AppContextJanitor j;
                //j.actx.warn("You are manually copyting from device to host the requested element");
                m_values.lockForRead();
                cudaStream_t stream_ = m_values.stream == noStream ? 0 : streamRefFromHandle(m_values.stream).get();

                // cudaMemcpy works for multiple of 8 Bytes it seems
                float       res{0.f};
                cudaError_t err = cudaMemcpyAsync(&res,
                                                  std::bit_cast<float*>(m_values.data()) + index,
                                                  sizeof(float),
                                                  ::cudaMemcpyDeviceToHost,
                                                  stream_);
                assert(err == ::cudaSuccess);
                m_values.unlockForRead();
                return res;
            }
#endif
        }
    }

    // note: we don't know how many active threads enter in device mode, so the only way is to have them
    // all compute the max
    __host__ __device__ float DenselySampledSpectrum::maxValue() const
    {
        float         max  = -std::numeric_limits<float>::infinity();
        int32_t const size = m_lambdaMax - m_lambdaMin + 1;

        m_values.lockForRead();

        float const* buffer = getBuffer();

#if !defined(__CUDA_ARCH__)
        // Host path: Vectorize using glm and handle residual
        constexpr int vecSize  = 4; // Assuming glm::vec4
        int32_t       vecCount = size / vecSize;
        int32_t       residual = size % vecSize;

        for (int32_t i = 0; i < vecCount; ++i)
        {
            glm::vec4 v = std::bit_cast<glm::vec4 const*>(&buffer[0])[i];
            max         = glm::max(max, glm::compMax(v));
        }
        for (int32_t i = size - residual; i < size; ++i)
        {
            if (float num = buffer[i]; max < num)
                max = num;
        }

#else
        // TODO This doesn't work, doesn't account for non active threads inside each warp of the block use warp explicit reduction, basically:
        // compute capability upper limit : cc 6.1
        // ```
        // unsigned __ballot_sync(unsigned mask, int predicate);
        // unsigned __activemask();
        // ```
        // since it might be that warps of the block don't execute this function, the reduction should
        // be performed inside each warp, accounting for awke threads inside the warp using Warp Vote Functions
        // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-vote-functions
        // Device path: Block-local reduction
        // Dynamically allocated shared memory (assumes there's at least lest `blockThreadCount() * sizeof(float)` Gytes)
        // Device path: Warp-local reduction
        // Dynamically allocated shared memory (at least `blockThreadCount() * sizeof(float)` bytes required)
        extern __shared__ float sharedMax[];

        int32_t threadId        = blockThreadIndex();    // Thread ID within the block
        int32_t threadsPerBlock = blockThreadCount();    // Total threads in the block
        int32_t warpId          = threadId / warpSize;   // Warp ID within the block
        int32_t laneId          = warpWideThreadIndex(); // Thread ID within the warp

        float localMax = -std::numeric_limits<float>::infinity();

        // Each thread processes part of the array
        for (int32_t i = threadId; i < size; i += threadsPerBlock)
        {
            float num = buffer[i];
            localMax  = fmaxf(localMax, num);
        }

        // Perform warp-local reduction
        unsigned int activeMask = __activemask(); // Mask of active threads in the current warp
        for (int offset = 16; offset > 0; offset /= 2)
        {
            float otherMax = __shfl_down_sync(activeMask, localMax, offset);
            if (laneId + offset < 32) // Ensure we only reduce with valid threads in the warp
                localMax = fmaxf(localMax, otherMax);
        }

        // Write the warp-local maximum to shared memory
        if (laneId == 0) // The first thread in each warp writes to shared memory
            sharedMax[warpId] = localMax;

        __syncthreads(); // Synchronize across the block

        // Reduce across warps within the block
        if (warpId == 0)
        {
            localMax = (laneId < (threadsPerBlock + 31) / 32) ? sharedMax[laneId] : -std::numeric_limits<float>::infinity();
            for (int offset = 16; offset > 0; offset /= 2)
            {
                float otherMax = __shfl_down_sync(__activemask(), localMax, offset);
                if (laneId + offset < 32)
                    localMax = fmaxf(localMax, otherMax);
            }

            // Thread 0 in warp 0 writes the block-local maximum
            if (laneId == 0)
                max = localMax;
        }
#endif

        m_values.unlockForRead();
        assert(max >= 0.f);
        return max;
    }

    // `SampledSpectrum` is supposed to be small, frmo 4 to 16, so anything complicated is not worth it
    __host__ __device__ SampledSpectrum DenselySampledSpectrum::sample(SampledWavelengths const& wavelengths) const
    {
        SampledSpectrum ret;
        m_values.lockForRead();

        float const* buffer = getBuffer();

        for (uint32_t i = 0; i < numSpectrumSamples(); ++i)
        {
            int32_t index = static_cast<int32_t>(glm::round(wavelengths.lambda[i])) - m_lambdaMin;
            if (index < 0 || index > (m_lambdaMax - m_lambdaMin))
                ret.values[i] = 0.f;
            else
                ret.values[i] = buffer[index];
        }

        m_values.unlockForRead();
        return ret;
    }

    // TODO check if cuda capable, and if yes, invoke a kernel from host
    __host__ __device__ void DenselySampledSpectrum::scale(float factor)
    {
        m_values.lockForWrite();
        float* buffer = getBuffer();

#if defined(__CUDA_ARCH__)
        // TODO better
        for (int32_t i = 0; i < (m_lambdaMax - m_lambdaMin); ++i)
            buffer[i] *= factor;
#else
        size_t size = (m_lambdaMax - m_lambdaMin) * sizeof(float);
        for (int32_t i = 0; i <= (m_lambdaMax - m_lambdaMin); ++i)
            buffer[i] *= factor;

        if (!m_values.resource()->hostHasAccess())
        {
            if (m_values.copyFromHostAsync(buffer, size, false))
                m_values.syncWithStream(false);
        }
#endif

        m_values.unlockForWrite();
    }

    __host__ __device__ float* DenselySampledSpectrum::getBuffer()
    {
        float* buffer = nullptr;
#if !defined(__CUDA_ARCH__)
        size_t const           size = (m_lambdaMax - m_lambdaMin + 1) * sizeof(float);
        dmt::AppContextJanitor j;
        if (!m_values.resource()->hostHasAccess())
        {
            auto* ptr = std::bit_cast<float*>(j.actx.stackAllocate(size, alignof(float), EMemoryTag::eBuffer, 0));
            m_values.copyToHostSync(ptr, false);
            buffer = ptr;
        }
        else
            buffer = std::bit_cast<float*>(m_values.data());
#else
        buffer = std::bit_cast<float*>(m_values.data());
#endif
        return buffer;
    }

    __host__ __device__ float const* DenselySampledSpectrum::getBuffer() const
    {
        float const* buffer = nullptr;
#if !defined(__CUDA_ARCH__)
        size_t const           size = (m_lambdaMax - m_lambdaMin + 1) * sizeof(float);
        dmt::AppContextJanitor j;
        if (!m_values.resource()->hostHasAccess())
        {
            auto* ptr = std::bit_cast<float*>(j.actx.stackAllocate(size, alignof(float), EMemoryTag::eBuffer, 0));
            m_values.copyToHostSync(ptr, false);
            buffer = ptr;
        }
        else
            buffer = std::bit_cast<float const*>(m_values.data());
#else
        buffer = std::bit_cast<float const*>(m_values.data());
#endif
        return buffer;
    }
    */

} // namespace dmt

#endif