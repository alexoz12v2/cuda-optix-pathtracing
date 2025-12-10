#ifndef DMT_TRAINING_COOPERATIVE_GROUPS_SAMPLE_RUNNER_H
#define DMT_TRAINING_COOPERATIVE_GROUPS_SAMPLE_RUNNER_H

// cuda stuff
#include <cuda.h>
#include <cuda_runtime.h>

// our stuff
#include "sample-utils.h"

namespace dmt {

    /// an RAII wrapper over a CUcontext on a CUDA device https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#context
    class SampleRunner
    {
    public:
        /// creates a new context for the first eligible CUDA Device and pushes it to that
        /// the indicated cuda device should already be initialized (meaning a primary context
        /// should have been already created for it -> cudaInitDevice, cudaSetDevice or cuDevicePrimaryCtxRetain)
        explicit SampleRunner(int device);
        SampleRunner(SampleRunner const&)                = delete;
        SampleRunner(SampleRunner&&) noexcept            = delete;
        SampleRunner& operator=(SampleRunner const&)     = delete;
        SampleRunner& operator=(SampleRunner&&) noexcept = delete;
        /// pops the context if it's current in the calling owner thread and destroys it
        /// \warning: if any other threads are using this context, they are screwed!
        ~SampleRunner();

        operator bool() const { return m_device >= 0 && m_context; }

        /// \note doesn't check for parameter consistency. It's only responsibility is to parse them
        void saxpySample(CommandData const& params) const;

    private:
        bool ensureCurrentContext() const;

        int       m_device  = -1;
        CUcontext m_context = nullptr;
    };

} // namespace dmt

namespace dmt::saxpy {
    static auto const* const InputDim = L"InputDim";
    static auto const* const Gen      = L"Gen";
    static auto const* const Factor   = L"Factor";
} // namespace dmt::saxpy

#endif // DMT_TRAINING_COOPERATIVE_GROUPS_SAMPLE_RUNNER_H
