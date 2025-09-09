#include "cuda-test.h"

#include "platform/platform-context.h"

namespace dmt {
    std::vector<char const*> getnvccOpts(bool debug)
    {
        std::vector<char const*> opts{
            "--gpu-architecture=compute_75",
            "--use_fast_math",
            "--relocatable-device-code=true",
            "--std=c++20",
        };

        if (debug)
        {
            opts.push_back("-lineinfo");
            opts.push_back("-G");
        }

        return opts;
    }

    std::unique_ptr<char[]> compilePTX(dmt::os::Path const&            path,
                                       NVRTCLibrary*                   nvrtcApi,
                                       std::string_view                kernelFileName,
                                       std::vector<char const*> const& nvccOpts)
    {
        Context ctx;

        std::ifstream file{path.toUnderlying().c_str()};

        if (!file)
        {
            ctx.log("File not found", {});
            return nullptr;
        }
        std::string srcKernel{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
        return compilePTX(srcKernel, nvrtcApi, kernelFileName, nvccOpts);
    }

    std::unique_ptr<char[]> compilePTX(std::string_view                srcKernel,
                                       NVRTCLibrary*                   nvrtcApi,
                                       std::string_view                kernelFileName,
                                       std::vector<char const*> const& nvccOpts)
    {
        Context ctx;

        nvrtcProgram prog = nullptr;
        if (auto res = nvrtcApi->nvrtcCreateProgram(&prog, srcKernel.data(), kernelFileName.data(), 0, nullptr, nullptr);
            res != ::NVRTC_SUCCESS)
        {
            ctx.error("({}) nvrtcCreateProgram Error: {}",
                      std::make_tuple(kernelFileName, nvrtcApi->nvrtcGetErrorString(res)));
            return nullptr;
        }

        std::ranges::for_each(nvccOpts, [&ctx](char const* str) {
            ctx.log("  opts: {}", std::make_tuple(std::string_view(str)));
        });

        if (nvrtcApi->nvrtcCompileProgram(prog, static_cast<int32_t>(nvccOpts.size()), nvccOpts.data()) != ::NVRTC_SUCCESS)
        {
            size_t logSize = 0;
            nvrtcApi->nvrtcGetProgramLogSize(prog, &logSize);
            auto logBuf = std::make_unique<char[]>(logSize + 1);
            if (!logBuf)
                return nullptr;
            logBuf[logSize] = '\0';
            nvrtcApi->nvrtcGetProgramLog(prog, logBuf.get());

            std::string logStr(logBuf.get());

            // Break the log into chunks of 256 characters
            size_t const chunkSize = 256;
            for (size_t i = 0; i < logStr.size(); i += chunkSize)
            {
                std::string_view chunk(logStr.data() + i, std::min(chunkSize, logStr.size() - i));
                ctx.error("({}) nvrtcCompileProgram Failed:\n{}", std::make_tuple(kernelFileName, chunk));
            }

            return nullptr;
        }

        size_t ptxSize = 0;

        if (auto res = nvrtcApi->nvrtcGetPTXSize(prog, &ptxSize); res != ::NVRTC_SUCCESS)
        {
            ctx.log("{}", std::make_tuple(nvrtcApi->nvrtcGetErrorString(res)));
            return nullptr;
        }

        std::unique_ptr<char[]> ptxBuffer = std::make_unique<char[]>(ptxSize);
        nvrtcApi->nvrtcGetPTX(prog, ptxBuffer.get());
        nvrtcApi->nvrtcDestroyProgram(&prog);

        return ptxBuffer;
    }
} // namespace dmt
