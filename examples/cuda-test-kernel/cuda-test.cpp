#include "cuda-test.h"

#include "platform/platform-context.h"
#include <algorithm>

namespace dmt {
    std::vector<char const*> getnvccOpts(bool debug)
    {
        std::vector<char const*> opts{"--gpu-architecture=compute_60", // TODO check compatibility with current context device
                                      "--use_fast_math",
                                      "--relocatable-device-code=true",
                                      "--std=c++17",
                                      "-default-device"};

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

#if 1
        // TODO Remove: Dump compiled file to current working directory such that debugger picks on it
        std::ofstream ptxFile{"saxpy.cu"};
        ptxFile << srcKernel;
        ptxFile.flush();
#endif
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

    // ------------------------------------------------------------
    // Precompute the filter distribution in host memory
    // ------------------------------------------------------------
    PiecewiseConstant2D precalculateMitchellDistrib(filtering::Mitchell const& filter,
                                                    int                        Nx,
                                                    int                        Ny,
                                                    std::pmr::memory_resource* mem)
    {
        Bounds2f domain;
        domain.pMin = Point2f(-filter.radius().x, -filter.radius().y);
        domain.pMax = Point2f(filter.radius().x, filter.radius().y);

        dstd::Array2D<float> values(Nx, Ny);

        Vector2f cellSize((domain.pMax.x - domain.pMin.x) / Nx, (domain.pMax.y - domain.pMin.y) / Ny);

        for (int y = 0; y < Ny; ++y)
        {
            for (int x = 0; x < Nx; ++x)
            {
                // Cell center
                float px = domain.pMin.x + (x + 0.5f) * cellSize.x;
                float py = domain.pMin.y + (y + 0.5f) * cellSize.y;

                float val    = filter.evaluate(Point2f(px, py));
                values(x, y) = std::max(0.0f, val);
            }
        }

        return PiecewiseConstant2D(values, domain, mem);
    }

    GpuSamplerHandle uploadFilterDistrib(CUDADriverLibrary*         cudaApi,
                                         PiecewiseConstant2D const& cpuDistrib,
                                         filtering::Mitchell const& cpuFilter)
    {
        //device pointer
        GpuSamplerHandle handle;

        int Nx = cpuDistrib.resolution().x;
        int Ny = cpuDistrib.resolution().y;

        // Flatten conditional CDF (Nx+1 per row)
        std::vector<float> conditionalFlat;
        conditionalFlat.reserve(Ny * (Nx + 1));
        for (int y = 0; y < Ny; ++y)
        {
            auto const& rowCdf = cpuDistrib.conditionalCdfRow(y);
            conditionalFlat.insert(conditionalFlat.end(), rowCdf.begin(), rowCdf.end());
        }

        // Flatten marginal CDF (Ny+1)
        auto const& marginalCdf = cpuDistrib.marginalCdf();

        // Allocate GPU memory
        cudaApi->cuMemAlloc((CUdeviceptr*) &handle.dConditionalCdf, conditionalFlat.size() * sizeof(float));
        cudaApi->cuMemAlloc((CUdeviceptr*) &handle.dMarginalCdf, marginalCdf.size() * sizeof(float));

        // Copy to GPU
        cudaApi->cuMemcpyHtoD((CUdeviceptr)handle.dConditionalCdf, conditionalFlat.data(), conditionalFlat.size() * sizeof(float));
        cudaApi->cuMemcpyHtoD((CUdeviceptr)handle.dMarginalCdf, marginalCdf.data(), marginalCdf.size() * sizeof(float));

        // Fill GPU sampler struct
        handle.sampler.filter.radiusX = cpuFilter.radius().x;
        handle.sampler.filter.radiusY = cpuFilter.radius().y;
        handle.sampler.filter.B       = cpuFilter.b();
        handle.sampler.filter.C       = cpuFilter.c();

        handle.sampler.distrib.conditionalCdf = handle.dConditionalCdf;
        handle.sampler.distrib.marginalCdf    = handle.dMarginalCdf;
        handle.sampler.distrib.Nx             = Nx;
        handle.sampler.distrib.Ny             = Ny;
        handle.sampler.distrib.pMin           = Point2f(cpuDistrib.domain().pMin.x, cpuDistrib.domain().pMin.y);
        handle.sampler.distrib.pMax           = Point2f(cpuDistrib.domain().pMax.x, cpuDistrib.domain().pMax.y);
        handle.sampler.distrib.integral       = cpuDistrib.integral();

        return handle;
    }


} // namespace dmt
