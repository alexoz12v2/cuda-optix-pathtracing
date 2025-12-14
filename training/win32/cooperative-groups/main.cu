// std stuff
#include <iostream>
#include <vector>
#include <string>
#include <cctype>
#include <algorithm>
#include <iterator>

// Windows Stuff
#include "Windows.h"
#include "ShlObj.h"
#include "Objbase.h"

// cuda stuff
#include <cuda_runtime.h>
#include <cooperative_groups.h>

// our stuff
#include "cudautils/cudautils-camera.cuh"
#include "cudautils/cudautils-vecmath.cuh"
#include "the-macros.h"
#include "example.h"
#include "sample-runner.h"
#include "sample-utils.h"

using namespace dmt;

static auto numberValidator(std::wstring_view param, bool floatingAccepted) -> decltype(auto)
{
    return [param, floatingAccepted](std::wstring_view str) -> std::wstring {
        if (str.empty())
            return std::wstring(param) + L" Empty";

        bool seenDot = false;

        for (wchar_t ch : str)
        {
            if (ch >= L'0' && ch <= L'9')
                continue;

            if (floatingAccepted && ch == L'.')
            {
                if (seenDot) // second dot → invalid
                    return std::wstring(param) + L" Not a Number";

                seenDot = true;
                continue;
            }

            // Anything else is invalid
            return std::wstring(param) + L" Not a Number";
        }

        // If str == "." → invalid as a number
        if (floatingAccepted && (str.size() == 1 && str[0] == L'.'))
            return std::wstring(param) + L" Not a Number";

        return L"";
    };
}

/// ## About CUDA Runtime and Context Management
/// A [Context](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#context)
/// is the CUDA Equivalent of a process. All resources and modules allocated for the process are associated to a
/// context.
/// An Implicit Context will be created for each device called Primary Context whenever you call `cudaInitDevice`

struct CameraSample
{
    Point2f pointFilm{0.f, 0.f};
    Point2f pointLens{0.f, 0.f};
    float   time         = 0.f;
    float   filterWeight = 0.f;
};

struct RaysPool
{
    Vector3f* pDirVec = nullptr;
    Point3f*  pOrgVec = nullptr;
    int       dim     = 0;
};

__device__ void generate_camera_ray_mega(cooperative_groups::thread_group g)
{
    int lane = g.thread_rank();
    if (lane == 0)
    {
        //gen ray
    }
    g.sync();
}

//__constant__ int d_lookup;

__global__ void megakernel(DeviceCamera* dp_dc)
{
    int                                       nPixel = dp_dc->width * dp_dc->height;
    cooperative_groups::thread_block          block  = cooperative_groups::this_thread_block();
    cooperative_groups::thread_block_tile<32> warp   = cooperative_groups::tiled_partition<32>(block);
    for (int i = threadIdx.x; i < nPixel; i += gridDim.x * blockDim.x)
    {
        //sampling
        CameraSample cs;
        generate_camera_ray_mega(warp);
    }
}

int wmain()
{
    // - Setup console properly such that ANSI escape codes work
    for (HANDLE out : {GetStdHandle(STD_OUTPUT_HANDLE), GetStdHandle(STD_ERROR_HANDLE)})
    {
        DWORD mode = 0;
        GetConsoleMode(out, &mode);
        mode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
        mode |= DISABLE_NEWLINE_AUTO_RETURN;
        SetConsoleMode(out, mode);
    }
    std::ios::sync_with_stdio();

    // - Print some colored stuff
    std::cout << ANSI_RED "Hello Beautiful World" ANSI_RST << std::endl;

    std::vector<std::wstring> args;
    {
        std::wstring cmd  = GetCommandLineW();
        size_t       prev = 0;
        for (size_t pos = 0; (pos = cmd.find_first_of(L" \t", prev)) != std::wstring::npos; /**/)
        {

             if ((prev - pos+ 1 > 2) && cmd[pos - 1] == L'"' && cmd[prev] == L'"')
                args.emplace_back(cmd.substr(prev + 1, pos - prev - 2));
            else args.emplace_back(cmd.substr(prev, pos - prev));
            prev = pos + 1;
        }
        if (prev < cmd.size())
            args.emplace_back(cmd.substr(prev));
    }

    std::cout << "[DEBUG] Parsed Command Line:\n\t";
    for (std::wstring const& arg : args)
        std::wcout << arg << L' ';
    std::cout << std::endl;

    size_t firstArg = 1;
    {
        std::wstring moduleBuffer;
        moduleBuffer.resize(512);
        if (DWORD const count = GetModuleFileNameW(nullptr, moduleBuffer.data(), moduleBuffer.size()); count > 0)
        {
            moduleBuffer.resize(count);
            if (moduleBuffer != args[0])
                firstArg -= 1;
        }
    }

    if (args.size() < firstArg + 1)
    {
        std::cerr << ANSI_RED "Insufficient Arguments. At least a command must be specified" ANSI_RST << std::endl;
        return 1;
    }
    std::vector<std::wstring> const mainCommand{args[firstArg], args[firstArg + 1]};

    // initialize COM Apartment for this process
    if (HRESULT const res = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED); !SUCCEEDED(res))
        win32::printResultAndExitProcess(res);

    _1basics::printCudaCapableDevices();

    // build command line parser to parametrize training
    TrainingSampleParser mainParser;
    mainParser.AddCommand(L"Run")
        .AddAlias(L"Run")
        .SetRequired()
        .WithValue()
        .WithValidator([](std::wstring_view val) {
        static std::vector<std::wstring> const Samples{L"kernel", L"sample"};

        if (std::ranges::find(std::as_const(Samples), val) == Samples.cend())
            return L"Unknown run command";
        return L"";
    }).Build();

    CommandData data;
    if (std::vector<std::wstring> const errors = mainParser.Parse(mainCommand, data); !errors.empty())
    {
        for (auto error : errors)
            std::wcerr << error << std::endl;
        return 1;
    }

    KeyedCommandInstance const& runCommand = data.KeyedCmds.at(L"Run");
    std::vector<std::wstring>   subCommand;
    std::wstring                subCommandName;
    if (args.size() > firstArg + 2)
    {
        subCommandName = args[firstArg + 2];
        if (args.size() > firstArg + 3)
            std::copy(args.begin() + firstArg + 3, args.end(), std::back_inserter(subCommand));
    }
    if (runCommand.values[0] == L"sample")
    {
        if (subCommandName == L"saxpy")
        {
            if (subCommand.empty())
            {
                std::cerr << "Syntax: saxpy -InputDim 1024 -Gen <Gen> -Factor <num>" << std::endl;
                return 1;
            }
            TrainingSampleParser  saxpyParser;
            [[maybe_unused]] bool commandInserted = saxpyParser.AddCommand(L"InputDim")
                                                        .AddAlias(L"InputDim")
                                                        .SetRequired()
                                                        .WithValue()
                                                        .WithValidator(numberValidator(L"InputDim", false))
                                                        .Build();
            assert(commandInserted);
            commandInserted = saxpyParser.AddCommand(L"Gen")
                                  .AddAlias(L"Gen")
                                  .SetRequired()
                                  .WithValue()
                                  .WithValidator([](std::wstring_view str) {
                if (str != L"OnePlusTwo")
                    return L"Unsupported value Generator";
                return L"";
            }).Build();
            assert(commandInserted);
            commandInserted = saxpyParser.AddCommand(L"Factor")
                                  .AddAlias(L"Factor")
                                  .SetRequired()
                                  .WithValue()
                                  .WithValidator(numberValidator(L"Factor", true))
                                  .Build();
            assert(commandInserted);

            CommandData saxpyData;
            if (auto const errs = saxpyParser.Parse(subCommand, saxpyData); !errs.empty())
            {
                std::wcerr << WANSI_RED << errs[0] << WANSI_RST << std::endl;
                return 1;
            }

            CUDA_SUCC(cudaInitDevice(0, 0, 0));
            SampleRunner runner(0);
            if (runner)
                runner.saxpySample(saxpyData);
            else
            {
                std::cerr << ANSI_RED "Falied to initialize cuda context, exiting" ANSI_RST << std::endl;
                return 1;
            }
        }
        else
        {
            std::wcerr << WANSI_RED L"Unrecognized sample '" << subCommandName << L'\'' << std::endl;
            return 1;
        }
    }
    else if (runCommand.values[0] == L"kernel")
    {
        cudaInitDevice(0, 0, 0);
        //get device property
        int            device = 0;
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);
        int nSM             = prop.multiProcessorCount;
        int nThreadPerBlock = prop.maxThreadsPerBlock;

        DeviceCamera  h_dc;
        DeviceCamera* dp_dc;
        cudaMallocHost(&dp_dc, sizeof(DeviceCamera));
        cudaMemcpy(dp_dc, &h_dc, sizeof(DeviceCamera), cudaMemcpyHostToDevice);
        //cudaMemcoyToSymnbol
        //Inital assumption: number of samples cannot exceed the number of threads per block
        //cycling for obtain more samples???
        megakernel<<<nSM, nThreadPerBlock>>>(dp_dc);
    }
    return 0;
}
