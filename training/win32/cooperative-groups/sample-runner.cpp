#include "sample-runner.h"

// our stuff
#include "the-macros.h"
#include "parsing-utils.h"
#include "example.h"

// library stuff
#include <iostream>
#include <iomanip>
#include <cassert>
#include <sstream>
#include <chrono>

namespace dmt {
    SampleRunner::SampleRunner(int device)
    {
        int         theDevice = -1;
        cudaError_t err       = ::cudaSuccess;
        CUresult    cuErr     = ::CUDA_SUCCESS;
        CUdevice    cuDevice  = 0;

        err = cudaGetDevice(&theDevice);
        if (!CUDA_SUCC(err))
            return;
        if (theDevice != device)
        {
            std::cerr << ANSI_RED "[SampleRunner Error] The current device " << theDevice
                      << "is not equal to 'device' = " << device << std::endl;
            return;
        }

        cuErr = cuDeviceGet(&cuDevice, device);
        if (!CU_SUCC(cuErr))
            return;
        cuErr = cuCtxCreate(&m_context, 0, cuDevice);
        if (!CU_SUCC(cuErr))
            return;
        // cuCtxCreate includes a cuCtxPushCurrent operation, no need to push context
        m_device = device;
    }

    SampleRunner::~SampleRunner()
    {
        CUcontext theContext = nullptr;
        if (CU_SUCC(cuCtxGetCurrent(&theContext)))
        {
            if (theContext != m_context)
                cuCtxPopCurrent(&m_context);
        }
        CU_SUCC(cuCtxDestroy(m_context));
    }

    bool SampleRunner::ensureCurrentContext() const
    {
        CUcontext theContext = nullptr;
        if (CU_SUCC(cuCtxGetCurrent(&theContext)))
        {
            if (theContext != m_context)
            {
                std::cout << ANSI_YLW "[Sample Runner] Warning: Sample Context not current for device " << m_device
                          << " setting it" ANSI_RST << std::endl;
                return CU_SUCC(cuCtxPushCurrent(m_context));
            }
            return true;
        }
        return false;
    }

    void SampleRunner::saxpySample(CommandData const& params) const
    {
        assert(static_cast<bool>(*this));
        static constexpr size_t maxPrint = 9;
        static constexpr auto   printVec =
            [](std::ostringstream& theString, float const* vec, char const* name, size_t printCount) {
            theString << "\t" << name << " : [ ";
            for (size_t i = 0; i < printCount; ++i)
            {
                theString << vec[i];
                if (i < printCount - 1)
                    theString << ", ";
                else
                    theString << " ... ]\n";
            }
        };
        // parse parameters
        size_t const       inputDim = *parse_number<size_t>(params.KeyedCmds.at(saxpy::InputDim).values[0]);
        std::wstring const gen      = params.KeyedCmds.at(saxpy::Gen).values[0];
        float const        factor   = *parse_number<float>(params.KeyedCmds.at(saxpy::Factor).values[0]);

        // allocate input memory and output
        std::vector<float> first(inputDim, 0.f);
        std::vector<float> second(inputDim, 0.f);
        std::vector<float> result(inputDim, 0.f);
        if (gen == L"OnePlusTwo")
        {
            std::fill_n(first.begin(), inputDim, 1.f);
            std::fill_n(second.begin(), inputDim, 2.f);
        }

        // print hello message and input streams
        size_t const printCount = std::min<size_t>(maxPrint, inputDim);
        std::cout << "Executing 'saxpy' sample: 's = ax + b'";
        {
            size_t const       inputPrintSpace = std::max<size_t>(prettyPrintFloatMaxWidth(first.data(), printCount),
                                                            prettyPrintFloatMaxWidth(second.data(), printCount));
            std::ostringstream theString;
            theString << std::setw(inputPrintSpace) << std::fixed << std::setprecision(3);
            printVec(theString, first.data(), "a", printCount);
            printVec(theString, second.data(), "b", printCount);
            std::cout << '\n' << theString.str() << std::endl;
        }

        // execute sample
        std::chrono::high_resolution_clock::time_point const begin = std::chrono::high_resolution_clock::now();
        _1basics::saxpy(first.data(), second.data(), result.data(), factor, inputDim);
        std::chrono::high_resolution_clock::time_point const end = std::chrono::high_resolution_clock::now();
        size_t const microSeconds = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();

        // print result
        std::cout << "[Sample Runner] saxpy sample finished in " << microSeconds << " us. Result:" << std::endl;
        {
            std::ostringstream theString;
            size_t const       inputPrintSpace = prettyPrintFloatMaxWidth(result.data(), printCount);
            theString << std::setw(inputPrintSpace) << std::fixed << std::setprecision(3);
            printVec(theString, result.data(), "s", printCount);
            std::cout << theString.str() << std::endl;
        }
    }
} // namespace dmt