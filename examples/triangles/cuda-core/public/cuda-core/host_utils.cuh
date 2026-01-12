#ifndef DMT_CUDA_CORE_HOST_UTILS_CUH
#define DMT_CUDA_CORE_HOST_UTILS_CUH

#include "cuda-core/common_math.cuh"
#include "cuda-core/bsdf.cuh"
#include "cuda-core/light.cuh"
#include "cuda-core/host_scene.cuh"
#include "cuda-core/rng.cuh"

#include <cassert>
#include <chrono>
#include <cstdint>
#include <algorithm>
#include <memory>
#include <fstream>
#include <filesystem>
#include <string>
#include <vector>
#include <queue>
#include <ranges>

// --------------------------------------------------------------------------
// Timing to stdout
// --------------------------------------------------------------------------
class AvgAndTotalTimer {
 public:
  // Alpha determines the "weight" of the most recent tick vs history.
  // 0.1 is standard (10% weight to new sample, 90% to history).
  // Higher alpha = faster response to changes, less smoothing.
  explicit AvgAndTotalTimer(const double smoothingFactor = 0.1)
      : m_totalDuration(0),
        m_averageDuration(0.0),
        m_alpha(std::max(0.0, std::min(1.0, smoothingFactor))),
        m_isFirstTick(true) {
    reset();
  }

  // Resets the start time to now.
  // Does NOT clear the history/total (use clear() for that if needed).
  void reset() { m_startTime = std::chrono::steady_clock::now(); }

  // Completely resets the timer state (clears totals and averages).
  void clearStats() {
    m_totalDuration = std::chrono::milliseconds(0);
    m_averageDuration = 0.0;
    m_isFirstTick = true;
    reset();
  }

  // Calculates elapsed time since last reset/tick.
  // Updates the Total and the Exponential Moving Average.
  // Resets the anchor time for the next interval.
  void tick() {
    const auto now = std::chrono::steady_clock::now();

    // Calculate exact duration as double for precision in EMA
    std::chrono::duration<double, std::milli> elapsed = now - m_startTime;
    double currentMillis = elapsed.count();

    // Update Total
    // We cast to integer milliseconds for the total accumulator
    m_totalDuration += std::chrono::duration_cast<std::chrono::milliseconds>(
        now - m_startTime);

    // Update Exponential Moving Average
    if (m_isFirstTick) {
      m_averageDuration = currentMillis;
      m_isFirstTick = false;
    } else {
      // EMA Formula: NewAvg = (Value * alpha) + (OldAvg * (1 - alpha))
      m_averageDuration =
          (currentMillis * m_alpha) + (m_averageDuration * (1.0 - m_alpha));
    }

    // Reset start time to now, so the next tick measures the next interval
    m_startTime = now;
  }

  // Returns the exponential moving average in milliseconds
  uint64_t avgMillis() const {
    return static_cast<uint64_t>(m_averageDuration);
  }

  // Returns the total accumulated time of all ticks in milliseconds
  uint64_t elapsedMillis() const {
    return static_cast<uint64_t>(m_totalDuration.count());
  }

  // Helper: Get the average as a double (if you need sub-millisecond precision)
  double avgMillisPrecise() const { return m_averageDuration; }

 private:
  std::chrono::steady_clock::time_point m_startTime;
  std::chrono::milliseconds m_totalDuration;
  double m_averageDuration;  // Stored as double to prevent integer truncation
                             // errors during math
  double m_alpha;            // Smoothing factor (0.0 to 1.0)
  bool m_isFirstTick;
};

// --------------------------------------------------------------------------
// Upload Functions
// --------------------------------------------------------------------------
DeviceHaltonOwen* copyHaltonOwenToDeviceAlloc(uint32_t blocks,
                                              uint32_t threads);

// TODO remove when switching to shapes/BVH
TriangleSoup triSoupFromTriangles(const HostTriangleScene& hostScene,
                                  uint32_t const bsdfCount,
                                  size_t maxTrianglesPerChunk = 1'000'000);
BSDF* deviceBSDF(std::vector<BSDF> const& h_bsdfs);
void deviceLights(std::vector<Light> const& h_lights,
                  std::vector<Light> const& h_infiniteLights, Light** d_lights,
                  Light** d_infiniteLights);
DeviceCamera* deviceCamera(DeviceCamera const& h_camera);

// --------------------------------------------------------------------------
// General Utils
// --------------------------------------------------------------------------
float4* deviceOutputBuffer(uint32_t const width, uint32_t const height);

std::filesystem::path getExecutableDirectory();

template <bool PreferHighTPB = false>
__host__ void optimalOccupancyFromBlock(void* krnl, uint32_t smemBytes,
                                        bool residentOnly, uint32_t& blocks,
                                        uint32_t& threads) {
  cudaDeviceProp deviceProp{};
  CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, 0));

  struct OccupancyMeasure {
    int blocks, threads;
    bool operator<(OccupancyMeasure const& other) const {
      if constexpr (PreferHighTPB)
        return threads < other.threads;
      else
        return blocks < other.blocks;
    }
  };
  std::priority_queue<OccupancyMeasure> occupancies;
  // instead of using cudaDeviceProp.maxThreadsPerBlock, cap it to 512 as
  // it would hurt occupancy due to register pressure
  auto multiples = std::views::iota(1, 16) |
                   std::views::transform([](int i) { return i * 32; }) |
                   std::views::reverse;  // 512, 480, 448 ... 32
  for (uint32_t t : multiples) {
    OccupancyMeasure current{.blocks = 0, .threads = (int)t};
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &current.blocks, krnl, current.threads, smemBytes));
    // round down to multiple of number of SMs
    current.blocks = (current.blocks / deviceProp.multiProcessorCount) *
                     deviceProp.multiProcessorCount;
    // account for current only if we don't care about residency of threads
    // or blocks per multiprocessor fits inside SM
    if (!residentOnly ||
        deviceProp.maxThreadsPerMultiProcessor >=
            current.blocks / deviceProp.multiProcessorCount * current.threads) {
      occupancies.emplace(current);
    }
  }
  blocks = occupancies.top().blocks;
  threads = occupancies.top().threads;
}

void writeOutputBuffer(float4 const* d_outputBuffer, uint32_t const width,
                       uint32_t const height, char const* name = "output.bmp",
                       bool isHost = false);
void writeOutputBufferRowMajor(float4 const* outputBuffer, uint32_t const width,
                               uint32_t const height,
                               char const* name = "output.bmp");

// --------------------------------------------------------------------------
// Scenes
// --------------------------------------------------------------------------
void cornellBox(HostTriangleScene* h_scene, std::vector<Light>* h_lights,
                std::vector<Light>* h_infiniteLights,
                std::vector<BSDF>* h_bsdfs, DeviceCamera* h_camera);

#endif