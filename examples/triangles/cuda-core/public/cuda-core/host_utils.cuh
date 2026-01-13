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

// MinBlocks used when preferring TPB
template <bool PreferHighTPB = false, int MinBlocksPerSM = 1,
          int MinThreads = 32>
__host__ void optimalOccupancyFromBlock(void* krnl, uint32_t smemBytes,
                                        bool residentOnly,
                                        uint32_t& totalBlocks,
                                        uint32_t& threadsPerBlock) {
  static_assert(MinThreads % 32 == 0, "MinThreads must be a multiple of 32");

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

  struct Measure {
    int blocksPerSM;
    int threads;

    bool operator<(Measure const& o) const {
      if constexpr (PreferHighTPB) {
        if (threads != o.threads) return threads < o.threads;
        return blocksPerSM < o.blocksPerSM;
      } else {
        if (blocksPerSM != o.blocksPerSM) return blocksPerSM < o.blocksPerSM;
        return threads < o.threads;
      }
    }
  };

  std::priority_queue<Measure> pq;

  for (int threads = 512; threads >= 32; threads -= 32) {
    // 🔹 Enforce minimum threads only when optimizing for blocks
    if constexpr (!PreferHighTPB) {
      if (threads < MinThreads) continue;
    }

    Measure m{};
    m.threads = threads;

    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &m.blocksPerSM, krnl, threads, smemBytes));

    if (m.blocksPerSM < MinBlocksPerSM) continue;

    if (residentOnly) {
      if (m.blocksPerSM * threads > prop.maxThreadsPerMultiProcessor) continue;
    }

    pq.push(m);
  }

  if (pq.empty()) {
    totalBlocks = 0;
    threadsPerBlock = 0;
    return;
  }

  auto best = pq.top();
  threadsPerBlock = best.threads;
  totalBlocks = best.blocksPerSM * prop.multiProcessorCount;
}

void writeOutputBuffer(float4 const* d_outputBuffer, uint32_t const width,
                       uint32_t const height, char const* name = "output.bmp",
                       bool isHost = false);
void writeOutputBufferRowMajor(float4 const* outputBuffer, uint32_t const width,
                               uint32_t const height,
                               char const* name = "output.bmp");
#if DMT_ENABLE_MSE
__host__ void writeMeanAndMSERowMajor(float4 const* mean,
                                      float4 const* deltaSqr, uint32_t width,
                                      uint32_t height, std::string baseName);
#endif

// --------------------------------------------------------------------------
// Scenes
// --------------------------------------------------------------------------
void cornellBox(HostTriangleScene* h_scene, std::vector<Light>* h_lights,
                std::vector<Light>* h_infiniteLights,
                std::vector<BSDF>* h_bsdfs, DeviceCamera* h_camera);

#endif
