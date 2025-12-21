#ifndef DMT_PLATFORM_PUBLIC_PLATFORM_MEMORY_STACKALLOCATOR_H
#define DMT_PLATFORM_PUBLIC_PLATFORM_MEMORY_STACKALLOCATOR_H

#include "platform-macros.h"
#include "platform-memory.h"

#include <cstdint>

namespace dmt {

class VirtualStackAllocator;

enum class EVirtualStackAllocatorDecommitMode : uint8_t {
  // Default mode, does not decommit pages until the allocator is destroyed
  AllOnDestruction = 0,
  // Decommit all pages once none are in use
  AllOnStackEmpty = 1,
  // Tracks the high water mark and uses it to free excess memory that is not
  // expected to be used again soon
  ExcessOnStackEmpty = 2,
  numModes
};

struct ScopedStackAllocatorBookmark {
 public:
  ~ScopedStackAllocatorBookmark();

 private:
  friend class VirtualStackAllocator;

  ScopedStackAllocatorBookmark(void* InRestorePointer,
                               VirtualStackAllocator* Owner)
      : RestorePointer(InRestorePointer), Owner(Owner) {}

  ScopedStackAllocatorBookmark() = delete;
  ScopedStackAllocatorBookmark(ScopedStackAllocatorBookmark const&) = delete;
  ScopedStackAllocatorBookmark& operator=(ScopedStackAllocatorBookmark const&) =
      delete;
  ScopedStackAllocatorBookmark(ScopedStackAllocatorBookmark&&) = delete;
  ScopedStackAllocatorBookmark& operator=(ScopedStackAllocatorBookmark&&) =
      delete;

  void* RestorePointer;
  VirtualStackAllocator* Owner;
};

class VirtualStackAllocator {
  VirtualStackAllocator(size_t RequestedStackSize,
                        EVirtualStackAllocatorDecommitMode Mode);
  VirtualStackAllocator();

  ~VirtualStackAllocator();

  DMT_FORCEINLINE ScopedStackAllocatorBookmark CreateScopedBookmark() {
    return ScopedStackAllocatorBookmark(NextAllocationStart, this);
  }

  void* Allocate(size_t Size, size_t Alignment);

  size_t GetAllocatedBytes() const {
    return (char*)NextAllocationStart - (char*)VirtualMemory;
  }

  size_t GetCommittedBytes() const {
    return (char*)NextUncommittedPage - (char*)VirtualMemory;
  }

 private:
  VirtualStackAllocator(VirtualStackAllocator const& Other) = delete;
  VirtualStackAllocator(VirtualStackAllocator&& Other) = delete;
  void operator=(VirtualStackAllocator const& Other) = delete;
  void operator=(VirtualStackAllocator&& Other) = delete;

  friend ScopedStackAllocatorBookmark;

  // VirtualStackAllocator();

  DMT_FORCEINLINE void Free(void* RestorePointer) {
    if (RestorePointer <= NextAllocationStart) abort();

    NextAllocationStart = RestorePointer;

    if (NextAllocationStart == VirtualMemory &&
        DecommitMode != EVirtualStackAllocatorDecommitMode::AllOnDestruction)
        [[unlikely]] {
      DecommitUnusedPages();
    }
  }

  // Frees unused pages according to the current decommit mode
  void DecommitUnusedPages();

  void* VirtualMemory;
  void* NextUncommittedPage = nullptr;
  void* NextAllocationStart = nullptr;

  size_t TotalReservationSize;
  size_t const PageSize;

  EVirtualStackAllocatorDecommitMode DecommitMode =
      EVirtualStackAllocatorDecommitMode::AllOnDestruction;
  void* RecentHighWaterMark = nullptr;
};
}  // namespace dmt
#endif  // DMT_PLATFORM_PUBLIC_PLATFORM_MEMORY_STACKALLOCATOR_H
