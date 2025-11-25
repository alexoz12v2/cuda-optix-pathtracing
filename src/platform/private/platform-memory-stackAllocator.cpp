#include "platform-memory-stackAllocator.h"
#include "platform-memory.h"
#include "platform-utils.h"

namespace dmt {

    template <typename T>
    static inline T* OffsetPointer(T* Start, size_t Offset)
    {
        return reinterpret_cast<T*>(reinterpret_cast<intptr_t>(Start) + Offset);
    }

    template <typename T, typename U>
    static inline ptrdiff_t PointerDifference(T* End, U* Start)
    {
        return static_cast<ptrdiff_t>(reinterpret_cast<intptr_t>(End) - reinterpret_cast<intptr_t>(Start));
    }

    ScopedStackAllocatorBookmark::~ScopedStackAllocatorBookmark()
    {
        if (RestorePointer != nullptr)
        {
            assert(Owner != nullptr);
            Owner->Free(RestorePointer);
        }
    }

    VirtualStackAllocator::VirtualStackAllocator(size_t RequestedStackSize, EVirtualStackAllocatorDecommitMode Mode) :
    PageSize(os::systemAlignment()),
    DecommitMode(Mode)
    {
        TotalReservationSize = dmt::Align(RequestedStackSize, PageSize);

        if (TotalReservationSize > 0)
        {
            VirtualMemory       = os::reserveVirtualAddressSpace(TotalReservationSize);
            NextUncommittedPage = VirtualMemory;
            NextAllocationStart = NextUncommittedPage;
            RecentHighWaterMark = NextUncommittedPage;
        }
    }

    VirtualStackAllocator::~VirtualStackAllocator()
    {
        assert(GetAllocatedBytes() == 0);
        if (NextUncommittedPage != nullptr)
        {
            os::freeVirtualAddressSpace(VirtualMemory, TotalReservationSize);
        }
    }

    void* VirtualStackAllocator::Allocate(size_t Size, size_t Alignment)
    {
        void* const AllocationStart = dmt::Align(NextAllocationStart, Alignment);
        if (Size > 0)
        {
            void* const AllocationEnd   = OffsetPointer(AllocationStart, Size);
            void* const UsableMemoryEnd = OffsetPointer(VirtualMemory, TotalReservationSize - PageSize);

            if (AllocationEnd > UsableMemoryEnd)
            {
                //FPlatformMemory::OnOutOfMemory(Size, Alignment);
            }

            // After the high water mark is established, needing to commit pages should be rare
            if ((AllocationEnd > NextUncommittedPage)) [[unlikely]]
            {
                // We need to commit some more pages. Let's see how many
                uintptr_t RequiredAdditionalCommit = PointerDifference(AllocationEnd, NextUncommittedPage);
                // CommitByPtr doesn't round up the size for you
                size_t SizeToCommit = dmt::Align(RequiredAdditionalCommit, PageSize);
                os::commitPhysicalMemory(NextUncommittedPage, SizeToCommit);

                //LLM_IF_ENABLED(FLowLevelMemTracker::Get().OnLowLevelAlloc(ELLMTracker::Default, NextUncommittedPage, SizeToCommit, LLM_TAG_NAME(VirtualStackAllocator)));

                NextUncommittedPage = dmt::Align(AllocationEnd, PageSize);
            }

            if ((char*)AllocationEnd > (char*)RecentHighWaterMark)
            {
                RecentHighWaterMark = dmt::Align(AllocationEnd, PageSize);
            }

            NextAllocationStart = AllocationEnd;
        }

        return AllocationStart;
    }

    void VirtualStackAllocator::DecommitUnusedPages()
    {
        // This should only be called when the allocator is empty
        assert(NextAllocationStart == VirtualMemory);

        if (DecommitMode == EVirtualStackAllocatorDecommitMode::AllOnStackEmpty)
        {
            os::decommitPhysicalMemory(VirtualMemory, PointerDifference(NextUncommittedPage, VirtualMemory));
            NextUncommittedPage = VirtualMemory;
        }
        else if (DecommitMode == EVirtualStackAllocatorDecommitMode::ExcessOnStackEmpty)
        {
            // In this mode, each time we get down to zero memory in use we consider decommitting some of the memory above the most recent high water mark
            ptrdiff_t AmountToFree = (intptr_t)NextUncommittedPage - (intptr_t)RecentHighWaterMark;

            // We will only decommit memory if it would free up at least 25% of the current commit. This helps prevent us from thrashing pages if our
            // memory usage is consistant but not exactly constant and ensures we only pay to decommit if it will actually result in a significant savings
            ptrdiff_t MinimumToDecommit = PointerDifference(NextUncommittedPage, VirtualMemory) >> 2;
            if (AmountToFree > MinimumToDecommit)
            {
                // We have used less memory this time than the last time, decommit the excess
                os::decommitPhysicalMemory(RecentHighWaterMark, AmountToFree);
                NextUncommittedPage = RecentHighWaterMark;
            }
        }
        RecentHighWaterMark = VirtualMemory;
    }
} // namespace dmt