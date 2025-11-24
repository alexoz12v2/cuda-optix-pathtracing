#include "platform/platform-memory.h"

struct WindowsPlatformMemory
{
    /** Set to true if we encounters out of memory. */
    static DMT_PLATFORM_API bool bIsOOM;

    /** Set to size of allocation that triggered out of memory, zero otherwise. */
    static DMT_PLATFORM_API uint64_t OOMAllocationSize;

    /** Set to alignment of allocation that triggered out of memory, zero otherwise. */
    static DMT_PLATFORM_API uint32_t OOMAllocationAlignment;

    static DMT_FORCEINLINE void* Memzero(void* Dest, size_t Count) { return memset(Dest, 0, Count); }

    class PlatformVirtualMemoryBlock : public dmt::BasicVirtualMemoryBlock
    {
    public:
        PlatformVirtualMemoryBlock() {}

        PlatformVirtualMemoryBlock(void* InPtr, uint32_t InVMSizeDivVirtualSizeAlignment) :
        BasicVirtualMemoryBlock(InPtr, InVMSizeDivVirtualSizeAlignment)
        {
        }
        PlatformVirtualMemoryBlock(PlatformVirtualMemoryBlock const& Other)            = default;
        PlatformVirtualMemoryBlock& operator=(PlatformVirtualMemoryBlock const& Other) = default;

        DMT_PLATFORM_API void Commit(size_t InOffset, size_t InSize);
        DMT_PLATFORM_API void Decommit(size_t InOffset, size_t InSize);
        DMT_PLATFORM_API void FreeVirtual();

        DMT_FORCEINLINE void CommitByPtr(void* InPtr, size_t InSize)
        {
            Commit(size_t(((uint8_t*)InPtr) - ((uint8_t*)Ptr)), InSize);
        }

        DMT_FORCEINLINE void DecommitByPtr(void* InPtr, size_t InSize)
        {
            Decommit(size_t(((uint8_t*)InPtr) - ((uint8_t*)Ptr)), InSize);
        }

        DMT_FORCEINLINE void Commit() { Commit(0, GetActualSize()); }

        DMT_FORCEINLINE void Decommit() { Decommit(0, GetActualSize()); }

        DMT_FORCEINLINE size_t GetActualSize() const
        {
            return VMSizeDivVirtualSizeAlignment * GetVirtualSizeAlignment();
        }

        static DMT_PLATFORM_API PlatformVirtualMemoryBlock
            AllocateVirtual(size_t Size, size_t InAlignment = PlatformVirtualMemoryBlock::GetVirtualSizeAlignment());
        static DMT_PLATFORM_API size_t GetCommitAlignment();
        static DMT_PLATFORM_API size_t GetVirtualSizeAlignment();
    };

    static DMT_PLATFORM_API dmt::PlatformMemoryConstants const& GetConstants();
    [[noreturn]] static DMT_PLATFORM_API void                   OnOutOfMemory(uint64_t Size, uint32_t Alignment);
};

typedef WindowsPlatformMemory PlatformMemory;