#include "platform-logging.h"

#include <unistd.h>
#if defined(_POSIX_ASYNCHRONOUS_IO)
#include <aio.h> // https://man7.org/linux/man-pages/man7/aio.7.html https://www.gnu.org/software/libc/manual/html_node/Asynchronous-I_002fO.html
#else
#error "Only supported implementation uses aio.h"
#endif

namespace dmt {
    static_assert(sizeof(aiocb) <= 168);
    struct PaddedAioCb
    {
        aiocb         acb;
        unsigned char padding[sizeof(AioSpace) - sizeof(aiocb)];
    };

    static_assert(sizeof(PaddedAioCb) == sizeof(AioSpace) && alignof(PaddedAioCb) == alignof(AioSpace));
    static_assert(sizeof(PaddedAioCb) == 256 && std::is_trivial_v<PaddedAioCb> && std::is_standard_layout_v<PaddedAioCb>);

    // TODO add proper memory management
    static_assert(std::is_standard_layout_v<LinuxAsyncIOManager> && sizeof(LinuxAsyncIOManager) == asyncIOClassSize);

    LinuxAsyncIOManager::LinuxAsyncIOManager() :
    m_aioQueue(reinterpret_cast<AioSpace*>(std::aligned_alloc(alignof(PaddedAioCb), numAios * sizeof(PaddedAioCb)))),
    m_lines(reinterpret_cast<Line*>(std::aligned_alloc(alignof(Line), numAios * sizeof(Line))))
    {
        if (!m_aioQueue || !m_lines)
            std::abort();

        initAio();
    }

    void LinuxAsyncIOManager::initAio()
    {
        // Initialize the aiocb structures
        for (uint32_t i = 0; i < numAios; ++i)
        {
            PaddedAioCb& paddedAioCb = *reinterpret_cast<PaddedAioCb*>(&m_aioQueue[i]);
            std::memset(&m_aioQueue[i], 0, sizeof(PaddedAioCb));
            paddedAioCb.acb.aio_fildes     = STDOUT_FILENO;
            paddedAioCb.acb.aio_offset     = 0; // Default to write operation
            paddedAioCb.acb.aio_buf        = reinterpret_cast<volatile void*>(m_lines[i].buf);
            paddedAioCb.acb.aio_lio_opcode = LIO_WRITE; // Default to write operation
            //paddedAioCb.aio_reqprio    =, scheduling priority. Requires additional macros
            paddedAioCb.acb.aio_sigevent.sigev_notify = SIGEV_NONE; // SIGEV_NONE, SIGEV_SIGNAL -> sigev_signo, SIGEV_THREAD -> sigev_notify_attributes, specify thread: (linux only) sigev_notify_thread_id (used only by timers)
            paddedAioCb.acb.aio_lio_opcode = LIO_WRITE; // only used by lio_listio when you schedule multiple operations
        }
    }

    void LinuxAsyncIOManager::cleanup() noexcept
    {
        sync();
        std::free(m_aioQueue);
        std::free(m_lines);
    }

    LinuxAsyncIOManager::LinuxAsyncIOManager(LinuxAsyncIOManager&& other) noexcept :
    m_aioQueue(std::exchange(other.m_aioQueue, nullptr)),
    m_lines(std::exchange(other.m_lines, nullptr))
    {
    }

    LinuxAsyncIOManager::~LinuxAsyncIOManager() noexcept { cleanup(); }

    LinuxAsyncIOManager& LinuxAsyncIOManager::operator=(LinuxAsyncIOManager&& other) noexcept
    {
        if (this != &other)
        {
            cleanup();
            m_aioQueue = std::exchange(other.m_aioQueue, nullptr);
            m_lines    = std::exchange(other.m_lines, nullptr);
        }
        return *this;
    }

    char* LinuxAsyncIOManager::operator[](uint32_t idx) { return m_lines[idx].buf; }

    uint32_t LinuxAsyncIOManager::findFirstFreeBlocking()
    {
        // Ensure we complete previous operations before starting a new one
        // necessary only if we are switching from STDOUT to STDERR
        // syncIfNeeded(fildes); // we only support STDOUT

        // Find an available aiocb slot (simple round-robin or any available slot)
        while (true) // if everything is full, then poll
        {
            for (uint32_t i = 0; i < numAios; ++i)
            {
                PaddedAioCb& paddedAioCb = *reinterpret_cast<PaddedAioCb*>(&m_aioQueue[i]);
                if (aio_error(&paddedAioCb.acb) != EINPROGRESS)
                {
                    return i;
                }
            }
        }
    }

    bool handleStatus(int32_t status, aiocb& outAio, uint32_t maxAttempts)
    {

        assert(status != EBADF && "Asynchronous write File Descriptor invalid!");
        assert(status != EINVAL && "Invalid `aio_offset` or `aio_reqprio`!");
        uint32_t attempt = 1;
        for (; status == EAGAIN && attempt != maxAttempts; ++attempt)
        {
            status = aio_write(&outAio);
        }

        return attempt == maxAttempts;
    }

    bool LinuxAsyncIOManager::enqueue(uint32_t idx, size_t size)
    {
        // no sync needed as handled by findFirstFreeBlocking
        // the m_lines[idx].buf should be written externally
        // Find an available aiocb slot (simple round-robin or any available slot)
        PaddedAioCb& paddedAioCb = *reinterpret_cast<PaddedAioCb*>(&m_aioQueue[idx]);
        assert(aio_error(&paddedAioCb.acb) != EINPROGRESS);
        paddedAioCb.acb.aio_nbytes = size;
        int status                 = aio_write(&paddedAioCb.acb);
        return handleStatus(status, paddedAioCb.acb, maxAttempts);
    }

    void LinuxAsyncIOManager::sync() const
    {
        for (uint32_t i = 0; i != LinuxAsyncIOManager::numAios; ++i)
        {
            PaddedAioCb& paddedAioCb = *reinterpret_cast<PaddedAioCb*>(&m_aioQueue[i]);
            while (aio_error(&paddedAioCb.acb) == EINPROGRESS)
            {
                // busy waiting...
            }
        }
    }

    // LOGGING 2.0 ----------------------------------------------------------------------------------------------------
    LogHandler createConsoleHandler(LogHandlerAllocate _alloc, LogHandlerDeallocate _dealloc)
    {
        LogHandler handler{};
        handler.hostAllocate   = _alloc;
        handler.hostDeallocate = _dealloc;
        handler.hostFlush      = []() {};
        handler.hostFilter     = [](void* _data, LogRecord const& record) -> bool { return true; };
        handler.hostCallback   = [](void* _data, LogRecord const& record) {
            puts(std::bit_cast<char const*>(record.data));
        };
        handler.hostCleanup = [](LogHandlerDeallocate _dealloc, void* _data) {};
        return handler;
    }

} // namespace dmt
