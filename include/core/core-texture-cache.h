#pragma once

#include "core/core-macros.h"
#include "core/core-texture.h"

namespace dmt {
    /// - its static method creates, from the `ImageTexturev2` object, a temporary file which will be deleted
    ///   upon process termination. Such a file stores a header of metadata about the uncompressed image, such as
    ///    - pixel format (and size)
    ///    - resolution, number of mip levels
    ///   Followed by the data
    /// - its `readMip` method takes the mip level and outputs a `void*` buffer of the specific pixel format, provided a
    ///   long enough buffer (a call with `nullptr` can estimate its size)
    /// - It is thread safe without any explicit synchronization except for the fact that, if a `MipCacheFile` instance
    ///   exists in any thread, the file is locked and cannot be modified/deleted by anything else in the system
    class MipCacheFile
    {
    };

    /// LRU cache which maintains
    /// - a hash table which associates {file path, mip level} -> address of allocated buffer,
    ///   where the allocated buffer is the desired mip of a texture. Such a buffer is managed by taking a large portion of
    ///   memory, whose size is predefined at start, and creates a pool allocator whose base element is a tile of 32x32 bytes,
    ///   since we know we are caching large texture
    /// - a singly linked list, allocated in an arena, whose maximum capacity is determined by the maximum number of entries
    ///   the cache can have (computed as a worst case scenario from maximum bytes and tile size)
    ///   This list keeps track of the LRU order, such that the least recently used can be deleted with a `pop_front`
    class TextureCache
    {
    };
} // namespace dmt
