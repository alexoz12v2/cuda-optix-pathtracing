# platform

This layer should contain only Host side, OS Specific utilities

This takes from the still used bits from the `platform-*` functionality

- `platform-utils`
    - `os::Path`
- `platform-file` and `core-texture-cache` are merged and reorganized such that the logic to dump data into files
  is separated from the specifics of generating mip levels
- `platform-threadPool` is transferred as-is
- `platform-memory-*` is killed
- `platform-context` and `platform-logging` are completely removed and reworked into an easier `logging` interface,
  more similiar to `std::cout`
