# cuda-optix-pathtracing

Implementing a simple path tracer in C++ CUDA

(Despite the name, we are not using OptiX anymore)

- CUDA Toolkit Version used: 12.6
- Operating Systems Supported: Windows, Linux (tested Ubuntu 24.04)

## Windows Setup

- Download Visual Studio 2022 and make sure to have the MSVC v143 (`Microsoft.VisualStudio.Component.VC.Tools.x86.x64`)
  at least version `14.42.34433` installed. 
- Download CUDA GPU Computing Toolkit 12.6 and install it in its default path
- Download the Windows SDK 10 (Expected Version: `10.0.22621.0`)
- Install ninja, either from Visual Studio or standalone (With `winget`) (at least version 1.11)
- Install cmake, either bundled from your IDE or from `winget`. Used Version: minimum `4.1.0`
- Install OptiX 8.0 (not used but still required in the build)
- Install FBX SDK 2020.3.7 (default location)

If you want to use a IDE which expects an environment properly setup, open a powershell version and use the provided
scripts to setup the environment properly

Example

```powershell
& "Y:\cuda-optix-pathtracing\scripts\resetenv.ps1"| &"Y:\cuda-optix-pathtracing\scripts\applyenv.ps1"
& "C:\Program Files\JetBrains\CLion 2025.2.3\bin\clion64.exe"
```

Command to properly configure the cmake build project with the MSVC toolchain

```shell
cmake "-DCMAKE_BUILD_TYPE=Debug" "-DCMAKE_MAKE_PROGRAM=/path/to/ninja.exe" \
  "-DCMAKE_C_COMPILER=path/to/cl.exe" -G Ninja "-DCMAKE_BUILD_TYPE=Debug" \
  "-DCMAKE_TOOLCHAIN_FILE=cmake/Windows.MSVC.toolchain.cmake" "-DCMAKE_POLICY_VERSION_MINIMUM=3.5" \
  -S "proj/dir" -B "your/binary/dir"
```

Note that the compiler variable is not strictly necessary since the toolchain file should take care of prepping cache
variables.

WARNING: `scripts\resetenv.ps1` and `cmake\Windows.MSVC.toolchain.cmake` should be kept in sync

## Linux setup

### Install `cmake` and `llvm` libraries

For the newer versions on Ubuntu/Debian-like distributions with `apt`, using LLVM apt repository and kitware's apt repository to install `cmake` and `clang`, `clang-format`, `clang-tidy` is a good idea.

also, you need to install `doxygen` and `graphviz`, and `ninja-build`

So, when all repositories are properly setup (LLVM: <https://apt.llvm.org/>, kitware: <https://apt.kitware.com/>)

```sh
sudo apt-get install cmake clang clang-format clang-tidy lldb lld doxygen graphviz ninja-build patchelf
```

### Install the clang++ 18/14 compiler

- clang-18 This is the maximum version supported by nvcc 13
- clang-14 This is the maximum version supported by nvcc 12 (use unsupported compiler flag to raise this)

```sh
wget -qO- https://apt.llvm.org/llvm.sh | sudo bash -s -- 18
# verify with 
clang++-18 --version
# this is the compiler in CMakePresets.json for Debug-Linux
```

We need to make sure that it associates itself with the correct headers (C++20 headers), hence try the command

```sh
clang++-18 -v -E -stdlib=libc++ -x c++ /dev/null
# look for found candidates ...
```

If it uses headers from a gcc version which doesn't support C++20, eg gcc 13, we need to use other C++ headers, like the libc++ ones
from llvm, which can be installed on a debian like distribution with

```sh
sudo apt-get update # make sure to have the LLVM apt repository
sudo apt install libc++-18-dev libc++abi-18-dev -y
```

### Install NVIDIA drivers and CUDA Toolkit

Then you need to check whether the current Graphics driver is the NVIDIA proprietary one, hence try to run

```sh
nvidia-smi # shows information at a glance for each NVIDIA GPU, usages and processes
dpkg -l | grep nvidia-driver # list debian package (apt is built on top of dpkg), should show nvidia-driver-575 or similar
```

To check whether cuda is installed

```sh
nvcc --version
echo $CUDA_HOME
echo $PATH | grep cuda
```

Check if CUDA driver api is present inside `/usr/lib/x86_64-linux-gnu` or `/usr/lib/nvidia-xxx/`, in particular files

- `libcuda.so` Driver API, required for cuda.h programs linking with `-lcuda`
- `libGLX_nvidia.so` OpenGL interop
- `libnvidia-ml.so` NVML (monitoring library)

Then you might want to check the CUDA libraries, like the CUDA runtime (`libcudart.so`).

The typical installation path for CUDA is `/usr/local/cuda`, which is a symlink to `/usr/local/cuda-xx-x` (version).
If absent, follow <https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#package-manager-installation> (network repository installation)

- In particular, install the `runfile (local)` for CUDA Toolkit 12.6 (without installing drivers `--no-drm`)

Therefore finally reboot the system and perform the post installment actions <https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#post-installation-actions>

Where to apply configuration:

- `~/.bashrc` Applies to your user whenever you open a new terminal (in particular in `$HOME/.local/bin/env` if dot sourced)
- `~/.zshrc` Same, if you use zsh
- `/etc/profile.d/cuda.sh` System-wide for all users (requires `sudo`)

Or, if you prefer, such environment setup can be done on a per session basis with our script

```sh
bash # enter a nested console process, such that exit can clean up the environment
source scripts/setup_env.sh --optix-dir $path --autodesk-fbx-sdk-dir $anotherPath
```


### Setup environment Script

Note: `--optix-dir` and `--autodesk-fbx-sdk-dir` are to be removed

```sh
bash
source scripts/setup_env.sh --optix-dir ~/optix/optix8.0/ --autodesk-fbx-sdk-dir ~/autodesk-fbx-sdk/2020.3.7/ --pre-turing
exit
```

### Configure Command Linux

First time

```sh
git submodule init
git submodule update --init --recursive extern/implot
```

At the start of every terminal session

```sh
source scripts/setup_env.sh --optix-dir ~/optix/optix8.0/ --autodesk-fbx-sdk-dir ~/autodesk-fbx-sdk/2020.3.7 # example
```

### Linux CMake configure command line 

```shell
# within the environment
export CUDAHOSTCXX=/usr/bin/clang++-18
cmake -G Ninja -S . -B cmake-build-rel-lto -DCMAKE_MAKE_PROGRAM=ninja \ 
  -DCMAKE_C_COMPILER=clang-18 -DCMAKE_CXX_COMPILER=clang++-18 -DCMAKE_CUDA_HOST_COMPILER=clang++-18 \
  -DDMT_NVCC_MAXREGCOUNT=64 -DDMT_DEVICE_LINK_TIME_OPTIMIZATION=ON -DCUDAToolkit_ROOT=/usr/local/cuda-12.8 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc
```

## Linux deal with corrupted coverage files

```sh
find build/ -name "*.gcda" -delete
```

Then use `cmake --build` with flag `--clean-first`
