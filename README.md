# cuda-optix-pathtracing

Implementing a simple path tracer in C++ CUDA using OptiX

- CUDA Toolkit Version used: 11.8
- Operating Systems Supported: Windows, Linux (tested Ubuntu 24.04)

## Install [TEV Display server](github.com/Tom94/tev) and play around with `pbrt`

`pbrt` uses `tev` to display its image during rendering (as an alternative to `glfw`). If both `tev` and `pbrt` are on the path, then you can execute (Powershell)

- windows

    ```powershell
    Start-Process -FilePath "tev" -ArgumentList "--hostname","127.0.0.1:14158"
    ```

- linux

    '''sh
    tev --hostname 127.0.0.1:14158 &
    ```

Once you have a `tev` server up and running,

```sh
# (GPU version)
pbrt --gpu --log-level verbose --display-server 127.0.0.1:14158 .\villa-daylight.pbrt
# (CPU version)
pbrt --wavefront --log-level verbose --display-server 127.0.0.1:14158 .\villa-daylight.pbrt
```

Alternatively, `pbrt` can also display to a native, `glfw` based window with the `--interactive` option.
(one of `--interactive` and `--display-server <addr:port>` can be used, not both). It's laggy so I don't reccomend it.

## Linux setup

### Install `cmake` and `llvm` libraries

For the newer versions on Ubuntu/Debian-like distributions with `apt`, using LLVM apt repository and kitware's apt repository to install `cmake` and `clang`, `clang-format`, `clang-tidy` is a good idea.

also, you need to install `doxygen` and `graphviz`, and `ninja-build`

So, when all repositories are properly setup (LLVM: <https://apt.llvm.org/>, kitware: <https://apt.kitware.com/>)

```sh
sudo apt-get install cmake clang clang-format clang-tidy lldb lld doxygen graphviz ninja-build patchelf
```

### Install the clang++ 20 compiler

This is the maximum version supported by nvcc 13

```sh
wget -qO- https://apt.llvm.org/llvm.sh | sudo bash -s -- 20
# verify with 
clang++-20 --version
# this is the compiler in CMakePresets.json for Debug-Linux
```

We need to make sure that it associates itself with the correct headers (C++20 headers), hence try the command

```sh
clang++-20 -v -E -stdlib=libc++ -x c++ /dev/null
# look for found candidates ...
```

If it uses headers from a gcc version which doesn't support C++20, eg gcc 13, we need to use other C++ headers, like the libc++ ones
from llvm, which can be installed on a debian like distribution with

```sh
sudo apt-get update # make sure to have the LLVM apt repository
sudo apt install libc++-20-dev libc++abi-20-dev -y
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

- In particular, install the `runfile (local)` for CUDA Toolkit 11.8 (without installing drivers `--no-drm`)

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

### Install Optix 8.0

From this link <https://developer.nvidia.com/designworks/optix/downloads/legacy> pick Optix 8.0

Then move into the directory you want to install optix in, eg `~/optix/optix8.0`

Example:

```sh
chmod 755 ~/Downloads/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64.sh
cd ~/optix/optix8.0 # --optix-dir for the setup_env.sh script
~/Downloads/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64.sh --exclude-subdir
```

### Install Autodesk FBX

Follow procedure from <https://help.autodesk.com/cloudhelp/2018/ENU/FBX-Developer-Help/getting_started/installing_and_configuring/configuring_the_fbx_sdk_for_linux.html>

Choose version **2020.3.7**, then extract it to destination. Example:

```sh
mkdir -p ~/autodesk-fbx-sdk/2020.3.7
mkdir -p ~/autodesk-fbx-sdk/installer/2020.3.7
tar -xzf ~/Downloads/fbx202037_fbxsdk_gcc_linux.tar.gz -C  ~/autodesk-fbx-sdk/installer/2020.3.7
chmod +x ~/autodesk-fbx-sdk/installer/2020.3.7/fbx202037_fbxsdk_linux
~/autodesk-fbx-sdk/installer/2020.3.7/fbx202037_fbxsdk_linux ~/autodesk-fbx-sdk/2020.3.7/
# if you want lib files to be read write by owner
cd ~/autodesk-fbx-sdk/202.3.7
chmod 644 lib/release/*.a lib/release/*.so lib/debug/*.a lib/debug/*.so
```

### Setup environment Script

```sh
bash # enter new terminal
source scripts/setup_env.sh --optix-dir ~/optix/optix8.0/ --autodesk-fbx-sdk-dir ~/autodesk-fbx-sdk/2020.3.7 # example
exit # when you're finished or want to change something
```

### Graphics packages for GLFW

The simplest solution is to just install them system-wide with `apt` (more difficult to maintain than building them from source)

```sh
sudo apt install libwayland-dev libxkbcommon-dev xorg-dev
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

Then everytime (add `--fresh` to cleanup previus `cmake` runs)

```sh
cmake .. --preset Debug-Linux -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DGLFW_BUILD_WAYLAND=ON -DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc
```

### Ninja Requirements

We require `ninja` with version >= 1.11 (C++20 module support)

## Linux deal with corrupted coverage files

```sh
find build/ -name "*.gcda" -delete
```

Then use `cmake --build` with flag `--clean-first`
