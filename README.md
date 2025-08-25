# cuda-optix-pathtracing
Implementing a simple path tracer in C++ CUDA using OptiX

## Workflow
- La repository \`e suddivisa in `src`, `examples`, `test`, `include`
- Le cartelle `include` e `src` contengono una sottocartella per ogni modulo C++20, al quale corrisponde un
  CMake target. Si definiscono nelle due cartelle rispettivamente Module Interface Units e Module Implementation 
  Units relative al modulo
- (WIndows) usare powershell 7 (che non e' [bundled con windows by default](https://powershellexplained.com/2017-12-29-Powershell-what-is-pwsh/)) `winget install microsoft.powershell`
### CMake Configuration
Dalla directory della repository
```
mkdir build
cd build
```
In seguito, se da Windows, allora (`Debug-VS` esempio di configurazione, vedere `CMakePresets.json`)
```
cmake .. --preset Debug-VS
```
Su Linux
```
cmake .. --preset Debug-Linux
```
### Building the Project
Una volta aver generato il progetto con un generator default, settato dai presets (`ninja` per Linux, 
`Visual Studio 17 2022` per Windows), 
- Aprire la soluzione generata nella cartella `/build` se su Windows
- Fare la build o da IDE `CLion` o da linea di comando su linux
In particolare, una build da linea di comando, dalla directory del progetto, dell'intero progetto
```
cmake --build --preset Debug-Linux
```
Oppure per fare la build di un solo cmake target, esempio `testdmt`
```
cmake --build --preset Debug-Linux --target testdmt
```

### Running Tests
Si usa la libreria [Catch2](https://github.com/catchorg/Catch2/tree/v2.x) assieme a 
[Fake Function Framework](https://github.com/meekrosoft/fff/tree/master) per i tests, dunque vengono creati
dei cmake targets.
Se l'IDE non lo supporta, si possono lanciare i tests da linea di comando grazie agli eseguibili forniti da Catch2
``` 
/path/a/cuda-optix-pathtracing/build/${preset}/bin/${test-target} --rng-seed=2312312 -r xml -d yes --order lex
```
Esempio di output di un test case:
```
<?xml version="1.0" encoding="UTF-8"?>
<Catch2TestRun name="dmt-test-testdmt" rng-seed="213123" xml-format-version="3" catch2-version="3.7.0">
  <TestCase name="[testdmt] Test Case for testing" filename="/mnt/SSD_Data/uni/cuda-optix-pathtracing/test/testdmt/projtest.test.cpp" line="9">
    <Section name="Main stuff" filename="/mnt/SSD_Data/uni/cuda-optix-pathtracing/test/testdmt/projtest.test.cpp" line="11">
      <OverallResults successes="3" failures="0" expectedFailures="0" skipped="false" durationInSeconds="8e-06"/>
    </Section>
    <OverallResult success="true" skips="0" durationInSeconds="5.8e-05"/>
  </TestCase>
  <OverallResults successes="3" failures="0" expectedFailures="0" skips="0"/>
  <OverallResultsCases successes="1" failures="0" expectedFailures="0" skips="0"/>
</Catch2TestRun>
```
### Docs Generation
Dopo la generazione, devi fare la build del target `dmt-doxygen`
```
cmake --build --preset Debug-Linux --target dmt-doxygen
```

## Tasks
- [x] Completamento classe `Platform`
  - [x] Alessio: Logging
  - [ ] Alessio: Memory Allocation
  - [x] Anto: ThreadPool = 1 Thread IO + N Workers, Workers ascolta una lock-free queue
        [Job Scheduling Talk](https://www.youtube.com/watch?v=HIVBhKj7gQU), 
        [Job Scheduling Slides](https://www.createursdemondes.fr/wp-content/uploads/2015/03/parallelizing_the_naughty_dog_engine_using_fibers.pdf)
        [Atomics (Queue)](https://www.youtube.com/watch?v=ZQFzMfHIxng)
  - [x] Anto: Display = funzione OpenGL per mostrare una texture a schermo
        [BufferDisplay](https://github.com/mmp/pbrt-v4/blob/88645ffd6a451bd030d062a55a70a701c58a55d0/src/pbrt/gpu/cudagl.h#L64)
  - [ ] Anto(*): integrare parsing della struttura `pbrt`, [Link al Parser](https://github.com/mmp/pbrt-v4/blob/88645ffd6a451bd030d062a55a70a701c58a55d0/src/pbrt/parser.h#L109)
- [ ] Modulo delle classi di modello (sia SoA che AoS)
  - [ ] BVH Tree
  - [ ] Surface Interaction
  - [ ] classi base Ray, AABB, Vector, Matrix
  - [ ] Funzioni Shading: BRDF, Texture Mapping, Mapping Spettro -> RGB
- [ ] Alessio: Migliorie agli script di building (vedi [Link](https://cmake.org/cmake/help/latest/module/FindCUDA.html))
  - [x] `add_custom_target` per copiare cartella `assets/`
  - [x] supporto CUDA `add_cuda_library` (deprecato da cmake 3.27, ora CUDA first class citizen)
  - [ ] di default cmake setta il CUDA compiler per compute capability 5.2. Settare la 6.1
  - [ ] integrazione CUDA check compatibility with cmake, [link](https://github.com/mmp/pbrt-v4/blob/88645ffd6a451bd030d062a55a70a701c58a55d0/cmake/checkcuda.cu#L4)
  - [ ] supporto per librerie dinamiche `add_library(${target} SHARED)`
  - [ ] far funzionare gli script su Windows su un altro build tool diverso da VS
- [ ] Implementazione con OptiX
  - [ ] Formazione
- [ ] GUI imgui implot
- [ ] MiddleWare e Memoria ([Riferimento File Format PBRT](https://pbrt.org/fileformat-v4))
  - [ ] (Anto) Inserimento nella classe `SceneDescription` dei metodi necessari per la gestione del GraphicsState (`scene.h` in PBRT)
  - [ ] (Alessio) Algoritmo di pseudo random number generation sia per `__host__` che `__device__` (xorshift)
  - [ ] Completamento delle classi di modello, versione standard e versione SOA, (dove entrambe devono 
      usare allocatori di memoria Device `BaseMemoryResource` (MemPool, Buddy), mentre l'allocatore stesso
      e la struttura dati stessa stanno in Managed Memory, `UnifiedMemoryResource`)
      La prima classe di modello e' `Spectrum` (servono anche per completare il parse del world block)
      - [ ] (Alessio) `Spectrum` -> direttiva `LightSource`
      - [ ] (Anto) `Light`
      - [ ] (Alessio) `Texture`
      - [ ] `Shape` (...)
      - [ ] (Anto) `SurfaceInteraction`
      - [ ] (Anto) `Sampler` (RNG)
      - [ ] (Anto) `Filter`
      - [ ] (Anto) `Film`
      - [ ] (Alessio) `Camera` -> (perspective)
      - [ ] (Alessio) `BSDF`
 - [ ] (Alessio) Testing delle strutture dati preesistenti (`DynaArray`) con piu' allocatori
 - [ ] (Anto) Gestione della direttiva `Shape` (Parte 1: Memorizzazione della mesh in un file binario intermedio)
 - [ ] (Anto) Gestione in `SceneParser` delle direttive `Import` e `Include`, dove
     - `Import`: Aggiunge nella `SceneDescription` tutte le "Named Entities" (Textures, Materials, Medium), Compare solo nel World Block (`m_parsingStep == EParsingStep::eWorld`)
     - `Include`: Semanticamente equivalente a fare copy-paste del file indicato nel file corrente. Specifica una path relativa alla `m_basePath`, e aggiunge un file al `m_fileStack`.
 - [ ] (Alessio) Scrittura di una classe `BVH`, costruita in un file binario a partire dai vari pezzi della scena,
      caricata livello per livello con la `cuFile API` in una cache in device memory ([paper](https://dcgi.fel.cvut.cz/home/bittner/publications/cag2014.pdf))

 
### Appunti e links su CUDA memory allocation
- introduzione sull'argomento: https://stackoverflow.com/questions/73155788/efficient-reallocation-of-cuda-memory
- Stream ordered allocations: https://developer.nvidia.com/blog/using-cuda-stream-ordered-memory-allocator-part-1/
- link esempi github 1: https://github.com/NVIDIA-developer-blog/code-samples/blob/master/posts/cuda-vmm/cuvector.cpp
- link esempi github 2: https://github.com/eyalroz/cuda-api-wrappers/blob/master/src/cuda/api/context.hpp#L131o
- link esempi github 3: https://github.com/NVIDIA/cuda-samples/blob/9c688d7ff78455ed42e345124d1495aad6bf66de/Samples/2_Concepts_and_Techniques/streamOrderedAllocation/streamOrderedAllocation.cu#L126

### Appunti e links su CUDA debugging
- Accendi il programma "NVIDIA NSight Monitor"
- Documentazione [Qui](https://docs.nvidia.com/nsight-visual-studio-edition/cuda-debugger/)
- Dopo la build, su Visual Studio, "Extensions" -> "NSight" -> "Start CUDA Debuggging (Next Gen)"

### Promemoria vari
- vedere il C runtime linkato/richiesto da MSVC in un `.obj`
```
dumpbin /ALL middleware-model.cu.obj | findstr "Runtime"
# esempio di output
   /FAILIFMISMATCH:RuntimeLibrary=MDd_DynamicDebug
```

### VSCode Notes
#### Intellisense and CMake
To make sure that VSCode can see CMake's `target_compile_definitisons` and more, the build system needs to export a *JSON Compilation Database*, which
is an array of compilation commands specified by LLVM [Here](https://clang.llvm.org/docs/JSONCompilationDatabase.html). (file `compile_commands.json` in the build directory).
CMake supports creation of the compilation database out of the box by setting, either in the `CMakePresets.json`, by command line, or inside a script file, 
the variable [`CMAKE_EXPORT_COMPILE_COMMANDS`](https://cmake.org/cmake/help/latest/variable/CMAKE_EXPORT_COMPILE_COMMANDS.html), *Only* for Ninja Generators and Makefile Generators.
For the Visual Studio Generator, ie MSBuild, we need a different approach illustrated [Here](https://clangpowertools.com/blog/generate-json-compilation-database.html).
- Install Visual Studio extension "Clang Power Tools"
- Right click on the Solution inside "Solution Explorer"
- Click "Clang Power Tools" -> "Export Compilation Database"

Once the compilation database is generated, in the build directory, there should be a `compile_commands.json` file.
To let vscode see this, create inside the `.vscode` directory a symlink to the `compile_commands.json` file. 
(Insert your actual absolute path as target argument. Showing windows cmd command)
```
mklink .\.vscode\compile_commands.json Y:\cuda-optix-pathtracing\build\Debug-VS\compile_commands.json  
```
Then insert inside the `.vscode/c_cpp_properties.json` the following
```json
{
  "configurations": [
    {
      // ...
      "compileCommands": "${workspaceFolder}/.vscode/compile_commands.json"
    }
  ]
  // ...
}
```
*Remember to refresh the compilation database everytime you run cmake's configure step* (symlink has to be created once)

#### Debugging
- (Linux/MacOS) you need "CMake Tools" and ["Nsight Visual Studio Code Edition"](https://docs.nvidia.com/nsight-visual-studio-code-edition/install-setup/index.html) and use the following `.vscode/launch.json` file
  ```json
  {
      "version": "0.2.0",
      "configurations": [
          {
              "name": "CUDA C++: Launch",
              "type": "cuda-gdb",
              "request": "launch",
              "program": "${workspaceFolder}/build/path/to/stuff.exe"
          }
      ]
  }
  ```
  Then you can launch a cmake executable target in the "CMake" tab, right click and "Debug"
- (Windows) You can't. Use Visual Studio

#### Seeing modules
I still don't know

### Strumenti Extra per il `Ninja Multi-Config` Generator
- necessario avere ninja
  ```
  winget install ninja-build.ninja
  ```
- scaricare utility `vswhere` e metterla nella env `Path`, [Link](https://github.com/microsoft/vswhere)
- Configurare command prompt con l'ambiente necessario. Su windows:
  ```
  cmd.exe /k "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Auxiliary\\Build\\vcvarsall.bat" x64
  ```
  Che puo' essere messo in `.vscode\settings.json` per averlo pronto

### clang tidy
Apparently, the built-in support for `clang-tidy` doesn't work on a per cmake target basis out of the box with the target property `CXX_CLANG_TIDY`, because
the `clang-tidy` executable is called as part of the build process, before the `.ifc` or `.pcm` file has a chance to be generated.
So the only alternative is to add `clang-tidy` as a separete tool. We therefore need a script which can take into account the current compilation database `compile_commands.json`.
Therefore, download and put on the `Path` the [`run-clang-tidy`](https://github.com/lmapii/run-clang-tidy/tree/main) python script
And then run the command (showing an example build configuration)
```
run-clang-tidy --build-root build\Debug-WinNinja\  tidy.json
```
This will try to check everything, so if some C++20 modules have not been compiled, it will give an error for those

## Installazione CUDA OptiX
- Navigare su [Questo Link](https://developer.nvidia.com/designworks/optix/download) e scaricare la SDK 8.0.0
- Se lo installi in una directory diversa da
  ```
  (Win32) "C:\ProgramData/NVIDIA Corporation/OptiX SDK 8.0.0"
  (Unix)  "~/NVIDIA-OptiX-SDK-8.0.0-linux64"
  ```
  Allora verificare che la variabile di ambiente `OPTIX80_PATH` sia settata correttamente
- se la variabile di ambiente `OPTIX80_PATH` non viene settata in installazione, settarla manualmente

## Linking da un exsecutable target a CUDA
Ogni cmake target di tipo library non possiede l'opzione di `nvlink` "`-dlink`", la quale triggera il linking con il device code.
Dunque sta agli executable targets performare questo step aggiuntivo. Il problema e' che se un executable target non possiede nessuna sorgente `.cu`,
il linker chiamato di default sara' quello della C++ toolchain in uso piuttosto di quello di CUDA, causando un linker error del tipo
```
error LNK2019: unresolved external symbol __cudaRegisterLinkedBinary_988c167c_11_cudaTest_cu_68a51f74 
```
- Un fix possibile e' quello di aggiungere nel target un file `.cu` vuoto

## Install [TEV Display server](github.com/Tom94/tev) and play around with `pbrt`
`pbrt` uses `tev` to display its image during rendering (as an alternative to `glfw`). If both `tev` and `pbrt` are on the path, then you can execute (Powershell)
```
(powershell)
Start-Process -FilePath "tev" -ArgumentList "--hostname","127.0.0.1:14158"
(linux)
tev --hostname 127.0.0.1:14158 &
```
Once you have a `tev` server up and running,
```
(GPU version)
pbrt --gpu --log-level verbose --display-server 127.0.0.1:14158 .\villa-daylight.pbrt
(CPU version)
pbrt --wavefront --log-level verbose --display-server 127.0.0.1:14158 .\villa-daylight.pbrt
```

Alternatively, `pbrt` can also display to a native, `glfw` based window with the `--interactive` option. 
(one of `--interactive` and `--display-server <addr:port>` can be used, not both). It's laggy so I don't reccomend it.

### Compilation Problem on windows

- no CUDA toolset found: make sure that all files from `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\extras\visual_studio_integration\MSBuildExtensions\` (substisute you own CUDA directory) are contained in `C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Microsoft\VC\v170\BuildCustomizations\`

## Linux setup

### Install `cmake` and `llvm` libraries

For the newer versions on Ubuntu/Debian-like distributions with `apt`, using LLVM apt repository and kitware's apt repository to install `cmake` and `clang`, `clang-format`, `clang-tidy` is a good idea.

also, you need to install `doxygen` and `graphviz`, and `ninja-build`

So, when all repositories are properly setup (LLVM: <https://apt.llvm.org/>, kitware: <https://apt.kitware.com/>)

```sh
sudo apt-get install cmake clang clang-format clang-tidy lldb lld doxygen graphviz ninja-build
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

Example for Ubuntu 24.04 (noble) (use `lsb_release --all` to know your Ubuntu information)

```sh
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb # download deb file to current directory
sudo dpkg -i cuda-keyring_1.1-1_all.deb # debian package manager to install the downloaded file
sudo apt-get update # update apt repositories
sudo apt install cuda-toolkit
```

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
