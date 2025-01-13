# cuda-optix-pathtracing
Implementing a simple path tracer in C++ CUDA using Embree and OptiX

## Workflow
- La repository \`e suddivisa in `src`, `examples`, `test`, `include`
- Le cartelle `include` e `src` contengono una sottocartella per ogni modulo C++20, al quale corrisponde un
  CMake target. Si definiscono nelle due cartelle rispettivamente Module Interface Units e Module Implementation 
  Units relative al modulo
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