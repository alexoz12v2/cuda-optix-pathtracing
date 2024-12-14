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
- [ ] Completamento classe `Platform`
  - [ ] Alessio: Logging
  - [ ] Alessio: Memory Allocation
  - [ ] Anto: ThreadPool = 1 Thread IO + N Workers, Workers ascolta una lock-free queue
        [Job Scheduling Talk](https://www.youtube.com/watch?v=HIVBhKj7gQU), 
        [Job Scheduling Slides](https://www.createursdemondes.fr/wp-content/uploads/2015/03/parallelizing_the_naughty_dog_engine_using_fibers.pdf)
        [Atomics (Queue)](https://www.youtube.com/watch?v=ZQFzMfHIxng)
  - [ ] Anto: Display = funzione OpenGL per mostrare una texture a schermo
        [BufferDisplay](https://github.com/mmp/pbrt-v4/blob/88645ffd6a451bd030d062a55a70a701c58a55d0/src/pbrt/gpu/cudagl.h#L64)
  - [ ] Anto(*): integrare parsing della struttura `pbrt`, [Link al Parser](https://github.com/mmp/pbrt-v4/blob/88645ffd6a451bd030d062a55a70a701c58a55d0/src/pbrt/parser.h#L109)
- [ ] Modulo delle classi di modello (sia SoA che AoS)
  - [ ] BVH Tree
  - [ ] Surface Interaction
  - [ ] classi base Ray, AABB, Vector, Matrix
  - [ ] Funzioni Shading: BRDF, Texture Mapping, Mapping Spettro -> RGB
- [ ] Alessio: Migliorie agli script di building (vedi [Link](https://cmake.org/cmake/help/latest/module/FindCUDA.html))
  - [ ] Link Time Optimization
  - [ ] `add_custom_target` per copiare cartella `assets/`
  - [ ] supporto CUDA `add_cuda_library` (deprecato da cmake 3.27, ora CUDA first class citizen)
  - [ ] supporto per librerie dinamiche `add_library(${target} SHARED)`
  - [ ] far funzionare gli script su Windows su un altro build tool diverso da VS
- [ ] Implementazione CPU con [Embree](https://www.embree.org/api.html)
  - [ ] costruzione della `RTCScene` a partire dalla rappresentazione `pbrt`
  - [ ] ...
- [ ] Implementazione con OptiX
  - [ ] Formazione
- [ ] GUI imgui implot