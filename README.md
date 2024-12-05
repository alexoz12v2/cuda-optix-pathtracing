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