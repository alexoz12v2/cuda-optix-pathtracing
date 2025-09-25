# Useful Commands

Cmake Format

```sh
cmake-format -i $(find ./ ./src ./cmake \
  -type f \( -name "CMakeLists.txt" -o -name "*.cmake" \) \
  -not -path "*/build/*")
```
