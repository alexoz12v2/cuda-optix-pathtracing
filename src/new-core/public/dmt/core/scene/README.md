# core scene

Files implementing Scene structures and asset layer, directly parsed from disk or from any other parsing strategy

- plain structs
- no CUDA keyword
- no OS code
- no SIMD
- no virtuals
- no allocation policies

This replaces part of the old code layout

- `core-parser`
- `core-mesh-parser`
- `core-material`
- `core-light`
- `core-primitive`
