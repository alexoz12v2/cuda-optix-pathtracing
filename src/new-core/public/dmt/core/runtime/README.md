# runtime

in here there should be initialization code for each type of rendering backend

| Layer        | What it is                   | What it does                                               | What it must NOT do           |
|--------------|------------------------------|------------------------------------------------------------|-------------------------------|
| **Platform** | OS / driver abstraction      | Files, logging, threads, CUDA runtime API                  | Rendering, policy, algorithms |
| **Runtime**  | Process-level state & policy | Initialization, global configuration, capability detection | Rendering, OS abstraction     |

```text
platform/
  file.*
  logging.*
  memory.*
  thread_pool.*
  os_utils.*
  cuda_platform.*   // thin wrapper over CUDA runtime (renderer type agnostic CUDA Initialisation)
```

- Platform is stateless

Instead, runtime is stateful.

“What global state does this program need?”

Runtime answers:

- Which CPU features are enabled?
- How many threads do we use?
- Is CUDA enabled? Which device?
- What is the global allocator?
- What backends are available?

Examples

```text
runtime/
cpu_runtime.*
cuda_runtime.*
runtime.h
```

Runtime is:

- Application-specific
- Owns initialization order
- Chooses policies
- Sets global knobs

Runtime does NOT:

- Perform rendering
- Contain algorithms
- Contain kernels
