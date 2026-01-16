import bpy
import sys

# usage: blender -b <scene file> -P bpy_set_samples.py -- <N>

# Get samples from command line after "--"
argv = sys.argv
argv = argv[argv.index("--") + 1:] if "--" in argv else []
if not argv:
  raise RuntimeError("No sample count provided")

samples = int(argv[0])

scene = bpy.context.scene

# ensure cycles
scene.render.engine = 'CYCLES'

# disable adaptive sampling for fairness
scene.cycles.use_adaptive_sampling = False

# set samples
scene.cycles.samples = samples

# determinism for benchmarking and no denoising
scene.cycles.use_denoising = False
scene.cycles.seed = 0

# Render frame 1
bpy.ops.render.render(write_still=True)

print(f"Rendered with Samples: {samples}")

