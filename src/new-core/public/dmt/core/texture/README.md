# texture

While it's fine that CPU uses EWA filtering software implemented, CUDA should use whatever is natively available on
the hardware (still, TODO later)

This module should only contain functionality to treat a buffer of data as a signal to be sampled and processed, given
its layout and format, disregarding how was it loaded into memory

this takes bits and pieces from `core-texture`, and should have **No Relation** with parsing, as this only uses
computation
on already loaded textures (possibly, only parsing loads this to convert from linear to morton and opposite)
