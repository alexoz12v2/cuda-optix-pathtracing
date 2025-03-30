import re

def list_kernels(ptx_file):
    with open(ptx_file, "r") as f:
        ptx = f.read()
    
    # Find all kernel entry points
    kernels = re.findall(r"\.entry\s+(\S+)", ptx)
    return kernels

kernels = list_kernels("my_kernel.ptx")
print("Kernels:", kernels)
