import shutil
import argparse
from pathlib import Path
import sys
import os

# Add the parent directory of `pyshared` to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pyshared import dll_names


def copy_cuda_libs(cuda_path: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    lib_files = dll_names.dll_files()

    lib_dirs = [
        cuda_path / "bin",
        cuda_path / "lib64",
    ]  # Common CUDA library directories

    for lib_dir in lib_dirs:
        if lib_dir.exists():
            for pattern in lib_files:
                for file in lib_dir.glob(pattern):
                    shutil.copy(file, output_dir)
                    print(
                        f"Copied {file.resolve()} -> {output_dir.resolve() / file.name}"
                    )


def main():
    parser = argparse.ArgumentParser(
        description="Copy CUDA shared libraries (cudart & nvrtc)"
    )
    parser.add_argument("cuda_path", type=Path, help="Path to the CUDA installation")
    parser.add_argument("output_dir", type=Path, help="Directory to copy libraries to")
    args = parser.parse_args()

    if not args.cuda_path.exists():
        raise ValueError(f"Path {str(args.cuda_path)} doesn't exist")

    if not args.output_dir.exists():
        raise ValueError(f"Path {str(args.output_dir)} doesn't exist")

    copy_cuda_libs(args.cuda_path, args.output_dir)


if __name__ == "__main__":
    main()

# Example Usage (windows powershell):
# py -3.11 .\scripts\copy_dlls.py "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\" "..\"
