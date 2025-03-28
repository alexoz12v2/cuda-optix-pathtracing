from pathlib import Path
import argparse
import sys
import os

# Add the parent directory of `pyshared` to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pyshared import dll_names


def which_lib_from_string(value: str) -> dll_names.EWhichLib:
    match value:
        case "cudart":
            return dll_names.EWhichLib.eCudart
        case "nvrtc":
            return dll_names.EWhichLib.eNvrtc


def find_first_matching_file(directory: str, pattern: str) -> str:
    """Find the first file matching a pattern in the given directory."""
    path = Path(directory)

    if not path.is_dir() or not path.exists():
        raise ValueError(f"Provided path '{directory}' is not a valid directory.")

    for subpath in ["", "bin", "lib64"]:
        p = path if subpath == "" else path / subpath
        if p.exists() and p.is_dir():
            matches = sorted(
                p.glob(pattern),
                key=lambda f: (".alt." in f.name, f.name),  # Prioritize non-alt files
            )
            if matches:
                break

    return str(matches[0]) if matches else None  # Return the best match or None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find the first matching shared library in a directory"
    )
    parser.add_argument(
        "-d", "--dir", required=True, type=str, help="Path to the directory to search"
    )
    parser.add_argument(
        "-l",
        "--lib",
        required=True,
        choices=["cudart", "nvrtc"],
        help="System Library to find the file for",
    )
    args = parser.parse_args()
    result = find_first_matching_file(
        args.dir, dll_names.dll_files(which_lib_from_string(args.lib))
    )

    if result:
        print(result)
    else:
        print("No matching file. Command Line: " + " ".join(sys.argv), file=sys.stderr)


if __name__ == "__main__":
    main()
