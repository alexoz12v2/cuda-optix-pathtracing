from collections import namedtuple
import json
from pprint import pprint
import re
import subprocess
import platform
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Tuple


def get_exports_windows(dll_path: Path) -> List[str]:
    """Extracts exported function names from a Windows DLL using dumpbin."""
    try:
        output = subprocess.check_output(
            ["dumpbin", "/EXPORTS", str(dll_path)], text=True, errors="ignore"
        )
    except FileNotFoundError:
        raise RuntimeError(
            "dumpbin not found. Ensure Visual Studio is installed and configured."
        )

    exports = []
    for line in output.splitlines():
        match = re.search(r"\s+\d+\s+[0-9A-Fa-f]+\s+[0-9A-Fa-f]+\s+(\w+)", line)
        if match:
            exports.append(match.group(1))

    return exports


def get_exports_linux(so_path: Path) -> List[str]:
    """Extracts exported function names from a Linux shared object using objdump."""
    try:
        output = subprocess.check_output(
            ["objdump", "-T", str(so_path)], text=True, errors="ignore"
        )
    except FileNotFoundError:
        raise RuntimeError("objdump not found. Ensure binutils is installed.")

    exports = []
    for line in output.splitlines():
        columns = line.split()
        if (
            len(columns) > 6 and columns[1] == "g" and "DF" in columns[2]
        ):  # Global function
            exports.append(columns[-1])

    return exports


def filter_latest_versions(exports, version_pattern):
    """Filters out older versions of functions based on a version pattern."""
    if not version_pattern:
        return {export: export for export in exports}  # Return as a dictionary {base_name: latest_version}

    version_regex = re.compile(version_pattern.replace("{n}", r"(\d+)"))
    latest_versions = {}

    for export in exports:
        match = version_regex.search(export)
        if match:
            base_name = export[:match.start()]
            version = int(match.group(1))

            # Check if base_name is in latest_versions and safely extract its version
            if base_name in latest_versions:
                existing_match = version_regex.search(latest_versions[base_name])
                existing_version = int(existing_match.group(1)) if existing_match else -1
            else:
                existing_version = -1  # Default to -1 so any valid version is higher

            # Store the highest version
            if version > existing_version:
                latest_versions[base_name] = export
        else:
            latest_versions[export] = export  # Function without versioning

    return latest_versions  # Return a dictionary { base_name: latest_version_name }


def generate_loader(library_path, includes, version_pattern, latest_only, json_file, header_name: str | None) -> Tuple[str, str]:
    """Generates C++ code for dynamically loading functions from a shared library."""
    header_string = ""
    implementation_string = ""
    is_windows = platform.system() == "Windows"
    exports = (
        get_exports_windows(library_path)
        if is_windows
        else get_exports_linux(library_path)
    )

    if latest_only:
        exports = filter_latest_versions(exports, version_pattern)

    library_name = Path(library_path).name
    library_class_name = library_name.replace('-', '_').capitalize()[:library_name.rfind('.')]

    implementation_string += f"#include \"{header_name}\"\n\n"
    implementation_string += """#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <cstdlib>

"""

    header_string += "\n" + "".join(f'#include "{inc}"\n' for inc in includes) + "\n"

    # Add the #undef macros
    for base_name in exports.keys():
        header_string += f"#ifdef {base_name}\n#undef {base_name}\n#endif\n"

    implementation_string += f"""

static void* libraryHandle = nullptr;

static void LoadLibraryOnce() {{
    if (!libraryHandle) {{
#ifdef _WIN32
        libraryHandle = LoadLibraryW(L"{library_name}");
#else
        libraryHandle = dlopen("{library_name}", RTLD_LAZY);
#endif
    }}
}}

static void UnloadLibrary() {{
    if (libraryHandle) {{
#ifdef _WIN32
        FreeLibrary(static_cast<HMODULE>(libraryHandle));
#else
        dlclose(libraryHandle);
#endif
        libraryHandle = nullptr;
    }}
}}

static void* LoadLibraryFunc(const char* func_name) {{
    LoadLibraryOnce();
    if (!libraryHandle) return nullptr;
#ifdef _WIN32
    return GetProcAddress(static_cast<HMODULE>(libraryHandle), func_name);
#else
    return dlsym(libraryHandle, func_name);
#endif
}}
"""
    header_string += f"""
class {library_class_name}LibraryFunctions {{
  public:
"""
    MethodData = namedtuple("MethodData", "name type latest_version")
    data: List[MethodData] = list()

    if json_file is not None:
        with json_file.open(encoding="UTF-8") as source:
            tmap = json.load(source)
            tmap = {k.lower(): v for k, v in tmap.items()}  # Case-insensitive mapping
    else:
        tmap = {}

    for base_name, latest_version in exports.items():
        # Use tmap lookup if available, otherwise default to void(*)()
        func_ptr_type = tmap.get(base_name.lower(), f"void(*)()")  # Default to function pointer
        t_sig = f"using {latest_version}_t = {func_ptr_type};"  # Using the latest version

        header_string += f"    {t_sig}\n"
        data.append(MethodData(base_name, t_sig.split(" ")[1], latest_version))

    header_string += "\n  public:\n"
    for methodData in data:
        header_string += f"    {methodData.type} {methodData.name};\n"

    header_string += "};\n"
    header_string += f"bool load{library_class_name}Functions({library_class_name}LibraryFunctions* funcList);\n"

    implementation_string += f"""
bool load{library_class_name}Functions({library_class_name}LibraryFunctions* funcList)
{{
"""
    for methodData in data:
        implementation_string += f"    funcList->{methodData.name} = reinterpret_cast<{library_class_name}LibraryFunctions::{methodData.type}>(LoadLibraryFunc(\"{methodData.latest_version}\"));\n"
        implementation_string += f"    if (!funcList->{methodData.name}) {{ return false; }}\n"
    
    implementation_string += "    return true;\n}\n"

    return (header_string, implementation_string)


# TODO add clang-format after generation
def main():
    parser = ArgumentParser(
        description="Generate a C++ wrapper for dynamically loading DLL/.so functions."
    )
    parser.add_argument(
        "library",
        type=Path,
        help="Path to the DLL/.so file to generate the wrapper for.",
    )
    parser.add_argument(
        "-i",
        "--includes",
        nargs="*",
        default=[],
        help="List of header files to include.",
    )
    parser.add_argument(
        "-v",
        "--version-pattern",
        type=str,
        default="_v{n}",
        help="Regex pattern for versioned function names.",
    )
    parser.add_argument(
        "-l",
        "--latest-only",
        action="store_true",
        help="Only export the latest version of each function.",
    )
    parser.add_argument(
        "-j",
        "--json-type-mapping",
        type=str,
        default="",
        help="Json which maps an exported symbol to a typedef/using declaration (fallback to void*)"
    )
    parser.add_argument(
        "-hf",
        "--header-file",
        type=str,
        required=True
    )
    parser.add_argument(
        "-cpp",
        "--cpp-file",
        type=str,
        required=True
    )
    # TODO 
    parser.add_argument(
        "-up",
        "--use-platform",
        action="store_true",
        help="Produce a translation unit which depends on the dmt-platform cmake target",
    )

    args = parser.parse_args()

    if not args.library.exists():
        print(f"Error: Library file '{args.library}' not found.")
        return
    
    header_file = Path(args.header_file)
    cpp_file = Path(args.cpp_file)
    header_name = header_file.name

    json_file = Path(args.json_type_mapping)
    if json_file.exists() and json_file.is_file():
        json_file = json_file.resolve()
    else:
        json_file = None

    header_string, implementation_string = generate_loader(
        args.library, args.includes, args.version_pattern, args.latest_only, json_file, header_name
    )

    with cpp_file.open("w", encoding="utf-8") as f:
        f.write(implementation_string)

    with header_file.open("w", encoding="utf-8") as f:
        f.write(header_string)


if __name__ == "__main__":
    main()

# Example Usage:
#  py -3.11 .\scripts\generate_dll_wrapper_file.py C:\Windows\System32\nvcuda.dll -i cuda.h -v "_v{n}" -l -j .\scripts\dll_wrapper_type_mapper_cuda_driver.json --cpp-file ..\stuff.cpp --header-file ..\stuff.h