from collections import namedtuple
import json
import re
import subprocess
import platform
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Tuple


MethodData = namedtuple("MethodData", "name type latest_version")
LibraryData = namedtuple("LibraryData", "drive search_path_expr")


def remove_json_comments(json_str: str) -> str:
    """Removes // and /* */ comments from a JSON-like string, preserving comments inside strings."""
    result = []
    in_string = False
    escape = False
    i = 0
    while i < len(json_str):
        char = json_str[i]

        if char == '"' and not escape:
            in_string = not in_string  # Toggle string state

        if not in_string:
            # Handle `//` single-line comments
            if json_str[i : i + 2] == "//":
                i = json_str.find("\n", i)  # Skip to end of line
                if i == -1:  # If no newline is found, exit
                    break
                continue

            # Handle `/* */` multi-line comments
            if json_str[i : i + 2] == "/*":
                i = json_str.find("*/", i)  # Find the closing */
                if i == -1:  # If no closing is found, exit
                    break
                i += 1  # Move past `*/`
                continue

        # Handle escape sequences inside strings
        escape = char == "\\" and in_string
        result.append(char)
        i += 1

    return "".join(result)


def remove_matching_quotes(s: str) -> str:
    # Check if the first and last characters are both quotes and are the same
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        # Remove both the first and last characters (the matching quotes)
        return s[1:-1]
    return s


def fix_backslashes(input_string: str) -> str:
    if platform.system() == "Windows":
        # We need to replace single backslashes with double backslashes, but leave existing double backslashes intact.
        result = []
        i = 0
        while i < len(input_string):
            result.append(input_string[i])
            if input_string[i] == "\\" and i + 1 < len(input_string):
                if input_string[i + 1] != "\\":
                    result.append("\\")
                    i += 1
                else:
                    result.append("\\")
                    i += 2
            else:
                i += 1
        return "".join(result)

    # If not Windows, return the input string unchanged
    return input_string


def get_header() -> str:
    return "// This file has been generated\n"


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
        return {
            export: export for export in exports
        }  # Return as a dictionary {base_name: latest_version}

    version_regex = re.compile(version_pattern.replace("{n}", r"(\d+)"))
    latest_versions = {}

    for export in exports:
        match = version_regex.search(export)
        if match:
            base_name = export[: match.start()]
            version = int(match.group(1))

            # Check if base_name is in latest_versions and safely extract its version
            if base_name in latest_versions:
                existing_match = version_regex.search(latest_versions[base_name])
                existing_version = (
                    int(existing_match.group(1)) if existing_match else -1
                )
            else:
                existing_version = -1  # Default to -1 so any valid version is higher

            # Store the highest version
            if version > existing_version:
                latest_versions[base_name] = export
        else:
            latest_versions[export] = export  # Function without versioning

    return latest_versions  # Return a dictionary { base_name: latest_version_name }


def write_type_declarations_and_populate_method_data(
    json_file, exports, data, header_string
):
    if json_file is not None:
        with json_file.open(encoding="UTF-8") as source:
            json_string = remove_json_comments(source.read())
            # uncomment to debug
            # with Path("Y:/why2.txt").open("w") as f:
            #     f.write("\n" + json_string)
            tmap = json.loads(json_string)
            tmap = {k.lower(): v for k, v in tmap.items()}  # Case-insensitive mapping
    else:
        # uncomment if you need to debug
        # with Path("Y:/why2.txt").open("w") as f:
        #     f.write("\nNOTHING")
        tmap = {}

    for base_name, latest_version in exports.items():
        # Use tmap lookup if available, otherwise default to void(*)()

        func_ptr_type = tmap.get(
            latest_version.lower(), tmap.get(base_name.lower(), "void(*)()")
        )  # Default to function pointer

        t_sig = (
            f"using {latest_version}_t = {func_ptr_type};"  # Using the latest version
        )

        header_string[0] += f"    {t_sig}\n"
        data.append(MethodData(base_name, t_sig.split(" ")[1], latest_version))


def prepare_library_population(
    library_dict: dict[str, Path], version_pattern, latest_only, macro_filter
):
    match platform.system():
        case "Windows":
            exports = get_exports_windows(library_dict["Windows"])
        case "Linux":
            exports = get_exports_linux(library_dict["Linux"])
        case _:
            raise ValueError("Unsupported platform")

    # TODO: cuGraphInstantiate -> cuGraphInstantiateWithFlags

    if latest_only:
        exports = filter_latest_versions(exports, version_pattern)
    else:
        export_set = set(exports)
        new_exports = {}

        for export in exports:
            # Skip exports that are already macro versions (i.e., ending with macro_filter)
            if macro_filter and export.endswith(macro_filter):
                continue

            # Try to find macro version
            if macro_filter:
                macro_version = f"{export}{macro_filter}"
                if macro_version in export_set:
                    new_exports[export] = macro_version
                    continue  # found macro version, good

            # If no macro version, fall back to normal export
            if export not in new_exports:
                new_exports[export] = export

        exports = new_exports

    library_name = Path(library_dict[platform.system()]).name
    library_class_name = library_name.replace("-", "_").capitalize()[
        : library_name.rfind(".")
    ]
    return exports, library_class_name


def append_class_header_termination(header_string, data):
    header_string[0] += "\n  public:\n"
    for methodData in data:
        header_string[0] += f"    {methodData.type} {methodData.name};\n"

    header_string[0] += "};\n"


def append_class_header_preamble(includes, header_string, exports, library_class_name, export_macro: str):
    header_string[0] += "\n" + "".join(f'#include "{inc}"\n' for inc in includes) + "\n"

    # Add the #undef macros
    for base_name in exports.keys():
        header_string[0] += f"#ifdef {base_name}\n#undef {base_name}\n#endif\n"

    if export_macro and len(export_macro) > 0:
        header_string[0] += f"""
class {export_macro} {library_class_name} {{
  public:
"""
    else:
        header_string[0] += f"""
class {library_class_name} {{
  public:
"""


def generate_loader(
    library_path: dict[str, Path],
    includes,
    version_pattern,
    latest_only,
    json_file,
    header_name: str | None,
    use_executable_dir: bool,
    export_macro: str,
    macro_filter,
    name
) -> Tuple[str, str]:
    """Generates C++ code for dynamically loading functions from a shared library."""
    header_string = [f"#pragma once\n{get_header()}"]
    implementation_string = [""]
    exports, library_class_name_NOT = prepare_library_population(
        library_path, version_pattern, latest_only, macro_filter
    )
    data: List[MethodData] = list()

    append_class_header_preamble(includes, header_string, exports, name, export_macro)
    write_type_declarations_and_populate_method_data(
        json_file, exports, data, header_string
    )
    append_class_header_termination(header_string, data)
    header_string[0] += (
        f"bool load{name}Functions({name}* funcList);\n"
    )

    implementation_string[0] += f'#include "{header_name}"\n{get_header()}\n'
    implementation_string[0] += """#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <cstdlib>

"""
    implementation_string[0] += f"""

static void* libraryHandle = nullptr;

static void LoadLibraryOnce() {{
    if (!libraryHandle) {{
#ifdef _WIN32
        libraryHandle = LoadLibraryW(L"{library_path["Windows"]}");
#else
        libraryHandle = dlopen("{library_path["Linux"]}", RTLD_LAZY);
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

    if export_macro and len(export_macro) > 0:
        implementation_string[0] += f"""
bool {export_macro} load{name}Functions({name}* funcList)
{{
"""
    else:
        implementation_string[0] += f"""
bool load{name}Functions({name}* funcList)
{{
"""
    for methodData in data:
        implementation_string[0] += (
            f'    funcList->{methodData.name} = reinterpret_cast<{name}::{methodData.type}>(LoadLibraryFunc("{methodData.latest_version}"));\n'
        )
        implementation_string[0] += (
            f"    if (!funcList->{methodData.name}) {{ return false; }}\n"
        )

    implementation_string[0] += "    return true;\n}\n"

    return (header_string[0], implementation_string[0])


def platform_generate_loader(
    library_dict: dict[str, Path],
    includes: List[str],
    version_pattern: str,
    latest_only: bool,
    json_file: Path,
    header_name: str | None,
    use_executable_dir: bool,
    export_macro: str,
    macro_filter,
    name
) -> Tuple[str, str]:
    header_string = [
        f'#pragma once\n{get_header()}\n#include "platform/platform-utils.h"\n'
    ]
    implementation_string = [f'#include "{header_name}"\n{get_header()}\n']
    exports, library_class_name_NOT = prepare_library_population(
        library_dict, version_pattern, latest_only, macro_filter
    )
    data: List[MethodData] = list()

    library_loading_dict: dict[str, LibraryData] = dict()

    for plat, library_path_str in library_dict.items():
        library_path = Path(library_path_str)
        search_path_parts = library_path.parent.parts
        library_loading_dict[plat] = LibraryData(
            '"' + library_path.drive + '"',
            (" / " if len(search_path_parts) > 0 else "")
            + " / ".join(f'"{s}"' for i, s in enumerate(search_path_parts) if i > 0),
        )

    append_class_header_preamble(includes, header_string, exports, name, export_macro)
    write_type_declarations_and_populate_method_data(
        json_file, exports, data, header_string
    )
    header_string[0] += "    void* m_library;\n"
    append_class_header_termination(header_string, data)
    if export_macro and len(export_macro) > 0:
        header_string[0] += (
            f"\nbool {export_macro} load{name}Functions(dmt::os::LibraryLoader const& loader, {name}* funcList);\n"
        )
    else:
        header_string[0] += (
            f"\nbool load{name}Functions(dmt::os::LibraryLoader const& loader, {name}* funcList);\n"
        )

    implementation_string[0] += (
        f"bool load{name}Functions(dmt::os::LibraryLoader const& loader, {name}* funcList) {{\n"
    )
    implementation_string[0] += "    using namespace std::string_view_literals;\n\n"

    def macro_from_plat(plat: str):
        match plat:
            case "Windows":
                return "DMT_OS_WINDOWS"
            case "Linux":
                return "DMT_OS_LINUX"

    plats = ["Windows", "Linux"]
    for i, plat in enumerate(plats):
        if i == 0:
            implementation_string[0] += f"#if defined({macro_from_plat(plat)})\n"
        else:
            implementation_string[0] += f"#elif defined({macro_from_plat(plat)})\n"

        library_name = library_dict[plat].name
        if use_executable_dir:
            implementation_string[0] += (
                "    dmt::os::Path path = dmt::os::Path::executableDir();\n"
            )
        else:
            implementation_string[0] += (
                f"    dmt::os::Path path = dmt::os::Path::root({library_loading_dict[plat].drive}){library_loading_dict[plat].search_path_expr};\n"
            )
        implementation_string[0] += (
            f'    funcList->m_library = loader.loadLibrary("{library_name}"sv, true, &path);\n'
        )
        if i == len(plats) - 1:
            implementation_string[0] += '#else\n#error "unsupported platform"\n#endif\n'

    implementation_string[0] += "    if (!funcList->m_library) { return false; }\n"
    for methodData in data:
        implementation_string[0] += (
            f'    funcList->{methodData.name} = reinterpret_cast<{name}::{methodData.type}>(dmt::os::lib::getFunc(funcList->m_library, "{methodData.latest_version}"));\n'
        )
        implementation_string[0] += (
            f"    if (!funcList->{methodData.name}) {{ return false; }}\n"
        )

    implementation_string[0] += "    return true;\n}\n"

    return (header_string[0], implementation_string[0])


# TODO add clang-format after generation
def main():
    print("Command-line arguments:", " ".join(sys.argv))
    # uncomment if you need to debug
    # with open("Y:/why.txt", "w") as f:
    #     f.write(" ".join(sys.argv))
    parser = ArgumentParser(
        description="Generate a C++ wrapper for dynamically loading DLL/.so functions."
    )
    parser.add_argument(
        "library",
        type=str,
        help="Path to the DLL/.so file to generate the wrapper for.",
    )
    parser.add_argument(
        "--name",
        type=str,
        help="name of the generated class.",
    )
    parser.add_argument(
        "-i",
        "--includes",
        nargs="*",
        default=[],
        help="space separated List of header files to include.",
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
        "-mf",
        "--macro-filter",
        type=str,
        default=None,
        help="String to be appended when searching for macro overridden symbol names",
    )
    parser.add_argument(
        "-j",
        "--json-type-mapping",
        type=str,
        default="",
        help="Json which maps an exported symbol to a typedef/using declaration (fallback to void*)",
    )
    parser.add_argument("-hf", "--header-file", type=str, required=True)
    parser.add_argument("-cpp", "--cpp-file", type=str, required=True)
    parser.add_argument(
        "-up",
        "--use-platform",
        action="store_true",
        default=False,
        help="Produce a translation unit which depends on the dmt-platform cmake target",
    )
    parser.add_argument(
        "-pexe",
        "--use-executable-path",
        action="store_true",
        default=False,
        help="If set, library lookup will use the executable directory to search for DLLs",
    )
    parser.add_argument(
        "-em",
        "--export-macro",
        required=False,
        type=str,
        default="",
        help="DLL import export macro which will be inserted on the class name and functions in the header"
    )

    args = parser.parse_args()

    json_string = fix_backslashes(remove_json_comments(args.library))
    # uncomment if you need to debug
    # with open("Y:/why.txt", "w+") as f:
    #     f.write(json_string)

    library_json = json.loads(json_string)
    library_json = {
        key: Path(remove_matching_quotes(value)) for key, value in library_json.items()
    }

    path = library_json[platform.system()]
    if not path.exists():
        raise ValueError(f"Path {path} doesn't exist")

    header_file = Path(remove_matching_quotes(args.header_file))
    cpp_file = Path(remove_matching_quotes(args.cpp_file))
    header_name = header_file.name
    # uncomment if you need to debug
    # with open("Y:/why.txt", "w+") as f:
    #     f.write(json_string)
    #     f.write("\n")
    #     f.write(str(header_file))
    #     f.write("\n")
    #     f.write(str(cpp_file))

    json_file = Path(remove_matching_quotes(args.json_type_mapping))
    # uncomment if you need to debug
    # with open("Y:/why.txt", "a+") as f:
    #     f.write(" ".join(sys.argv))
    #     f.write(args.json_type_mapping + "\n")
    #     f.write(str(json_file))

    if json_file.exists() and json_file.is_file():
        json_file = json_file.resolve()
    else:
        json_file = None

    # uncomment if you need to debug
    # with open("Y:/why.txt", "a+") as f:
    #     f.write("\n\nLATER: " + str(json_file))

    if args.use_platform:
        generate_loader_function = platform_generate_loader
    else:
        generate_loader_function = generate_loader

    header_string, implementation_string = generate_loader_function(
        library_json,
        args.includes,
        args.version_pattern,
        args.latest_only,
        json_file,
        header_name,
        args.use_executable_path,
        args.export_macro,
        args.macro_filter,
        args.name
    )

    cpp_file.parent.mkdir(parents=True, exist_ok=True)
    with cpp_file.open("w", encoding="utf-8") as f:
        f.write(implementation_string)

    header_file.parent.mkdir(parents=True, exist_ok=True)
    with header_file.open("w", encoding="utf-8") as f:
        f.write(header_string)


if __name__ == "__main__":
    main()

# Example Usage (windows powershell):
#  py -3.11 .\scripts\generate_dll_wrapper_file.py '{ "Windows": "C:\\Windows\\System32\\nvcuda.dll", "Linux": "" }' -i cuda.h -v "_v{n}" -l -j .\scripts\dll_wrapper_type_mapper_cuda_driver.json --cpp-file ..\stuff.cpp --header-file ..\stuff.h -up
#  py -3.11 .\scripts\generate_dll_wrapper_file.py "{ `"Windows`": `"$(${Env:\CUDA_PATH_V12_6}.replace('\','\\'))\\bin\\nvrtc64_120_0.dll `", `"Linux`": `"`" }" -i nvrtc.h -v "_v{n}" -l -j .\scripts\dll_wrapper_type_mapper_cuda_nvrtc.json --cpp-file ..\stuff.cpp --header-file ..\stuff.h -up
#  py -3.11 .\scripts\generate_dll_wrapper_file.py "{ `"Windows`": `"$(${Env:\CUDA_PATH_V12_6}.replace('\','\\'))\\bin\\cudart64_12.dll`", `"Linux`": `"`" }" -i cuda_runtime_api.h cuda_gl_interop.h -v "_v{n}" -l -j .\scripts\dll_wrapper_type_mapper_cuda_runtime.json --cpp-file ..\stuff2.cpp --header-file ..\stuff2.h -up
