import platform
from enum import Enum


class EWhichLib(Enum):
    eAll = -1
    eCudart = 0
    eNvrtc = 1


def dll_files(which_lib: EWhichLib = EWhichLib.eAll) -> list[str] | str:
    system = platform.system()
    match system:
        case "Windows":
            match which_lib:
                case EWhichLib.eAll:
                    return ["cudart64_*.dll", "nvrtc64_*.dll"]
                case EWhichLib.eCudart:
                    return "cudart64_*.dll"
                case EWhichLib.eNvrtc:
                    return "nvrtc64_*.dll"
        case "Linux":
            match which_lib:
                case EWhichLib.eAll:
                    return ["libcudart.so*", "libnvrtc.so*"]
                case EWhichLib.eCudart:
                    return "libcudart.so*"
                case EWhichLib.eNvrtc:
                    return "libnvrtc.so*"
        case _:
            raise RuntimeError(f"Unsupported OS: {system}")
