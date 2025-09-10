#!/bin/sh
# -----------------------------------------------------------------------------
# setup_env.sh
# Auto-detect latest CUDA under /usr/local/, optional LD_LIBRARY_PATH,
# set OPTIX80_PATH and FBX_SDK_PATH from command-line arguments
# Examples:
#   DEFINE_LD_LIBRARY_PATH=yes source setup_env.sh --optix-dir /path/to/OptiX-8.0 --autodesk-fbx-sdk-dir /path/to/FBX-SDK
#   source setup_env.sh --optix-dir /path/to/OptiX-8.0 --autodesk-fbx-sdk-dir /path/to/FBX-SDK
# -----------------------------------------------------------------------------

# -----------------------------
# Functions
# -----------------------------

check_dir_exists() {
    DIR="$1"
    NAME="$2"
    if [ -d "$DIR" ]; then
        echo "$NAME found: $DIR"
        return 0
    else
        echo "Warning: $NAME directory does not exist: $DIR"
        return 1
    fi
}

check_file_exists() {
    FILE="$1"
    MSG="$2"
    if [ -f "$FILE" ]; then
        echo "$MSG: $FILE exists"
        return 0
    else
        echo "Warning: $MSG missing: $FILE"
        return 1
    fi
}

prepend_path() {
    VAR_NAME="$1"
    NEW_PATH="$2"
    eval CURRENT_VAL=\$$VAR_NAME
    case ":$CURRENT_VAL:" in
        *":$NEW_PATH:"*) ;;
        *) export $VAR_NAME="$NEW_PATH:$CURRENT_VAL"; echo "Prepended $NEW_PATH to $VAR_NAME" ;;
    esac
}

manage_ld_library_path() {
    DIR="$1"
    ACTION="$2" # "add" or "remove"
    case "$ACTION" in
        add)
            case ":$LD_LIBRARY_PATH:" in
                *":$DIR:"*) echo "$DIR already in LD_LIBRARY_PATH" ;;
                *) export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$DIR"; echo "Added $DIR to LD_LIBRARY_PATH" ;;
            esac
            ;;
        remove)
            if [ -n "$LD_LIBRARY_PATH" ]; then
                OLD_LD="$LD_LIBRARY_PATH"
                LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v "$DIR" | paste -sd ":" -)
                export LD_LIBRARY_PATH
                echo "Removed $DIR from LD_LIBRARY_PATH"
                echo "Old: $OLD_LD"
                echo "New: $LD_LIBRARY_PATH"
            fi
            ;;
    esac
}

# -----------------------------
# Parse command-line arguments
# -----------------------------
OPTIX_ARG=""
FBX_ARG=""
PRE_TURING=0
while [ $# -gt 0 ]; do
    case "$1" in
        --optix-dir)
            shift
            OPTIX_ARG="$1"
            ;;
        --autodesk-fbx-sdk-dir)
            shift
            FBX_ARG="$1"
            ;;
        --pre-turing)
            PRE_TURING=1
            ;;
        *)
            echo "Warning: Unknown argument: $1"
            ;;
    esac
    shift
done

# -----------------------------
# Detect driver-supported CUDA version
# -----------------------------
DRIVER_CUDA=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}') # eg. "12.9"
CUDA_BASE="/usr/local"

if [ $PRE_TURING -eq 1 ]; then
    if [ -d "${CUDA_BASE}/cuda-11.8" ]; then
        echo "Using CUDA 11.8 due to --pre-turing flag"
        CUDA_DIR="${CUDA_BASE}/cuda-11.8"
    else
        echo "Warning: --pre-turing requested but ${CUDA_BASE}/cuda-11.8 not found"
        CUDA_DIR="" # fall back to normal logic below
    fi
fi

if [ -z "$CUDA_DIR" ] && [ -n "$DRIVER_CUDA" ]; then
    echo "Driver supports CUDA $DRIVER_CUDA"
    MAJOR_VERSION="${DRIVER_CUDA%.*}"

    # override: if major 12 then try minor 6
    if [ "$MAJOR_VERSION" = "12" ] && [ -d "${CUDA_BASE}/cuda-12.6" ]; then
        CUDA_DIR="${CUDA_BASE}/cuda-12.6"
    else
        CUDA_DIR="${CUDA_BASE}/cuda-${DRIVER_CUDA}"
    fi
fi

# Fallback: pick latest installed if the driver-specific one doesn’t exist
if [ -z "$CUDA_DIR" ] || [ ! -d "$CUDA_DIR" ]; then
    MAJOR_VERSION="${DRIVER_CUDA%.*}"
    CUDA_DIR="${CUDA_BASE}/cuda-${MAJOR_VERSION}"
fi

if [ -z "$CUDA_DIR" ] || [ ! -d "$CUDA_DIR" ]; then
    echo "Error: No CUDA installation found under $CUDA_BASE"
    return 1 2>/dev/null || exit 1
fi

export CUDA_HOME="$CUDA_DIR"
echo "Using CUDA installation: $CUDA_HOME"

# -----------------------------
# User option: LD_LIBRARY_PATH
# -----------------------------
DEFINE_LD_LIBRARY_PATH=${DEFINE_LD_LIBRARY_PATH:-"no"}
if [ "$DEFINE_LD_LIBRARY_PATH" = "yes" ]; then
    manage_ld_library_path "$CUDA_HOME/lib64" add
else
    manage_ld_library_path "$CUDA_HOME/lib64" remove
fi

# -----------------------------
# Update PATH for CUDA
# -----------------------------
prepend_path PATH "$CUDA_HOME/bin"

# -----------------------------
# Optional Nsight Compute
# -----------------------------
NSIGHT="/opt/nvidia/nsight-compute"
[ -d "$NSIGHT" ] && prepend_path PATH "$NSIGHT"

# -----------------------------
# Verify nvcc
# -----------------------------
if command -v nvcc >/dev/null 2>&1; then
    echo "nvcc found: $(nvcc --version | head -n 1)"
else
    echo "Warning: nvcc not found in PATH!"
fi

# -----------------------------
# Set OPTIX80_PATH
# -----------------------------
if [ -n "$OPTIX_ARG" ]; then
    if check_dir_exists "$OPTIX_ARG" "OptiX 8.0" && \
       check_file_exists "$OPTIX_ARG/doc/OptiX_API_Reference_8.0.0.pdf" "OptiX documentation"; then
        export OPTIX80_PATH="$OPTIX_ARG"
        echo "OPTIX80_PATH set to $OPTIX80_PATH"
    fi
else
    echo "OPTIX80_PATH not set (no --optix-dir provided)"
fi

# -----------------------------
# Set FBX_SDK_PATH
# -----------------------------
if [ -n "$FBX_ARG" ]; then
    if check_dir_exists "$FBX_ARG" "Autodesk FBX SDK" && \
       check_file_exists "$FBX_ARG/FBX_SDK_Online_Documentation.html" "FBX SDK documentation"; then
        export FBX_SDK_PATH="$FBX_ARG"
        echo "FBX_SDK_PATH set to $FBX_SDK_PATH"
    fi
else
    echo "FBX_SDK_PATH not set (no --autodesk-fbx-sdk-dir provided)"
fi

echo "CUDA, OptiX, and FBX SDK environment setup complete."

# -----------------------------
# Update shell prompt (PS1)
# -----------------------------
if [ -z "$_DMT_ENV_PS1_BACKUP" ]; then
    # Backup original prompt once
    export _DMT_ENV_PS1_BACKUP="$PS1"
fi

ENV_PREFIX="[dmt-env] "
case "$PS1" in
    *"$ENV_PREFIX"*) ;; # already has prefix
    *) export PS1="${ENV_PREFIX}${PS1}" ;;
esac

