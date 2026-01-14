<#
.SYNOPSIS
    Produce a clean environment for CLion/CMake + CUDA 12.6
.DESCRIPTION
    - Enumerates installed MSVC toolsets and picks the newest
    - Enumerates Windows SDKs via registry and picks the newest
    - Sets PATH, INCLUDE, LIB, LIBPATH, CUDA_* variables
    - Deduplicates all PATH-like entries
    - Prints variables to stdout in KEY=VALUE form
#>

Write-Host "[resetenv] Preparing clean environment..." -ForegroundColor Cyan

# ----------------------------
# Step 1: Minimal system environment
# ----------------------------
$baseVars = @(
    "ALLUSERSPROFILE", "APPDATA", "CommonProgramFiles", "CommonProgramFiles(x86)", "CommonProgramW6432",
    "COMPUTERNAME", "ComSpec", "HOMEDRIVE", "HOMEPATH", "LOCALAPPDATA", "LOGONSERVER", "NUMBER_OF_PROCESSORS",
    "OS", "PATHEXT", "PROCESSOR_ARCHITECTURE", "PROCESSOR_IDENTIFIER", "PROCESSOR_LEVEL", "PROCESSOR_REVISION",
    "ProgramData", "ProgramFiles", "ProgramFiles(x86)", "ProgramW6432", "PUBLIC", "SystemDrive", "SystemRoot",
    "TEMP", "TMP", "USERNAME", "USERPROFILE", "windir"
)

# Capture existing base vars
$envDict = @{ }
foreach ($v in $baseVars)
{
    if (Test-Path "Env:$v")
    {
        $envDict[$v] = (Get-Item "Env:$v").Value
    }
}

# ----------------------------
# Step 2: Detect newest MSVC
# ----------------------------
$vsRoot = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC"
$msvcVersions = Get-ChildItem $vsRoot -Directory | Sort-Object Name -Descending
if ($msvcVersions.Count -eq 0)
{
    throw "No MSVC toolset found in $vsRoot"
}
$msvcVer = $msvcVersions[0].Name
$msvcRoot = Join-Path $vsRoot $msvcVer

# ----------------------------
# Step 3: Detect newest Windows SDK
# ----------------------------
$sdks = Get-ChildItem "HKLM:\SOFTWARE\Microsoft\Windows Kits\Installed Roots" |
        Where-Object { $_.PSChildName -match "^10\." } |
        Sort-Object PSChildName -Descending |
        Select-Object -ExpandProperty Name |
        Split-Path -Leaf
$windowsSdkVersions = $sdks | Where-Object { $_ -match "^10\." } | Sort-Object -Descending
if ($windowsSdkVersions.Count -eq 0)
{
    throw "No Windows 10 SDK found in registry"
}
$kitsRoot10 = Get-ItemProperty "HKLM:\SOFTWARE\Microsoft\Windows Kits\Installed Roots" -Name KitsRoot10 | Select-Object -ExpandProperty KitsRoot10
$windowsSdkVer = $windowsSdkVersions[0]
$windowsSdkRoot = $kitsRoot10.TrimEnd('\')

# ----------------------------
# Step 4: CUDA 12.6
# ----------------------------
$cudaPath = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
if (-not (Test-Path $cudaPath))
{
    throw "CUDA v12.6 not found at $cudaPath"
}

# ----------------------------
# Step 5: Build PATH, INCLUDE, LIB manually
# ----------------------------
$paths = @(
    Join-Path $env:ProgramFiles "LLVM\bin";
    Join-Path $cudaPath "bin";
    Join-Path $msvcRoot "bin\Hostx64\x64";
    Join-Path $msvcRoot "lib\x64";
    Join-Path $windowsSdkRoot "bin\$windowsSdkVer\x64";
    Join-Path $windowsSdkRoot "bin\$windowsSdkVer\x86";
    Join-Path $windowsSdkRoot "bin";
    Join-Path $windowsSdkRoot "Lib\$windowsSdkVer\um\x64";
    Join-Path $windowsSdkRoot "Lib\$windowsSdkVer\um\x86";
    "C:\Windows\System32\WindowsPowerShell\v1.0\",
    "C:\Windows\System32\",
    "C:\Windows\System\",
    "C:\Windows\";
    Join-Path $Env:Userprofile "AppData\Local\Microsoft\WindowsApps";
    Join-Path $Env:Userprofile "AppData\Local\Microsoft\WinGet\Links";
    Join-Path $Env:Userprofile ".dotnet\tools";
    Join-Path $env:ProgramFiles "CMake\bin";
    Join-Path $env:ProgramFiles "dotnet";
    Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer";
    Join-Path $env:ProgramFiles "Git\cmd"
)

# Deduplicate paths while preserving order
$seen = @{ }
$paths = $paths | Where-Object { $_ -and (-not $seen.ContainsKey($_)); $seen[$_] = $true }

# TODO: Get-ChildItem of include directory
$includes = @(
    Join-Path $msvcRoot "include";
    Join-Path $windowsSdkRoot "Include\$windowsSdkVer\shared";
    Join-Path $windowsSdkRoot "Include\$windowsSdkVer\um";
    Join-Path $windowsSdkRoot "Include\$windowsSdkVer\winrt";
    Join-Path $windowsSdkRoot "Include\$windowsSdkVer\ucrt"
)

$libs = @(
    Join-Path $msvcRoot "lib\x64";
    Join-Path $windowsSdkRoot "Lib\$windowsSdkVer\um\x64";
    Join-Path $windowsSdkRoot "Lib\$windowsSdkVer\ucrt\x64"
)

# Deduplicate INCLUDE and LIB
$seenInc = @{ }; $includes = $includes | Where-Object { $_ -and (-not $seenInc.ContainsKey($_)); $seenInc[$_] = $true }
$seenLib = @{ }; $libs = $libs | Where-Object { $_ -and (-not $seenLib.ContainsKey($_)); $seenLib[$_] = $true }

# ----------------------------
# Step 6: Print environment in lexicographical order
# ----------------------------
$allVars = @{ }
# Base system vars
foreach ($k in $envDict.Keys)
{
    $allVars[$k] = $envDict[$k]
}

# Toolchain vars
$allVars["PATH"] = ($paths -join ';')
$allVars["INCLUDE"] = ($includes -join ';')
$allVars["LIB"] = ($libs -join ';')
$allVars["LIBPATH"] = ($libs -join ';')
$allVars["CUDA_HOME"] = $cudaPath
$allVars["CUDA_PATH"] = $cudaPath
$allVars["CUDA_PATH_V12_6"] = $cudaPath
$allVars["VCToolsInstallDir"] = $msvcRoot
$allVars["VCToolsRedistDir"] = $msvcRoot.Replace("Tools", "Redist") # WARNING: Replace only last and test path
$allVars["VCToolsVersion"] = $msvcVer

# Windows SDK vars
$allVars["WindowsLibPath"] = (Join-Path $windowsSdkRoot "UnionMetadata\$windowsSdkVer") + ";" + (Join-Path $windowsSdkRoot "References\$windowsSdkVer")
$allVars["WindowsSDK_ExecutablePath_x64"] = Join-Path $windowsSdkRoot "bin\NETFX 4.8 Tools\x64"
$allVars["WindowsSDK_ExecutablePath_x86"] = Join-Path $windowsSdkRoot "bin\NETFX 4.8 Tools"
$allVars["WindowsSdkBinPath"] = Join-Path $windowsSdkRoot "bin"
$allVars["WindowsSdkDir"] = $windowsSdkRoot
$allVars["WindowsSDKLibVersion"] = "$windowsSdkVer"
$allVars["WindowsSdkVerBinPath"] = Join-Path $windowsSdkRoot "bin\$windowsSdkVer"
$allVars["WindowsSDKVersion"] = "$windowsSdkVer"

# Print all vars in lexicographical order
$allVars.Keys | Sort-Object | ForEach-Object {
    Write-Output "$_=$( $allVars[$_] )"
}

Write-Host "[resetenv] Environment ready for CLion/CMake + CUDA 12.6." -ForegroundColor Green

exit 0
