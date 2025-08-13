# From the root of the repository
# import: Import-Module scripts/dev-utils.psm1
# remove: Remove-Module dev-utils
# once imported, you can use any function here

function Add-SourceToModule {
    [CmdletBinding(SupportsShouldProcess = $true)]
    param (
        [Parameter(Mandatory = $true, ValueFromPipelineByPropertyName = $true)]
        [string] $ModuleName,

        [Parameter(Mandatory = $true, ValueFromPipeline = $true)]
        [Alias("Name")]
        [AllowEmptyCollection()]
        [AllowNull()]
        [object] $FileName,

        [Parameter(Mandatory = $false)]
        [AllowNull()]
        [string] $config = "Debug-VS"
    )

    begin {
        $root = Get-Location
    }

    process {
        # Normalize FileName input: support single string, array, or pipeline
        if ($FileName -is [string]) {
            $fileList = @($FileName)
        }
        elseif ($FileName -is [System.Collections.IEnumerable]) {
            $fileList = @($FileName | ForEach-Object { $_.ToString() })
        }
        else {
            Write-Error "FileName must be a string or list of strings"
            return
        }

        $srcCMakeFile = Join-Path $root "src\$ModuleName\CMakeLists.txt"
        $includeDir = Join-Path $root "include\$ModuleName"

        if (-not (Test-Path $srcCMakeFile)) {
            Write-Error "CMakeLists.txt not found: $srcCMakeFile"
            return
        }

        if (-not (Test-Path $includeDir)) {
            Write-Error "Include directory not found: $includeDir"
            return
        }

        $cmakeContent = [System.Collections.ArrayList]::new()
        Get-Content $srcCMakeFile | ForEach-Object { $null = $cmakeContent.Add($_) }

        $headerInsertIndex = -1
        $implInsertIndex = -1

        for ($i = 0; $i -lt $cmakeContent.Count; $i++) {
            $line = $cmakeContent[$i].Trim()
            if ($line -match '^\s*\$\{CMAKE_SOURCE_DIR\}/include/' -and $line -match '\.h') {
                $headerInsertIndex = $i + 1
            }
            elseif ($line -match '^\s*[^#]*\.cpp') {
                $implInsertIndex = $i + 1
            }
        }

        if ($headerInsertIndex -eq -1 -or $implInsertIndex -eq -1) {
            Write-Error "Could not find valid header or implementation insertion point."
            return
        }

        foreach ($file in $fileList) {
            $baseName = [System.IO.Path]::GetFileNameWithoutExtension($file)
            $headerPath = Join-Path $includeDir "$ModuleName-$baseName.h"
            $implPath = Join-Path "$root/src/$ModuleName" "$ModuleName-$baseName.cpp"

            if (-not (Test-Path $headerPath)) {
                New-Item -Path $headerPath -ItemType File -Force | Out-Null
                Add-Content -Path $headerPath -Value "// $ModuleName-$baseName.h"
            }

            if (-not (Test-Path $implPath)) {
                New-Item -Path $implPath -ItemType File -Force | Out-Null
                Add-Content -Path $implPath -Value "// $ModuleName-$baseName.cpp"
            }

            $headerRelPath = '${CMAKE_SOURCE_DIR}/include/' + $ModuleName + '/' + "$ModuleName-$baseName.h"
            $implRelPath = "$ModuleName-$baseName.cpp"

            $indentHeader = ($cmakeContent[$headerInsertIndex - 1] -replace '\S.*$', '')
            $indentImpl = ($cmakeContent[$implInsertIndex - 1] -replace '\S.*$', '')

            Write-Host "Inserting `"$indentHeader$headerRelPath`""
            $cmakeContent.Insert($headerInsertIndex, "$indentHeader$headerRelPath")
            Write-Host "Inserting `"$indentImpl$implRelPath`""
            $cmakeContent.Insert($implInsertIndex + 1, "$indentImpl$implRelPath")

            $headerInsertIndex++
            $implInsertIndex++
        }

        Set-Content -Path $srcCMakeFile -Value $cmakeContent
        Write-Host "Updated $srcCMakeFile with new source files."
    }

    end {
        if ($config -in @("Debug-VS", "Debug-WinNinja")) {
            Write-Host "Running CMake with config: $config"
            Start-ProcessWithOutput -FilePath "cmake.exe" -ArgumentsList "..", "--preset", $config
        }
    }
}

Function Start-ProcessWithOutput {
    param (
        [Parameter(Mandatory)]
        [string]$FilePath,

        [Parameter()]
        [string[]]$ArgumentsList
    )

    begin {
        $tmp = [System.IO.Path]::GetTempFileName()

        # Start a background job that tails the output file
        $readJob = Start-ThreadJob `
            -ScriptBlock {
            param($Path)
            Get-Content -Path $Path -Wait
        } `
            -ArgumentList $tmp

        # Start the process, redirecting stdout to the temp file
        $process = Start-Process `
            -FilePath $FilePath `
            -ArgumentList $ArgumentsList `
            -NoNewWindow `
            -RedirectStandardOutput $tmp `
            -PassThru
    }

    process {
        do {
            # Get new output lines from the read job
            $lines = $readJob | Receive-Job
            foreach ($line in $lines) {
                if ($line) {
                    if ($line -match 'error|CMake Error') {
                        Write-Host "[cmake] $line" -ForegroundColor Red
                    }
                    else {
                        Write-Host "[cmake] $line"
                    }
                }
            }
            Start-Sleep -Milliseconds 200
        } while (-not $process.HasExited -or -not ($readJob | Receive-Job -OutVariable temp) -eq $null)
    }

    end {
        # Drain remaining output
        $lines = $readJob | Receive-Job -Keep
        foreach ($line in $lines) {
            if ($line) {
                if ($line -match 'error|CMake Error') {
                    Write-Host "[cmake] $line" -ForegroundColor Red
                }
                else {
                    Write-Host "[cmake] $line"
                }
            }
        }
    }

    clean {
        $readJob | Remove-Job -Force
        Remove-Item -Path $tmp -Force
        return $process.ExitCode
    }
}
