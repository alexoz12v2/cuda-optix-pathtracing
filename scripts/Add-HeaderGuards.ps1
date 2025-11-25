param(
    [Parameter(Mandatory)]
    [string]$RootDir
)

if (-not (Test-Path $RootDir))
{
    Write-Error "Directory does not exist: $RootDir"
    exit 1
}

# Normalize
$RootDir = (Resolve-Path $RootDir).Path

# File list
$files = Get-ChildItem -Path $RootDir -Recurse -File -Include *.h, *.hpp, *.cuh

foreach ($file in $files)
{

    # Compute relative path
    $rel = [System.IO.Path]::GetRelativePath($RootDir, $file.FullName)

    # Convert to include guard key:
    # - replace directory separators with _
    # - replace - with _
    # - remove invalid chars
    $guardCore = $rel `
         -replace '\\', '_' `
         -replace '/', '_' `
         -replace '-', '_' `
         -replace '\W', '_' `
        | ForEach-Object { $_.ToUpper() }

    $guard = "DMT_$guardCore"

    # Read file
    $content = Get-Content -Raw -Path $file.FullName

    # Skip if already has this guard
    if ($content -match "#ifndef\s+$guard")
    {
        Write-Host "Already processed: $rel"
        continue
    }

    # Replace #pragma once
    if ($content -match '#pragma\s+once')
    {

        Write-Host "Processing: $rel  →  $guard"

        $newTop = @"
#ifndef $guard
#define $guard
"@

        $newEnd = @"

#endif // $guard
"@

        # Remove #pragma once (only the first occurrence)
        $content = $content -replace '#pragma\s+once', $newTop

        # Add end-of-file #endif
        $content = $content.TrimEnd() + $newEnd

        # Write back
        Set-Content -Path $file.FullName -Value $content -Encoding UTF8

    }
    else
    {
        Write-Host "Skipping: $rel (no #pragma once)"
    }
}
