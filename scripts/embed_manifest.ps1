# test command
# New-ManifestFromTemplate -ManifestTemplateFilePath .\res\win32-application.manifest -ManifestTemplateParams @{ version="1.0.0.0"; name="dmt-stuff"; description="test" }
Function New-ManifestFromTemplate
{
    [CmdLetBinding()]
    param (
        [Parameter(Mandatory)]
        [string]$ManifestTemplateFilePath,

        [Parameter(Mandatory)]
        [hashtable]$ManifestTemplateParams
    )

    try
    {
        # Read the manifest template file
        $ManifestContent = Get-Content -Path $ManifestTemplateFilePath -Raw

        # Replace placeholders with actual values
        foreach ($Key in $ManifestTemplateParams.Keys)
        {
            $Placeholder = "{{" + $Key + "}}"
            $ManifestContent = $ManifestContent -replace [regex]::Escape($Placeholder), $ManifestTemplateParams[$Key]
        }

        return $ManifestContent
    }
    catch
    {
        Write-Error "Error processing manifest template: $_"
        return $null
    }
}

# Test Command
# Invoke-Manifest-And-Embed -ExecutableFilePath .\build\Debug-VS\bin\dmt-filejobtest-d.exe -ManifestTemplateFilePath .\res\win32-application.manifest -ManifestTemplateParams @{version="1.0.0.0"; name="dmt-filejobtest-d"; description="File Job Test (DEBUG)"} -PrintManifest
Function Invoke-Manifest-And-Embed
{
    [CmdLetBinding()]
    param (
        [Parameter(Mandatory)]
        [string]$ExecutableFilePath,

        [Parameter(Mandatory)]
        [string]$ManifestTemplateFilePath,

        [Parameter(Mandatory)]
        [hashtable]$ManifestTemplateParams,

        [switch]$PrintManifest
    )

    # Generate interpolated manifest content
    $ManifestContent = New-ManifestFromTemplate -ManifestTemplateFilePath $ManifestTemplateFilePath -ManifestTemplateParams $ManifestTemplateParams

    if (-not $ManifestContent)
    {
        Write-Error "Failed to generate manifest content."
        return 1
    }

    # Define output manifest file path
    $ManifestFilePath = "$ExecutableFilePath.manifest"

    # Write the interpolated manifest to a file
    $ManifestContent | Set-Content -Path $ManifestFilePath -Encoding UTF8

    Write-Host "Manifest file created: $ManifestFilePath"


    # Locate mt.exe using vswhere
    # Attempt to get Windows 10 SDK installation folder from registry
    $MtExePath = $null
    try
    {
        $SdkRoot = (Get-ItemProperty "HKLM:\SOFTWARE\Microsoft\Windows Kits\Installed Roots" -Name KitsRoot10 -ErrorAction Stop).KitsRoot10

        if ($SdkRoot -and (Test-Path $SdkRoot))
        {
            # Get all installed SDK versions
            $SdkVersions = Get-ChildItem "HKLM:\SOFTWARE\Microsoft\Windows Kits\Installed Roots" |
                    Where-Object { $_.PSChildName -match '^\d+\.\d+\.\d+\.\d+$' } |
                    Sort-Object PSChildName -Descending

            if ($SdkVersions)
            {
                $LatestVersion = $SdkVersions[0].PSChildName
                $Candidate = Join-Path -Path $SdkRoot -ChildPath "bin\$LatestVersion\x64\mt.exe"
                if (Test-Path $Candidate)
                {
                    $MtExePath = $Candidate
                }
            }
        }
    }
    catch
    {
        # Ignore errors and fallback
    }

    # Fallback: search Program Files (x86)
    if (-not $MtExePath)
    {
        $MtExePath = Get-ChildItem "${env:ProgramFiles(x86)}\Windows Kits\10\bin" -Recurse -Filter mt.exe |
                Where-Object { $_.FullName -match "\\x64\\" } |
                Sort-Object LastWriteTime -Descending |
                Select-Object -First 1

        if ($MtExePath)
        {
            $MtExePath = $MtExePath.FullName
        }
    }
    if (-not (Test-Path $MtExePath))
    {
        Write-Error "mt.exe not found at $MtExePath"
        return 1
    }

    # Validate the manifest
    Write-Host "Validating manifest..."
    & $MtExePath -manifest $ManifestFilePath -validate_manifest -nologo
    if ($LASTEXITCODE -ne 0)
    {
        Write-Error "Manifest validation failed!"
        Remove-Item -Path $ManifestFilePath -ErrorAction SilentlyContinue
        return 1
    }

    # Embed the manifest into the executable
    Write-Host "Embedding manifest into executable..."
    & $MtExe -manifest $ManifestFilePath -outputresource:"$ExecutableFilePath`;`#1" -nologo
    if ($LASTEXITCODE -ne 0)
    {
        Write-Error "Failed to embed manifest into the executable!"
        Remove-Item -Path $ManifestFilePath -ErrorAction SilentlyContinue
        return 1
    }

    # Cleanup: Remove the temporary manifest file
    Remove-Item -Path $ManifestFilePath -ErrorAction SilentlyContinue

    Write-Host "Manifest successfully embedded into $ExecutableFilePath"
    if ($PrintManifest)
    {
        Write-Host $ManifestContent
    }
    return 0
}