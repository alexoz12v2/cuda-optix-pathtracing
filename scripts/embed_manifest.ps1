# test command
# New-ManifestFromTemplate -ManifestTemplateFilePath .\res\win32-application.manifest -ManifestTemplateParams @{ version="1.0.0.0"; name="dmt-stuff"; description="test" }
Function New-ManifestFromTemplate {
    [CmdLetBinding()]
    param (
        [Parameter(Mandatory)]
        [string]$ManifestTemplateFilePath,
        
        [Parameter(Mandatory)]
        [hashtable]$ManifestTemplateParams
    )

    try {
        # Read the manifest template file
        $ManifestContent = Get-Content -Path $ManifestTemplateFilePath -Raw

        # Replace placeholders with actual values
        foreach ($Key in $ManifestTemplateParams.Keys) {
            $Placeholder = "{{" + $Key + "}}"
            $ManifestContent = $ManifestContent -replace [regex]::Escape($Placeholder), $ManifestTemplateParams[$Key]
        }

        return $ManifestContent
    }
    catch {
        Write-Error "Error processing manifest template: $_"
        return $null
    }
}

# Test Command
# Invoke-Manifest-And-Embed -ExecutableFilePath .\build\Debug-VS\bin\dmt-filejobtest-d.exe -ManifestTemplateFilePath .\res\win32-application.manifest -ManifestTemplateParams @{version="1.0.0.0"; name="dmt-filejobtest-d"; description="File Job Test (DEBUG)"} -PrintManifest
Function Invoke-Manifest-And-Embed {
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

    if (-not $ManifestContent) {
        Write-Error "Failed to generate manifest content."
        return 1
    }

    # Define output manifest file path
    $ManifestFilePath = "$ExecutableFilePath.manifest"

    # Write the interpolated manifest to a file
    $ManifestContent | Set-Content -Path $ManifestFilePath -Encoding UTF8

    Write-Host "Manifest file created: $ManifestFilePath"

    # Path to mt.exe (Microsoft Manifest Tool)
    if ($null -eq (Get-Command -Name "mt.exe" -ErrorAction SilentlyContinue)) {
        Write-Error "mt.exe is not on the path. Did you open the Developer Powershell?"
        return 1
    }
    $MtExe = "mt.exe"

    # Validate the manifest
    Write-Host "Validating manifest..."
    & $MtExe -manifest $ManifestFilePath -validate_manifest -nologo
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Manifest validation failed!"
        Remove-Item -Path $ManifestFilePath -ErrorAction SilentlyContinue
        return 1
    }

    # Embed the manifest into the executable
    Write-Host "Embedding manifest into executable..."
    & $MtExe -manifest $ManifestFilePath -outputresource:"$ExecutableFilePath`;`#1" -nologo
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to embed manifest into the executable!"
        Remove-Item -Path $ManifestFilePath -ErrorAction SilentlyContinue
        return 1
    }

    # Cleanup: Remove the temporary manifest file
    Remove-Item -Path $ManifestFilePath -ErrorAction SilentlyContinue

    Write-Host "Manifest successfully embedded into $ExecutableFilePath"
    if ($PrintManifest) {
        Write-Host $ManifestContent
    }
    return 0
}