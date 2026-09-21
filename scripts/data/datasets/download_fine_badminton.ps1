param(
    [string]$Destination = "$env:USERPROFILE\Downloads\Fine-Badminton.zip",
    [int]$Connections = 8
)

$ErrorActionPreference = "Stop"

if ($Connections -lt 1 -or $Connections -gt 16) {
    throw "Connections must be between 1 and 16."
}

$expectedBytes = 6455837331L
$expectedMd5 = "cb3fcaca0ae9b08c564ac2a19ae4e7a0"
$url = "https://zenodo.org/api/records/20292976/files/Fine-Badminton.zip/content"

$destinationPath = [IO.Path]::GetFullPath($Destination)
$destinationDirectory = [IO.Path]::GetDirectoryName($destinationPath)
$destinationName = [IO.Path]::GetFileName($destinationPath)
[IO.Directory]::CreateDirectory($destinationDirectory) | Out-Null

$driveName = [IO.Path]::GetPathRoot($destinationPath).TrimEnd("\").TrimEnd(":")
$drive = Get-PSDrive -Name $driveName
$existingBytes = if ([IO.File]::Exists($destinationPath)) {
    (Get-Item -LiteralPath $destinationPath).Length
} else {
    0L
}
$requiredBytes = [Math]::Max(0L, $expectedBytes - $existingBytes)
if ($drive.Free -lt $requiredBytes) {
    throw "Not enough free space on drive $driveName. Need $requiredBytes bytes."
}

$ariaRoot = Join-Path $env:TEMP "aria2-1.37.0"
$ariaExe = Get-ChildItem -LiteralPath $ariaRoot -Recurse -File -Filter aria2c.exe `
    -ErrorAction SilentlyContinue | Select-Object -First 1
if ($null -eq $ariaExe) {
    throw "aria2c was not found under $ariaRoot. Download the official Windows build first."
}

Write-Host "Downloading to $destinationPath"
Write-Host "The .aria2 control file allows interrupted downloads to resume."

& $ariaExe.FullName `
    --continue=true `
    --max-connection-per-server=$Connections `
    --split=$Connections `
    --min-split-size=10M `
    --file-allocation=none `
    --max-tries=0 `
    --retry-wait=5 `
    --timeout=60 `
    --dir=$destinationDirectory `
    --out=$destinationName `
    $url

if ($LASTEXITCODE -ne 0) {
    throw "aria2c exited with code $LASTEXITCODE"
}

$file = Get-Item -LiteralPath $destinationPath
if ($file.Length -ne $expectedBytes) {
    throw "Unexpected file size: $($file.Length), expected $expectedBytes"
}

Write-Host "Verifying MD5 (this can take a while)..."
$actualMd5 = (Get-FileHash -LiteralPath $destinationPath -Algorithm MD5).Hash.ToLowerInvariant()
if ($actualMd5 -ne $expectedMd5) {
    throw "MD5 mismatch: $actualMd5, expected $expectedMd5"
}

Write-Host "Download verified successfully: $destinationPath"
