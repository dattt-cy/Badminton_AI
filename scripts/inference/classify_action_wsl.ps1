param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$InputPath,

    [string]$Checkpoint = "models/checkpoints/action_recognition/best_top1_acc_2class_source_safe.pth",

    [ValidateSet("single", "far", "near", "any")]
    [string]$Target = "single"
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
$resolvedInput = (Resolve-Path -LiteralPath $InputPath).Path
$resolvedCheckpoint = if ([IO.Path]::IsPathRooted($Checkpoint)) {
    (Resolve-Path -LiteralPath $Checkpoint).Path
} else {
    (Resolve-Path -LiteralPath (Join-Path $repoRoot $Checkpoint)).Path
}

$repoWsl = (& wsl.exe -d Ubuntu -- wslpath -a $repoRoot).Trim()
$inputWsl = (& wsl.exe -d Ubuntu -- wslpath -a $resolvedInput).Trim()
$checkpointWsl = (& wsl.exe -d Ubuntu -- wslpath -a $resolvedCheckpoint).Trim()
$python = "/home/victomblack1602/miniforge3/envs/badminton-pyskl/bin/python"
$safeStem = [IO.Path]::GetFileNameWithoutExtension($resolvedInput) -replace '[^A-Za-z0-9_-]', '_'
$poseOutput = "outputs/${safeStem}_pose.npz"
$jsonOutput = "outputs/${safeStem}_prediction.json"

& wsl.exe -d Ubuntu --cd $repoWsl -- env `
    LD_LIBRARY_PATH=/usr/lib/wsl/lib `
    $python `
    scripts/inference/classify_action.py `
    $inputWsl `
    --checkpoint $checkpointWsl `
    --action-config configs/action_recognition/stgcnpp_badminton.py `
    --target $Target `
    --device cuda:0 `
    --pose-output $poseOutput `
    --json-output $jsonOutput

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "Pose: $repoRoot\$($poseOutput -replace '/', '\')"
Write-Host "Prediction: $repoRoot\$($jsonOutput -replace '/', '\')"
