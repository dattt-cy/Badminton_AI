param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$InputPath,

    [string]$Checkpoint = "models/checkpoints/action_recognition/best_top1_acc_3class.pth",

    [string]$ActionConfig = "configs/action_recognition/stgcnpp_multisense_3class.py",

    [ValidateSet("single", "far", "near", "any")]
    [string]$Target = "single",

    [ValidateRange(0.0, 1.0)]
    [double]$MinActionConfidence = 0.75,

    [ValidateRange(0.1, 30.0)]
    [double]$WindowSeconds = 3.5
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
$resolvedInput = (Resolve-Path -LiteralPath $InputPath).Path
$resolvedCheckpoint = if ([IO.Path]::IsPathRooted($Checkpoint)) {
    (Resolve-Path -LiteralPath $Checkpoint).Path
} else {
    (Resolve-Path -LiteralPath (Join-Path $repoRoot $Checkpoint)).Path
}
$resolvedActionConfig = if ([IO.Path]::IsPathRooted($ActionConfig)) {
    (Resolve-Path -LiteralPath $ActionConfig).Path
} else {
    (Resolve-Path -LiteralPath (Join-Path $repoRoot $ActionConfig)).Path
}

function ConvertTo-WslPath([string]$WindowsPath) {
    if ($WindowsPath -notmatch '^([A-Za-z]):\\(.*)$') {
        throw "Expected an absolute Windows drive path, got: $WindowsPath"
    }

    $drive = $Matches[1].ToLowerInvariant()
    $tail = $Matches[2] -replace '\\', '/'
    return "/mnt/$drive/$tail"
}

$repoWsl = ConvertTo-WslPath $repoRoot
$inputWsl = ConvertTo-WslPath $resolvedInput
$checkpointWsl = ConvertTo-WslPath $resolvedCheckpoint
$actionConfigWsl = ConvertTo-WslPath $resolvedActionConfig
$python = "/home/victomblack1602/miniforge3/envs/badminton-pyskl/bin/python"
$safeStem = [IO.Path]::GetFileNameWithoutExtension($resolvedInput) -replace '[^A-Za-z0-9_-]', '_'
$poseOutput = "outputs/${safeStem}_pose.npz"
$jsonOutput = "outputs/${safeStem}_prediction.json"

& wsl.exe -d Ubuntu --cd $repoWsl --exec env `
    LD_LIBRARY_PATH=/usr/lib/wsl/lib `
    $python `
    scripts/inference/classify_action.py `
    $inputWsl `
    --checkpoint $checkpointWsl `
    --action-config $actionConfigWsl `
    --target $Target `
    --device cuda:0 `
    --min-action-confidence $MinActionConfidence `
    --window-seconds $WindowSeconds `
    --pose-output $poseOutput `
    --json-output $jsonOutput

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "Pose: $repoRoot\$($poseOutput -replace '/', '\')"
Write-Host "Prediction: $repoRoot\$($jsonOutput -replace '/', '\')"
