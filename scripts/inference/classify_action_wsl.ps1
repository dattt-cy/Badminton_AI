param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$InputPath,

    [string]$Checkpoint = "models/checkpoints/action_recognition/stgcnpp_manual_clips_2class_motion_onset_best.pth",

    [string]$ActionConfig = "configs/action_recognition/experiments/stgcnpp_manual_clips_2class_motion_onset.py",

    [ValidateSet("single", "far", "near", "any")]
    [string]$Target = "single",

    [ValidateRange(0.0, 1.0)]
    [double]$MinActionConfidence = 0.75,

    [ValidateRange(0.1, 30.0)]
    [double]$WindowSeconds = 3.5,

    [string]$PoseOutputPath,

    [string]$JsonOutputPath
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
$resolvedJsonOutput = if ($JsonOutputPath) {
    [IO.Path]::GetFullPath($(if ([IO.Path]::IsPathRooted($JsonOutputPath)) {
        $JsonOutputPath
    } else {
        Join-Path $repoRoot $JsonOutputPath
    }))
} else {
    Join-Path $repoRoot "outputs/${safeStem}_prediction.json"
}
$resolvedPoseOutput = if ($PoseOutputPath) {
    [IO.Path]::GetFullPath($(if ([IO.Path]::IsPathRooted($PoseOutputPath)) {
        $PoseOutputPath
    } else {
        Join-Path $repoRoot $PoseOutputPath
    }))
} else {
    Join-Path $repoRoot "outputs/${safeStem}_pose.npz"
}
New-Item -ItemType Directory -Force -Path ([IO.Path]::GetDirectoryName($resolvedJsonOutput)) | Out-Null
$jsonOutputWsl = ConvertTo-WslPath $resolvedJsonOutput
$classifierArgs = @(
    "-d", "Ubuntu", "--cd", $repoWsl, "--exec", "env",
    "LD_LIBRARY_PATH=/usr/lib/wsl/lib", $python,
    "scripts/inference/classify_action.py", $inputWsl,
    "--checkpoint", $checkpointWsl,
    "--action-config", $actionConfigWsl,
    "--target", $Target,
    "--device", "cuda:0",
    "--min-action-confidence", "$MinActionConfidence",
    "--window-seconds", "$WindowSeconds",
    "--json-output", $jsonOutputWsl
)
if ([IO.Path]::GetExtension($resolvedInput).ToLowerInvariant() -ne ".npz") {
    New-Item -ItemType Directory -Force -Path ([IO.Path]::GetDirectoryName($resolvedPoseOutput)) | Out-Null
    $classifierArgs += @("--pose-output", (ConvertTo-WslPath $resolvedPoseOutput))
}

& wsl.exe @classifierArgs

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

if ([IO.Path]::GetExtension($resolvedInput).ToLowerInvariant() -ne ".npz") {
    Write-Host "Pose: $resolvedPoseOutput"
}
Write-Host "Prediction: $resolvedJsonOutput"
