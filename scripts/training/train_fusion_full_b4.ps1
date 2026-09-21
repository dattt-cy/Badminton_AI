$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$python = "E:\Program Files\Python\python.exe"
$outputDir = Join-Path $repoRoot "work_dirs\shuttleset_fusion_full_b4"
$arguments = @(
    "-u", (Join-Path $repoRoot "scripts\training\train_shuttleset_feature_fusion.py"),
    "--feature-cache", (Join-Path $repoRoot "work_dirs\shuttleset_fusion_full\features.npz"),
    "--output-dir", $outputDir,
    "--epochs", "20", "--batch-size", "4", "--learning-rate", "1e-3"
)
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
& $python @arguments
exit $LASTEXITCODE
