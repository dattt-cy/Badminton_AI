$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$python = "E:\Program Files\Python\python.exe"
$outputDir = Join-Path $repoRoot "work_dirs\shuttleset_fusion_epoch20_full_b4"
$arguments = @(
    "-u", (Join-Path $repoRoot "scripts\training\train_shuttleset_feature_fusion.py"),
    "--manifest", (Join-Path $repoRoot "data\manifests\shuttleset_npy.csv"),
    "--npy-root", "C:\Users\ADMIN\Downloads\dataset_npy_between_2_hits_with_max_limits\dataset_npy_between_2_hits_with_max_limits",
    "--rgb-cache", "C:\Users\ADMIN\Downloads\AI_Classifier_cache\shuttleset_v3_full\frames_16",
    "--rgb-checkpoint", (Join-Path $repoRoot "work_dirs\r2plus1d18_mixed_shuttleset_finebadminton\best.pth"),
    "--feature-cache", (Join-Path $outputDir "features.npz"),
    "--output-dir", $outputDir,
    "--full-data", "--epochs", "20", "--batch-size", "4", "--learning-rate", "1e-3"
)
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
& $python @arguments
exit $LASTEXITCODE
