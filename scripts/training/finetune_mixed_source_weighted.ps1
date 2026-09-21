$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$python = "E:\Program Files\Python\python.exe"
$outputDir = Join-Path $repoRoot "work_dirs\r2plus1d18_mixed_source_weighted_e23"

$arguments = @(
    "-u",
    (Join-Path $repoRoot "scripts\training\train_shuttleset_rgb_multitask.py"),
    "--resume", (Join-Path $repoRoot "work_dirs\r2plus1d18_mixed_shuttleset_finebadminton\best.pth"),
    "--epochs", "23",
    "--output-dir", $outputDir,
    "--manifest", (Join-Path $repoRoot "data\manifests\shuttleset_rgb.csv"),
    "--crop-hitter",
    "--crop-padding", "0.55",
    "--cache-dir", "C:\Users\ADMIN\Downloads\AI_Classifier_cache\shuttleset_v3_full",
    "--fine-dataset-root", "C:\Users\ADMIN\Downloads\Fine-Badminton",
    "--fine-train-manifest", (Join-Path $repoRoot "data\manifests\fine_badminton_splits\train.csv"),
    "--fine-val-manifest", (Join-Path $repoRoot "data\manifests\fine_badminton_splits\val.csv"),
    "--fine-cache-dir", "C:\Users\ADMIN\Downloads\AI_Classifier_cache\fine_v3_full",
    "--batch-size", "4",
    "--workers", "0",
    "--unfreeze-layer4",
    "--learning-rate", "1e-5",
    "--backbone-learning-rate", "1e-6",
    "--fine-stroke-loss-weight", "1.0",
    "--shuttle-stroke-loss-weight", "0.35",
    "--side-loss-weight", "1.0",
    "--reset-optimizer",
    "--amp"
)

New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
& $python @arguments
exit $LASTEXITCODE
