$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$python = "E:\Program Files\Python\python.exe"
$outputDir = Join-Path $repoRoot "work_dirs\r2plus1d18_rgb_v2_24x160_pilot"
$arguments = @(
    "-u", (Join-Path $repoRoot "scripts\training\train_shuttleset_rgb_multitask.py"),
    "--resume", (Join-Path $repoRoot "work_dirs\r2plus1d18_mixed_shuttleset_finebadminton\best.pth"),
    "--epochs", "22", "--output-dir", $outputDir,
    "--manifest", (Join-Path $repoRoot "data\manifests\shuttleset_rgb.csv"),
    "--frames", "24", "--crop-size", "160", "--crop-hitter", "--crop-padding", "0.55",
    "--cache-dir", "C:\Users\ADMIN\Downloads\AI_Classifier_cache\shuttleset_rgb_v2",
    "--fine-dataset-root", "C:\Users\ADMIN\Downloads\Fine-Badminton",
    "--fine-train-manifest", (Join-Path $repoRoot "data\manifests\fine_badminton_splits\train.csv"),
    "--fine-val-manifest", (Join-Path $repoRoot "data\manifests\fine_badminton_splits\val.csv"),
    "--fine-cache-dir", "C:\Users\ADMIN\Downloads\AI_Classifier_cache\fine_rgb_v2",
    "--samples-per-side", "200", "--val-samples-per-side", "50",
    "--fine-samples-per-class", "200", "--fine-val-samples-per-class", "50",
    "--batch-size", "2", "--workers", "0", "--unfreeze-layer4",
    "--learning-rate", "5e-6", "--backbone-learning-rate", "5e-7",
    "--side-loss-weight", "1.0", "--reset-optimizer", "--amp"
)
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
& $python @arguments
exit $LASTEXITCODE
