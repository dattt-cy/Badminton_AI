$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$python = "E:\Program Files\Python\python.exe"
$outputDir = Join-Path $repoRoot "work_dirs\r2plus1d18_mixed_drive_clean_v2_e22"
$arguments = @(
    "-u", (Join-Path $repoRoot "scripts\training\train_shuttleset_rgb_multitask.py"),
    "--resume", (Join-Path $outputDir "latest.pth"),
    "--epochs", "22", "--output-dir", $outputDir,
    "--manifest", (Join-Path $repoRoot "data\manifests\shuttleset_rgb.csv"),
    "--crop-hitter", "--crop-padding", "0.55",
    "--cache-dir", "C:\Users\ADMIN\Downloads\AI_Classifier_cache\shuttleset_v3_full",
    "--fine-dataset-root", "C:\Users\ADMIN\Downloads\Fine-Badminton",
    "--fine-train-manifest", (Join-Path $repoRoot "data\manifests\fine_badminton_splits\train.csv"),
    "--fine-val-manifest", (Join-Path $repoRoot "data\manifests\fine_badminton_splits\val.csv"),
    "--fine-cache-dir", "C:\Users\ADMIN\Downloads\AI_Classifier_cache\fine_v3_full",
    "--batch-size", "4", "--workers", "0", "--unfreeze-layer4",
    "--learning-rate", "5e-6", "--backbone-learning-rate", "5e-7",
    "--fine-stroke-loss-weight", "1.0", "--shuttle-stroke-loss-weight", "0.6",
    "--shuttle-stroke-exclude-raw-label", "soft_drive",
    "--shuttle-top-sample-boost", "1.5", "--shuttle-drive-sample-boost", "1.5",
    "--shuttle-forehand-sample-boost", "1.25", "--side-loss-weight", "1.0", "--amp"
)
& $python @arguments
exit $LASTEXITCODE
