$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$python = "E:\Program Files\Python\python.exe"
$outputDir = Join-Path $repoRoot "work_dirs\r2plus1d18_layer4_3datasets_b24"

Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "FINE-TUNING R(2+1)D-18 (LAYER 4 + HEADS) VOI 3 DATASET (BATCH 24)" -ForegroundColor Green
Write-Host "Che do: Partial Fine-Tuning (Dong bang Layers 1-3, chi train Layer 4)" -ForegroundColor Yellow
Write-Host "Khoi tao: Pretrained Kinetics-400 video backbone" -ForegroundColor Yellow
Write-Host "Toc do du kien: ~15 - 20 phut / Epoch" -ForegroundColor Green
Write-Host "Du lieu: ShuttleSet + FineBadminton + BFMD (46.019 mau cache)" -ForegroundColor Yellow
Write-Host "Bo sung 1.904 mau Drive va tang trong so phat Drive x2.5 (weight=6.808)" -ForegroundColor Yellow
Write-Host "=================================================================" -ForegroundColor Cyan

$arguments = @(
    "-u", (Join-Path $repoRoot "scripts\training\train_shuttleset_rgb_multitask.py"),
    "--manifest", (Join-Path $repoRoot "data\manifests\shuttleset_rgb.csv"),
    "--cache-dir", "C:\Users\ADMIN\Downloads\AI_Classifier_cache\shuttleset_v3_full",
    "--fine-train-manifest", (Join-Path $repoRoot "data\manifests\fine_badminton_splits\train.csv"),
    "--fine-val-manifest", (Join-Path $repoRoot "data\manifests\fine_badminton_splits\val.csv"),
    "--fine-cache-dir", "C:\Users\ADMIN\Downloads\AI_Classifier_cache\fine_v3_full",
    "--bfmd-train-manifest", (Join-Path $repoRoot "data\manifests\bfmd_splits\train.csv"),
    "--bfmd-val-manifest", (Join-Path $repoRoot "data\manifests\bfmd_splits\val.csv"),
    "--bfmd-cache-dir", "C:\Users\ADMIN\Downloads\AI_Classifier_cache\bfmd_v1",
    "--unfreeze-layer4",
    "--crop-hitter",
    "--crop-padding", "0.55",
    "--amp",
    "--batch-size", "4",
    "--grad-accum-steps", "6",
    "--learning-rate", "1e-4",
    "--backbone-learning-rate", "1e-5",
    "--stroke-class-boost", "drive=2.5", "net_attack=1.5",
    "--epochs", "12",
    "--output-dir", $outputDir
)

New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
& $python @arguments
exit $LASTEXITCODE

