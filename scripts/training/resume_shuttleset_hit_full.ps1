$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$outputDir = Join-Path $repoRoot "work_dirs\r2plus1d18_hit_full"
$latest = Join-Path $outputDir "latest.pth"
$pilot = Join-Path $repoRoot "work_dirs\r2plus1d18_shuttleset_hit_pilot\best.pth"
$cache = "C:\Users\ADMIN\Downloads\AI_Classifier_cache\shuttleset_hit_full"

$arguments = @(
    "-u",
    (Join-Path $repoRoot "scripts\training\train_shuttleset_hit_rgb_full.py"),
    "--manifest", (Join-Path $repoRoot "data\manifests\shuttleset_rgb.csv"),
    "--source-checkpoint", $(if (Test-Path -LiteralPath $latest) { $latest } else { $pilot }),
    "--cache-dir", $cache,
    "--output-dir", $outputDir,
    "--epochs", "5",
    "--batch-size", "4",
    "--workers", "0"
)

if (Test-Path -LiteralPath $latest) {
    $arguments += "--resume"
    Write-Host "Resuming from $latest"
} else {
    Write-Host "No latest checkpoint yet; warm-starting from $pilot"
}

& python @arguments
exit $LASTEXITCODE
