param([switch]$WhatIf)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$relativeTargets = @(
    ".dist_test",
    ".pytest_cache",
    "external/badminton_shot_type/venv",
    "external/badminton_shot_type",
    "external/BST-Badminton-Stroke-type-Transformer/.venv",
    "notebooks/Run_BST_Colab.ipynb",
    "outputs/archive_joint_correct_lift_supplement",
    "outputs/archive_video_audit",
    "outputs/archive_upload_pipeline_24",
    "outputs/archive_upload_rgb20_24",
    "outputs/shuttleset_crop_audit_200",
    "outputs/fine_badminton_preview_audit",
    "outputs/shuttleset_crop_pilot",
    "outputs/bst_052",
    "outputs/bst_073",
    "outputs/bst_fd095",
    "outputs/bst_fd089",
    "outputs/bst_086",
    "outputs/bst_bns088",
    "work_dirs/hit_verifier_cache",
    "work_dirs/hit_verifier_cache_seq",
    "work_dirs/_smoke_shuttleset_mixed_v3_cuda_batch4",
    "work_dirs/r2plus1d18_hit_smoke_test",
    "work_dirs/_smoke_r2plus1d18_shuttleset_hit",
    "work_dirs/shuttleset_fusion_pilot",
    "work_dirs/pipeline_test",
    "work_dirs/r2plus1d18_shuttleset_hit_verifier_v1",
    "work_dirs/r2plus1d18_shuttleset_hit_verifier_smoke",
    "work_dirs/batch_rgb_method1_eval",
    "work_dirs/batch_rgb_expanded_test",
    "work_dirs/batch_rgb_method1_final",
    "work_dirs/batch_rgb_padded_new_model",
    "work_dirs/batch_rgb_padded",
    "work_dirs/batch_rgb_archive",
    "work_dirs/batch_fusion_padded",
    "test_output.json",
    "test_output_089_lower.json",
    "test_output_089_upper.json",
    "test_output_2.json"
)

$targets = foreach ($relative in $relativeTargets) {
    $candidate = Join-Path $repoRoot $relative
    if (Test-Path -LiteralPath $candidate) {
        $resolved = (Resolve-Path -LiteralPath $candidate).Path
        if (-not $resolved.StartsWith($repoRoot, [StringComparison]::OrdinalIgnoreCase)) {
            throw "Refusing path outside repository: $resolved"
        }
        $resolved
    }
}

$cacheTargets = Get-ChildItem -LiteralPath $repoRoot -Directory -Recurse -Force -Filter "__pycache__" |
    ForEach-Object { $_.FullName } |
    Where-Object { $_.StartsWith($repoRoot, [StringComparison]::OrdinalIgnoreCase) }
$targets = @($targets) + @($cacheTargets)

$totalBytes = 0
foreach ($target in $targets) {
    $item = Get-Item -LiteralPath $target -Force
    $size = if ($item.PSIsContainer) {
        (Get-ChildItem -LiteralPath $target -Recurse -File -Force -ErrorAction SilentlyContinue |
            Measure-Object Length -Sum).Sum
    } else {
        $item.Length
    }
    $totalBytes += $size
    if (-not $WhatIf) {
        Remove-Item -LiteralPath $target -Recurse -Force
    }
    Write-Host $(if ($WhatIf) { "Would remove: $target" } else { "Removed: $target" })
}

Write-Host ("Targets: {0}; size: {1:N2} GB" -f $targets.Count, ($totalBytes / 1GB))
