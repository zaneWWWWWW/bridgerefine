$ErrorActionPreference = "Stop"

$Source = "E:\tyllltl\Cross-modal conversion\experiments\ordered_supplement_20261030"
$Destination = "E:\tyllltl\Cross-modal conversion\tmp\experiment-results-export-20261108"
$CvSource = "E:\tyllltl\Cross-modal conversion\experiments\cv_strict"
$OfficialSource = "E:\tyllltl\Cross-modal conversion\official_selfrdb"

New-Item -ItemType Directory -Path $Destination -Force | Out-Null
$AllowedExtensions = @(".json", ".csv", ".log", ".md", ".py", ".ps1", ".txt")

Get-ChildItem -LiteralPath $Source -Recurse -File |
    Where-Object { $AllowedExtensions -contains $_.Extension.ToLowerInvariant() } |
    ForEach-Object {
        $Relative = $_.FullName.Substring($Source.Length).TrimStart("\")
        $Target = Join-Path $Destination $Relative
        New-Item -ItemType Directory -Path (Split-Path $Target) -Force | Out-Null
        Copy-Item -LiteralPath $_.FullName -Destination $Target -Force
    }

$CvTarget = Join-Path $Destination "cv_strict"
New-Item -ItemType Directory -Path $CvTarget -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $CvSource "cv_strict_summary.json") `
    -Destination (Join-Path $CvTarget "cv_strict_summary.json") -Force
Get-ChildItem -LiteralPath $CvSource -Directory |
    ForEach-Object {
        $Result = Join-Path $_.FullName "result.json"
        if (Test-Path -LiteralPath $Result) {
            $FoldTarget = Join-Path $CvTarget $_.Name
            New-Item -ItemType Directory -Path $FoldTarget -Force | Out-Null
            Copy-Item -LiteralPath $Result -Destination (Join-Path $FoldTarget "result.json") -Force
        }
    }

$OfficialTarget = Join-Path $Destination "reproduction\official_selfrdb"
New-Item -ItemType Directory -Path $OfficialTarget -Force | Out-Null
$OfficialRelativeFiles = @(
    "datasets.py",
    "config_brain_cached.yaml",
    "generate_cached_priors.py",
    "run_official_resume.ps1",
    "run_cached_task_worker.ps1",
    "backbones\op\fused_act.py",
    "backbones\op\upfirdn2d.py"
)
foreach ($Relative in $OfficialRelativeFiles) {
    $SourcePath = Join-Path $OfficialSource $Relative
    if (Test-Path -LiteralPath $SourcePath) {
        $TargetPath = Join-Path $OfficialTarget $Relative
        New-Item -ItemType Directory -Path (Split-Path $TargetPath) -Force | Out-Null
        Copy-Item -LiteralPath $SourcePath -Destination $TargetPath -Force
    }
}

$Metadata = [ordered]@{
    exported_at = (Get-Date).ToString("o")
    source_root = $Source
    included_extensions = $AllowedExtensions
    excluded_patterns = @("*.npy", "*.pt", "*.ckpt", "images", "raw predictions")
    official_selfrdb_commit = "6262ede"
    official_checkpoint = "logs/official_cached_seed42_b8_e20/version_0/checkpoints/epoch=18-step=129694.ckpt"
    note = "Large checkpoints and arrays remain on the experiment host."
}
$Metadata | ConvertTo-Json -Depth 4 |
    Set-Content -LiteralPath (Join-Path $Destination "EXPORT_MANIFEST.json") -Encoding utf8

Get-ChildItem -LiteralPath $Destination -Recurse -File |
    Measure-Object -Property Length -Sum |
    Select-Object Count, Sum
