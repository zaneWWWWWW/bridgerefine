$ErrorActionPreference = "Continue"

$RepoRoot = "E:\tyllltl\Cross-modal conversion"
$Python = "E:\tyllltl\envs\bridge\python.exe"
$ResultRoot = Join-Path $RepoRoot "experiments\ordered_supplement_20261030\results\stage3_cv_strict"
$LogFile = Join-Path $ResultRoot "stdout.log"
$ErrFile = Join-Path $ResultRoot "stderr.log"
$ExitFile = Join-Path $ResultRoot "exit_code.txt"

New-Item -ItemType Directory -Path $ResultRoot -Force | Out-Null
Remove-Item -LiteralPath $ExitFile -Force -ErrorAction SilentlyContinue
$env:BASELINE_VRAM_FRACTION = "1.0"
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:128"

$Arguments = @(
    "baselines\cv_strict.py",
    "--folds", "0,1,2,3,4"
)
$Process = Start-Process `
    -FilePath $Python `
    -ArgumentList $Arguments `
    -WorkingDirectory $RepoRoot `
    -RedirectStandardOutput $LogFile `
    -RedirectStandardError $ErrFile `
    -Wait `
    -PassThru
Set-Content -LiteralPath $ExitFile -Value $Process.ExitCode -Encoding ascii
exit $Process.ExitCode
