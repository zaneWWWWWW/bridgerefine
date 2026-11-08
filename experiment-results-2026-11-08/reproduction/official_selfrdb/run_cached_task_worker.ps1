$ErrorActionPreference = "Continue"

$RepoRoot = "E:\tyllltl\Cross-modal conversion\official_selfrdb"
$Python = "E:\tyllltl\envs\bridge\python.exe"
$RunRoot = Join-Path $RepoRoot "runs\official_cached_seed42"
$OutLog = Join-Path $RunRoot "stdout.log"
$ErrLog = Join-Path $RunRoot "stderr.log"
$ExitFile = Join-Path $RunRoot "exit_code.txt"

New-Item -ItemType Directory -Path $RunRoot -Force | Out-Null
Set-Location $RepoRoot
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"

$Arguments = @(
    "main.py",
    "fit",
    "--config",
    "config_brain_cached.yaml",
    "--trainer.logger.name",
    "official_cached_seed42_b8_e20"
)

$Process = Start-Process `
    -FilePath $Python `
    -ArgumentList $Arguments `
    -WorkingDirectory $RepoRoot `
    -RedirectStandardOutput $OutLog `
    -RedirectStandardError $ErrLog `
    -Wait `
    -PassThru
$Code = $Process.ExitCode
Set-Content -LiteralPath $ExitFile -Value $Code -Encoding ascii
exit $Code
