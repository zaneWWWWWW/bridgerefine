$ErrorActionPreference = "Continue"

$RepoRoot = "E:\tyllltl\Cross-modal conversion\official_selfrdb"
$Python = "E:\tyllltl\envs\bridge\python.exe"
$RunRoot = Join-Path $RepoRoot "runs\official_resume_e19"
$Checkpoint = Join-Path $RepoRoot "logs\official_cached_seed42_b8_e20\version_0\checkpoints\epoch=18-step=129694.ckpt"
$OutLog = Join-Path $RunRoot "stdout.log"
$ErrLog = Join-Path $RunRoot "stderr.log"
$ExitFile = Join-Path $RunRoot "exit_code.txt"

New-Item -ItemType Directory -Path $RunRoot -Force | Out-Null
Remove-Item -LiteralPath $ExitFile -Force -ErrorAction SilentlyContinue
Set-Location $RepoRoot
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:128"

$Arguments = @(
    "main.py",
    "fit",
    "--config",
    "config_brain_cached.yaml",
    "--data.train_batch_size=4",
    "--data.val_batch_size=4",
    "--data.num_workers=0",
    "--trainer.max_epochs=20",
    "--trainer.logger.name",
    "official_cached_seed42_resume_e19",
    "--ckpt_path=`"$Checkpoint`""
)

$Process = Start-Process `
    -FilePath $Python `
    -ArgumentList $Arguments `
    -WorkingDirectory $RepoRoot `
    -RedirectStandardOutput $OutLog `
    -RedirectStandardError $ErrLog `
    -Wait `
    -PassThru
Set-Content -LiteralPath $ExitFile -Value $Process.ExitCode -Encoding ascii
exit $Process.ExitCode
