$ErrorActionPreference = "Continue"

$RepoRoot = "E:\tyllltl\Cross-modal conversion"
$Python = "E:\tyllltl\envs\bridge\python.exe"
$ProbeCheckpoint = Join-Path $RepoRoot "official_selfrdb\logs\official_cached_seed42_b8_e20\version_0\checkpoints\epoch=18-step=129694.ckpt"
$PriorDir = Join-Path $RepoRoot "official_selfrdb\prior_cache"
$ResultRoot = Join-Path $RepoRoot "experiments\ordered_supplement_20261030\results\stage1_official_refiner"
$CoordinatorLog = Join-Path $ResultRoot "sequence.log"
$ExitFile = Join-Path $ResultRoot "exit_code.txt"
$GenerateScript = "official_selfrdb\generate_cached_priors.py"
$TrainScript = "experiments\ordered_supplement_20261030\stage1_official_refiner.py"

New-Item -ItemType Directory -Path $PriorDir -Force | Out-Null
New-Item -ItemType Directory -Path $ResultRoot -Force | Out-Null
Remove-Item -LiteralPath $ExitFile -Force -ErrorAction SilentlyContinue
$env:BASELINE_VRAM_FRACTION = "1.0"
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:128"

function Write-Log([string]$Message) {
    Add-Content -LiteralPath $CoordinatorLog `
        -Value "$(Get-Date -Format o) $Message" -Encoding utf8
}

$ExistingTest = Join-Path $RepoRoot "experiments\ordered_supplement_20261030\results\stage1_official_prior\official_test.npy"
$ExistingNames = Join-Path $RepoRoot "experiments\ordered_supplement_20261030\results\stage1_official_prior\official_test_names.json"
Copy-Item -LiteralPath $ExistingTest -Destination (Join-Path $PriorDir "official_test.npy") -Force
Copy-Item -LiteralPath $ExistingNames -Destination (Join-Path $PriorDir "official_test_names.json") -Force

$GenerateArgs = @(
    $GenerateScript,
    "--root", "`"$RepoRoot`"",
    "--checkpoint", "`"$ProbeCheckpoint`"",
    "--out", "`"$PriorDir`"",
    "--splits", "train", "val",
    "--batch-size", "8"
)
Write-Log "Generating official train/validation priors"
$GenerateLog = Join-Path $ResultRoot "generation.log"
$Generate = Start-Process `
    -FilePath $Python `
    -ArgumentList $GenerateArgs `
    -WorkingDirectory $RepoRoot `
    -RedirectStandardOutput $GenerateLog `
    -RedirectStandardError (Join-Path $ResultRoot "generation_err.log") `
    -Wait `
    -PassThru
Write-Log "Generation exit code: $($Generate.ExitCode)"
if ($Generate.ExitCode -ne 0) {
    Set-Content -LiteralPath $ExitFile -Value $Generate.ExitCode -Encoding ascii
    exit $Generate.ExitCode
}

foreach ($Seed in @(42, 123, 2026)) {
    $RunDir = Join-Path $ResultRoot "seed$Seed"
    New-Item -ItemType Directory -Path $RunDir -Force | Out-Null
    $Arguments = @(
        $TrainScript,
        "--root", "`"$RepoRoot`"",
        "--prior-dir", "`"$PriorDir`"",
        "--out", "`"$ResultRoot`"",
        "--seed", "$Seed",
        "--epochs", "20",
        "--batch-size", "16",
        "--patience", "6"
    )
    Write-Log "Training official-prior refiner seed $Seed"
    $Process = Start-Process `
        -FilePath $Python `
        -ArgumentList $Arguments `
        -WorkingDirectory $RepoRoot `
        -RedirectStandardOutput (Join-Path $RunDir "stdout.log") `
        -RedirectStandardError (Join-Path $RunDir "stderr.log") `
        -Wait `
        -PassThru
    Write-Log "Seed $Seed exit code: $($Process.ExitCode)"
    if ($Process.ExitCode -ne 0) {
        Set-Content -LiteralPath $ExitFile -Value $Process.ExitCode -Encoding ascii
        exit $Process.ExitCode
    }
}

Write-Log "Official-prior refiner sequence complete."
Set-Content -LiteralPath $ExitFile -Value 0 -Encoding ascii
