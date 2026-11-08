$ErrorActionPreference = "Continue"

$RepoRoot = "E:\tyllltl\Cross-modal conversion"
$Python = "E:\tyllltl\envs\bridge\python.exe"
$ResultRoot = Join-Path $RepoRoot "experiments\ordered_supplement_20261030\results"
$CoordinatorLog = Join-Path $ResultRoot "remaining_sequence.log"
$ExitFile = Join-Path $ResultRoot "remaining_sequence_exit_code.txt"
$OfficialRefinerExit = Join-Path $ResultRoot "stage1_official_refiner\exit_code.txt"

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

function Wait-OfficialRefiner {
    Write-Log "Waiting for official-prior refiner"
    while ($true) {
        $task = Get-ScheduledTask -TaskName "BridgeRefine-Official-Refiner" -ErrorAction SilentlyContinue
        if ($task -and $task.State -eq "Running") {
            Start-Sleep -Seconds 60
            continue
        }
        if (Test-Path -LiteralPath $OfficialRefinerExit) {
            return [int](Get-Content -LiteralPath $OfficialRefinerExit)
        }
        Start-Sleep -Seconds 60
    }
}

function Invoke-Experiment(
    [string]$Name,
    [string]$Script,
    [string[]]$Arguments
) {
    $runDir = Join-Path $ResultRoot $Name
    New-Item -ItemType Directory -Path $runDir -Force | Out-Null
    $fullArguments = @($Script) + $Arguments
    Write-Log "Starting $Name"
    $process = Start-Process `
        -FilePath $Python `
        -ArgumentList $fullArguments `
        -WorkingDirectory $RepoRoot `
        -RedirectStandardOutput (Join-Path $runDir "stdout.log") `
        -RedirectStandardError (Join-Path $runDir "stderr.log") `
        -Wait `
        -PassThru
    Write-Log "$Name exit code: $($process.ExitCode)"
    if ($process.ExitCode -ne 0) {
        Set-Content -LiteralPath $ExitFile -Value $process.ExitCode -Encoding ascii
        exit $process.ExitCode
    }
}

$officialExit = Wait-OfficialRefiner
Write-Log "Official-prior refiner exit code: $officialExit"
if ($officialExit -ne 0) {
    Set-Content -LiteralPath $ExitFile -Value $officialExit -Encoding ascii
    exit $officialExit
}

Invoke-Experiment "stage2_mechanism_fixed" `
    "experiments\ordered_supplement_20261030\stage2_mechanism.py" @(
        "--root", "`"$RepoRoot`"",
        "--out", "`"$(Join-Path $ResultRoot 'stage2_mechanism')\fixed`"",
        "--mode", "fixed"
    )

Invoke-Experiment "stage2_mechanism_retrain" `
    "experiments\ordered_supplement_20261030\stage2_mechanism.py" @(
        "--root", "`"$RepoRoot`"",
        "--out", "`"$(Join-Path $ResultRoot 'stage2_mechanism')\retrain`"",
        "--mode", "retrain",
        "--epochs", "20",
        "--batch-size", "16"
    )

Invoke-Experiment "stage2_sampling_retrain" `
    "experiments\ordered_supplement_20261030\stage2_sampling_retrain.py" @(
        "--root", "`"$RepoRoot`"",
        "--out", "`"$(Join-Path $ResultRoot 'stage2_sampling_retrain')`"",
        "--epochs", "20",
        "--batch-size", "16"
    )

Invoke-Experiment "stage3_resolution_256_ct" `
    "experiments\ordered_supplement_20261030\stage3_resolution_256.py" @(
        "--experiment", "ct_only",
        "--root", "`"$RepoRoot`"",
        "--out", "`"$(Join-Path $ResultRoot 'stage3_resolution_256')`"",
        "--epochs", "20",
        "--batch-size", "24"
    )

Invoke-Experiment "stage3_resolution_256_gan" `
    "experiments\ordered_supplement_20261030\stage3_resolution_256.py" @(
        "--experiment", "gan",
        "--root", "`"$RepoRoot`"",
        "--out", "`"$(Join-Path $ResultRoot 'stage3_resolution_256')`"",
        "--epochs", "20",
        "--batch-size", "4"
    )

Invoke-Experiment "stage3_roi3d" `
    "experiments\ordered_supplement_20261030\stage2_roi3d.py" @(
        "--root", "`"$RepoRoot`"",
        "--out", "`"$(Join-Path $ResultRoot 'stage3_roi3d')`""
    )

Invoke-Experiment "stage3_final_statistics" `
    "experiments\ordered_supplement_20261030\stage3_final_aggregate.py" @(
        "--root", "`"$RepoRoot`"",
        "--out", "`"$(Join-Path $ResultRoot 'stage3_final_statistics')`""
    )

$OfficialRefiner = Join-Path $ResultRoot "stage1_official_refiner\seed42\best.pt"
$OfficialCheckpoint = Join-Path $RepoRoot "official_selfrdb\logs\official_cached_seed42_b8_e20\version_0\checkpoints\epoch=18-step=129694.ckpt"
Invoke-Experiment "stage3_official_efficiency" `
    "experiments\ordered_supplement_20261030\stage3_official_efficiency.py" @(
        "--root", "`"$RepoRoot`"",
        "--checkpoint", "`"$OfficialCheckpoint`"",
        "--refiner", "`"$OfficialRefiner`"",
        "--out", "`"$(Join-Path $ResultRoot 'stage3_official_efficiency')`""
    )

Invoke-Experiment "stage3_cv_strict" `
    "baselines\cv_strict.py" @(
        "--folds", "0,1,2,3,4"
    )

Write-Log "Remaining experiment sequence complete."
Set-Content -LiteralPath $ExitFile -Value 0 -Encoding ascii
