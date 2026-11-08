$ErrorActionPreference = "Continue"

$RepoRoot = "E:\tyllltl\Cross-modal conversion"
$Python = "E:\tyllltl\envs\bridge\python.exe"
$ResultRoot = Join-Path $RepoRoot "experiments\ordered_supplement_20261030\results\stage1_fair_training"
$CoordinatorLog = Join-Path $ResultRoot "fair_resume_sequence.log"
$ExitFile = Join-Path $ResultRoot "fair_resume_sequence_exit_code.txt"
$RelativeTrainingScript = "experiments\ordered_supplement_20261030\stage1_fair_training.py"

New-Item -ItemType Directory -Path $ResultRoot -Force | Out-Null
Remove-Item -LiteralPath $ExitFile -Force -ErrorAction SilentlyContinue
$env:BASELINE_VRAM_FRACTION = "1.0"
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"

function Write-Log([string]$Message) {
    $Line = "$(Get-Date -Format o) $Message"
    Add-Content -LiteralPath $CoordinatorLog -Value $Line -Encoding utf8
}

$CtLongDir = Join-Path $ResultRoot "ct_long"
Copy-Item -LiteralPath (Join-Path $CtLongDir "best.pt") `
    -Destination (Join-Path $CtLongDir "best_pre_resume.pt") -Force
foreach ($Name in @("train_log.csv", "stdout.log", "stderr.log")) {
    $Source = Join-Path $CtLongDir $Name
    if (Test-Path -LiteralPath $Source) {
        Move-Item -LiteralPath $Source `
            -Destination (Join-Path $CtLongDir ("pre_resume_" + $Name)) -Force
    }
}

$Runs = @(
    @{ Name = "ct_long"; Epochs = "100"; Patience = "12"; Batch = "64"; Resume = $true },
    @{ Name = "ct_strong"; Epochs = "50"; Patience = "10"; Batch = "16"; Resume = $false },
    @{ Name = "joint_gan"; Epochs = "20"; Patience = "6"; Batch = "16"; Resume = $false }
)

foreach ($Run in $Runs) {
    $RunOut = Join-Path $ResultRoot $Run.Name
    New-Item -ItemType Directory -Path $RunOut -Force | Out-Null
    $Stdout = Join-Path $RunOut "stdout.log"
    $Stderr = Join-Path $RunOut "stderr.log"
    $Arguments = @(
        $RelativeTrainingScript,
        "--experiment", $Run.Name,
        "--root", "`"$RepoRoot`"",
        "--out", "`"$ResultRoot`"",
        "--epochs", $Run.Epochs,
        "--patience", $Run.Patience,
        "--batch-size", $Run.Batch
    )
    if ($Run.Resume) {
        $Arguments += @(
            "--resume",
            "`"$(Join-Path $RunOut 'best_pre_resume.pt')`""
        )
    }
    Write-Log "Starting $($Run.Name)"
    $Process = Start-Process `
        -FilePath $Python `
        -ArgumentList $Arguments `
        -WorkingDirectory $RepoRoot `
        -RedirectStandardOutput $Stdout `
        -RedirectStandardError $Stderr `
        -Wait `
        -PassThru
    Write-Log "$($Run.Name) exit code: $($Process.ExitCode)"
    if ($Process.ExitCode -ne 0) {
        Set-Content -LiteralPath $ExitFile -Value $Process.ExitCode -Encoding ascii
        exit $Process.ExitCode
    }
}

Write-Log "Fair resume sequence complete."
Set-Content -LiteralPath $ExitFile -Value 0 -Encoding ascii
