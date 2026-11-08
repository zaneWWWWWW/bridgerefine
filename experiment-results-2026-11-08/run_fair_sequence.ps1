$ErrorActionPreference = "Continue"

$RepoRoot = "E:\tyllltl\Cross-modal conversion"
$Python = "E:\tyllltl\envs\bridge\python.exe"
$ResultRoot = Join-Path $RepoRoot "experiments\ordered_supplement_20261030\results\stage1_fair_training"
$CoordinatorLog = Join-Path $ResultRoot "fair_sequence.log"
$ExitFile = Join-Path $ResultRoot "fair_sequence_exit_code.txt"

New-Item -ItemType Directory -Path $ResultRoot -Force | Out-Null
Remove-Item -LiteralPath $ExitFile -Force -ErrorAction SilentlyContinue
$env:BASELINE_VRAM_FRACTION = "1.0"
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"

function Write-Log([string]$Message) {
    $Line = "$(Get-Date -Format o) $Message"
    Add-Content -LiteralPath $CoordinatorLog -Value $Line -Encoding utf8
}

$Runs = @(
    @{ Name = "ct_long"; Epochs = "100"; Patience = "12"; Batch = "64" },
    @{ Name = "ct_strong"; Epochs = "50"; Patience = "10"; Batch = "16" },
    @{ Name = "joint_gan"; Epochs = "20"; Patience = "6"; Batch = "16" }
)

$RelativeTrainingScript = "experiments\ordered_supplement_20261030\stage1_fair_training.py"

foreach ($Run in $Runs) {
    $RunOut = Join-Path $ResultRoot $Run.Name
    New-Item -ItemType Directory -Path $RunOut -Force | Out-Null
    $Stdout = Join-Path $RunOut "stdout.log"
    $Stderr = Join-Path $RunOut "stderr.log"
    Write-Log "Starting $($Run.Name)"
    $Arguments = @(
        $RelativeTrainingScript,
        "--experiment", $Run.Name,
        "--root", "`"$RepoRoot`"",
        "--out", "`"$ResultRoot`"",
        "--epochs", $Run.Epochs,
        "--patience", $Run.Patience,
        "--batch-size", $Run.Batch
    )
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

Write-Log "Fair sequence complete."
Set-Content -LiteralPath $ExitFile -Value 0 -Encoding ascii
