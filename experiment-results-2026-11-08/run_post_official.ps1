$ErrorActionPreference = "Continue"

$RepoRoot = "E:\tyllltl\Cross-modal conversion"
$Python = "E:\tyllltl\envs\bridge\python.exe"
$TrainingScript = Join-Path $RepoRoot "experiments\ordered_supplement_20261030\stage1_fair_training.py"
$ResultRoot = Join-Path $RepoRoot "experiments\ordered_supplement_20261030\results\stage1_fair_training"
$OfficialTask = "BridgeRefine-Official-SelfRDB"
$OfficialExitFile = Join-Path $RepoRoot "official_selfrdb\runs\official_cached_seed42\exit_code.txt"
$CoordinatorLog = Join-Path $ResultRoot "coordinator.log"

New-Item -ItemType Directory -Path $ResultRoot -Force | Out-Null
$env:BASELINE_VRAM_FRACTION = "1.0"

function Write-Log([string]$Message) {
    $Line = "$(Get-Date -Format o) $Message"
    Add-Content -LiteralPath $CoordinatorLog -Value $Line -Encoding utf8
}

Write-Log "Waiting for $OfficialTask"
while ($true) {
    $Task = Get-ScheduledTask -TaskName $OfficialTask -ErrorAction SilentlyContinue
    $ExitCode = $null
    if (Test-Path -LiteralPath $OfficialExitFile) {
        $ExitCode = [int](Get-Content -LiteralPath $OfficialExitFile)
    }
    if ($Task -and $Task.State -eq "Running") {
        Start-Sleep -Seconds 60
        continue
    }
    if ($null -ne $ExitCode) {
        break
    }
    Start-Sleep -Seconds 60
}

Write-Log "Official SelfRDB exit code: $ExitCode"
if ($ExitCode -ne 0) {
    Write-Log "Official SelfRDB failed; stage 1 will not start."
    exit 1
}

$Runs = @(
    @{ Name = "ct_long"; Epochs = "100"; Patience = "12"; Batch = "64" },
    @{ Name = "ct_strong"; Epochs = "50"; Patience = "10"; Batch = "16" },
    @{ Name = "joint_gan"; Epochs = "20"; Patience = "6"; Batch = "16" }
)

foreach ($Run in $Runs) {
    $RunOut = Join-Path $ResultRoot $Run.Name
    New-Item -ItemType Directory -Path $RunOut -Force | Out-Null
    $Stdout = Join-Path $RunOut "stdout.log"
    $Stderr = Join-Path $RunOut "stderr.log"
    Write-Log "Starting $($Run.Name)"
    $Arguments = @(
        $TrainingScript,
        "--experiment", $Run.Name,
        "--root", $RepoRoot,
        "--out", $ResultRoot,
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
        exit $Process.ExitCode
    }
}

Write-Log "Stage 1 complete."
