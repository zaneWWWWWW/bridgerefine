$ErrorActionPreference = "Stop"

$RepoRoot = "E:\tyllltl\Cross-modal conversion"
$Worker = Join-Path $RepoRoot "experiments\ordered_supplement_20261030\run_post_official.ps1"
$TaskName = "BridgeRefine-Post-Official"

$Action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$Worker`"" `
    -WorkingDirectory $RepoRoot
$Trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(2)
$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit ([TimeSpan]::Zero)

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Settings $Settings `
    -Force | Out-Null
Start-ScheduledTask -TaskName $TaskName
Get-ScheduledTask -TaskName $TaskName |
    Select-Object TaskName, State
