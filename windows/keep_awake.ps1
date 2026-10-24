# Keep Windows from going to sleep while training runs. The screen may still turn off.
# Windows lets a program ask "the system is in use" (SetThreadExecutionState); the request
# lasts while this script runs, so stop it (Ctrl+C or close the window) to restore normal sleep.
# Closing the lid still follows the lid setting in Power Options.
#
#   powershell -ExecutionPolicy Bypass -File windows\keep_awake.ps1 -Hours 14
param([double]$Hours = 14)

Add-Type -Namespace Win32 -Name Power -MemberDefinition @'
[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);
'@
# ES_CONTINUOUS (0x80000000) | ES_SYSTEM_REQUIRED (0x1), written in decimal because
# Windows PowerShell reads 0x80000000 as a negative int
$flags = [uint32]2147483649
$until = (Get-Date).AddHours($Hours)
Write-Host "keep_awake: holding the system awake until $until"
try {
    while ((Get-Date) -lt $until) {
        [void][Win32.Power]::SetThreadExecutionState($flags)
        Start-Sleep -Seconds 60
    }
} finally {
    [void][Win32.Power]::SetThreadExecutionState([uint32]2147483648)  # ES_CONTINUOUS alone: release
    Write-Host "keep_awake: released at $(Get-Date)"
}
