# Run pi non-interactively on a task file inside a working directory, with a wall-clock cap.
#   powershell -File customware/run-pi-task.ps1 -WorkDir <dir> -TaskFile <md> -Minutes 25 -Log <file> [-Thinking medium]
param(
    [Parameter(Mandatory=$true)][string]$WorkDir,
    [Parameter(Mandatory=$true)][string]$TaskFile,
    [int]$Minutes = 25,
    [string]$Log = "pi-task.log",
    [string]$Thinking = "medium"
)
$task = Get-Content -Raw $TaskFile
# pi's bash tool spawns $SHELL (falling back to /bin/sh, which does not exist on Windows): point it at Git Bash
if (-not $env:SHELL) { $env:SHELL = "C:/Program Files/Git/usr/bin/bash.exe" }
$args_ = @('-p', '--no-session', '--thinking', $Thinking, $task)
$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = "cmd.exe"
$psi.Arguments = '/c pi ' + (($args_ | ForEach-Object { if ($_ -match '\s') { '"' + ($_ -replace '"', '\"') + '"' } else { $_ } }) -join ' ')
$psi.WorkingDirectory = $WorkDir
$psi.RedirectStandardOutput = $true
$psi.RedirectStandardError = $true
$psi.UseShellExecute = $false
$p = [System.Diagnostics.Process]::Start($psi)
$out = $p.StandardOutput.ReadToEndAsync()
$err = $p.StandardError.ReadToEndAsync()
$start = Get-Date
if (-not $p.WaitForExit($Minutes * 60 * 1000)) {
    "== time cap of $Minutes minutes reached, stopping pi" | Out-File -Append -Encoding utf8 $Log
    Get-CimInstance Win32_Process | Where-Object { $_.ParentProcessId -eq $p.Id } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    $p.Kill()
    $p.WaitForExit(5000) | Out-Null
}
$elapsed = (Get-Date) - $start
"== pi exit code $($p.ExitCode) after $([int]$elapsed.TotalSeconds) s" | Out-File -Append -Encoding utf8 $Log
"== stdout" | Out-File -Append -Encoding utf8 $Log
$out.Result | Out-File -Append -Encoding utf8 $Log
"== stderr" | Out-File -Append -Encoding utf8 $Log
$err.Result | Out-File -Append -Encoding utf8 $Log
