#!/usr/bin/env bash
# Is the box fit to run or measure the model? Prints what squeezed the night of 2026-09-30: VMs and WSL (commit
# charge), free RAM against the 94 GB of mmap'd model, CPU hogs next to the six expert threads, desktop VRAM,
# and whether a server already holds port 5000. Run before a launcher start and before any A/B.
powershell -NoProfile -Command '
$os = Get-CimInstance Win32_OperatingSystem
$free = $os.FreePhysicalMemory/1MB; $commit = ($os.TotalVirtualMemorySize-$os.FreeVirtualMemory)/1MB
"RAM: free {0:N1} GB, commit charge {1:N1} GB (the model needs ~70 GB of page cache for the experts to stay resident; the box has ~128 GB)" -f $free, $commit
if ($free -lt 75) { "  WARNING: below ~75 GB free the experts fall out of the page cache and prefill drops to ~20 t/s" }
try { Get-VM -ErrorAction Stop | Where-Object State -eq Running | ForEach-Object { "VM running: {0} ({1:N0} GB assigned): stop it before measuring" -f $_.Name, ($_.MemoryAssigned/1GB) } } catch {}
$wsl = (wsl.exe --list --running 2>$null | Out-String) -replace "\x00",""
if ($wsl -match "NixOS|Ubuntu|Debian") { "WSL running: " + (($wsl -split "`n" | Select-Object -Skip 1 | Where-Object { $_.Trim() }) -join ", ").Trim() }
$a = Get-Process | Select-Object Name,Id,@{n="cpu";e={$_.CPU}}; Start-Sleep -Seconds 3
$b = @{}; Get-Process | ForEach-Object { $b[$_.Id] = $_.CPU }
$hogs = $a | ForEach-Object { if ($b.ContainsKey($_.Id) -and $null -ne $_.cpu) { [pscustomobject]@{Name=$_.Name; Cores=[math]::Round(($b[$_.Id]-$_.cpu)/3,2)} } } | Where-Object { $_.Cores -gt 0.3 -and $_.Name -ne "llama-server" -and $_.Name -ne "powershell" } | Sort-Object Cores -Descending
if ($hogs) { "CPU hogs (cores busy): " + (($hogs | ForEach-Object { "{0} {1}" -f $_.Name, $_.Cores }) -join ", ") + "  (each core lost costs the six expert threads ~10-25% of decode)" } else { "CPU: quiet" }
'
echo "VRAM: $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ') (the 120k launcher wants <= 1 GB of desktop use on the 4060 Ti, else LLAMA_HOT_EXPERTS_MIB=5000)"
if curl -s -m 2 http://127.0.0.1:5000/health 2>/dev/null | grep -q '"ok"'; then echo "port 5000: a server is running"; else echo "port 5000: free"; fi
