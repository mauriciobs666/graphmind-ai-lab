<#
gpu_process_memory.ps1 - sample WDDM GPU memory per process + adapter total (K-066).

Re-creation of U0's lost scratch sampler (falkor-chat/docs/test-reports/
salesperson-single-model-fit-report.md, "Spill instrument"; review P3-1).
nvidia-smi cannot see WDDM paging, so this reads the Windows performance counters
  \GPU Process Memory(*)\Dedicated Usage / Shared Usage   (per process)
  \GPU Adapter Memory(*)\Dedicated Usage / Shared Usage   (adapter total)
for ONE adapter LUID, and prints one CSV line per (sample, row) to stdout, flushed,
so a killed run keeps every completed sample.

Rows: adapter total, then every process whose name matches -ProcessPattern
(default: llama-server and LM Studio - the embedder runs inside LM Studio's process,
ministral inside llama-server).

Usage (from WSL; redirect stdout to a file):
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$(wslpath -w gpu_process_memory.ps1)" `
      -IntervalSec 5 [-Count 0] [-Luid 0x00011540] > gpumem.csv
-Count 0 (default) samples until killed. -Luid defaults to the adapter with the
largest Dedicated Usage (the discrete GPU); the LUID changes across reboots - re-check it.
#>
param(
    [int]$IntervalSec = 5,
    [int]$Count = 0,
    [string]$Luid = "",
    [string]$ProcessPattern = "^(llama-server|LM Studio)$"
)

$ErrorActionPreference = "Stop"

if (-not $Luid) {
    $top = (Get-Counter "\GPU Adapter Memory(*)\Dedicated Usage").CounterSamples |
        Sort-Object CookedValue -Descending | Select-Object -First 1
    if ($top.InstanceName -match "luid_0x[0-9a-f]+_(0x[0-9a-f]+)_phys") { $Luid = $Matches[1] }
}
$luidRe = [regex]::Escape($Luid)

[Console]::Out.WriteLine("ts,scope,pid,process,dedicated_mib,shared_mib,luid")
[Console]::Out.Flush()

$n = 0
while ($Count -eq 0 -or $n -lt $Count) {
    $ts = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ss.fffzzz")
    $samples = (Get-Counter -Counter @(
        "\GPU Process Memory(*)\Dedicated Usage", "\GPU Process Memory(*)\Shared Usage",
        "\GPU Adapter Memory(*)\Dedicated Usage", "\GPU Adapter Memory(*)\Shared Usage"
    ) -ErrorAction SilentlyContinue).CounterSamples

    $adapter = @{ ded = 0; sh = 0 }
    $procs = @{}
    foreach ($s in $samples) {
        if ($s.InstanceName -notmatch $luidRe) { continue }
        $isShared = $s.Path -match "shared usage$"
        if ($s.Path -match "gpu adapter memory") {
            if ($isShared) { $adapter.sh += $s.CookedValue } else { $adapter.ded += $s.CookedValue }
        } elseif ($s.InstanceName -match "^pid_(\d+)_") {
            $procId = [int]$Matches[1]
            if (-not $procs.ContainsKey($procId)) { $procs[$procId] = @{ ded = 0; sh = 0 } }
            if ($isShared) { $procs[$procId].sh += $s.CookedValue } else { $procs[$procId].ded += $s.CookedValue }
        }
    }
    $mib = 1048576.0
    [Console]::Out.WriteLine(("{0},adapter,,,{1:F0},{2:F0},{3}" -f $ts, ($adapter.ded / $mib), ($adapter.sh / $mib), $Luid))
    foreach ($procId in $procs.Keys) {
        $p = Get-Process -Id $procId -ErrorAction SilentlyContinue
        if (-not $p -or $p.ProcessName -notmatch $ProcessPattern) { continue }
        $v = $procs[$procId]
        if ($v.ded -eq 0 -and $v.sh -eq 0) { continue }
        [Console]::Out.WriteLine(("{0},process,{1},{2},{3:F0},{4:F0},{5}" -f $ts, $procId, $p.ProcessName, ($v.ded / $mib), ($v.sh / $mib), $Luid))
    }
    [Console]::Out.Flush()
    $n++
    if ($Count -eq 0 -or $n -lt $Count) { Start-Sleep -Seconds $IntervalSec }
}
