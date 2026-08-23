Write-Output "=== Physical memory ==="
Get-CimInstance Win32_PhysicalMemory | ForEach-Object {
    $cap = [math]::Round($_.Capacity/1GB, 1)
    Write-Output "  $($_.Manufacturer) $($_.PartNumber) ${cap}GB @ $($_.Speed)MT/s slot=$($_.DeviceLocator)"
}
$totalPhys = ((Get-CimInstance Win32_PhysicalMemory).Capacity | Measure-Object -Sum).Sum / 1GB
Write-Output "  Total physical: $([math]::Round($totalPhys,1)) GB"
Write-Output ""

Write-Output "=== Memory state ==="
$os = Get-CimInstance Win32_OperatingSystem
Write-Output "  Total visible (OS-usable):  $([math]::Round($os.TotalVisibleMemorySize/1MB,2)) GB"
Write-Output "  Free (immediately avail):   $([math]::Round($os.FreePhysicalMemory/1MB,2)) GB"
$comp = Get-CimInstance Win32_ComputerSystem
Write-Output "  Total system memory:        $([math]::Round($comp.TotalPhysicalMemory/1GB,2)) GB"
$reservedForHardware = $totalPhys - ($os.TotalVisibleMemorySize/1MB/1024)
Write-Output "  Reserved for hardware/GPU:  $([math]::Round($reservedForHardware,2)) GB  <-- this is shared-GPU + firmware"
Write-Output ""

Write-Output "=== GPU memory ==="
Get-CimInstance Win32_VideoController | ForEach-Object {
    $name = $_.Name
    $adapter = if ($_.AdapterRAM) { [math]::Round($_.AdapterRAM/1GB,2) } else { 0 }
    Write-Output "  $name  AdapterRAM=${adapter}GB (this is dedicated VRAM, ignored if integrated)"
}
Write-Output ""

Write-Output "=== Per-process commit (top 15 by Working Set) ==="
Get-Process | Sort-Object WorkingSet64 -Descending | Select-Object -First 15 | ForEach-Object {
    $mb = [math]::Round($_.WorkingSet64/1MB)
    $priv = [math]::Round($_.PrivateMemorySize64/1MB)
    Write-Output "  $($_.ProcessName) PID=$($_.Id): WS=${mb} MB Private=${priv} MB"
}
Write-Output ""

Write-Output "=== Total process working set ==="
$sumWS = (Get-Process | Measure-Object WorkingSet64 -Sum).Sum / 1MB
$sumPriv = (Get-Process | Measure-Object PrivateMemorySize64 -Sum).Sum / 1MB
Write-Output "  Sum of all process WS:      $([math]::Round($sumWS/1024,2)) GB"
Write-Output "  Sum of all process Private: $([math]::Round($sumPriv/1024,2)) GB"
Write-Output "  System cache + drivers:     unaccounted (kernel pool, modified pages, file cache)"
